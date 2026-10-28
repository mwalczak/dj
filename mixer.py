"""Beat-matched auto-mix of the sorted set into one mp3.

Per transition:
  * cue points snap to an 8-bar phrase grid (skipping silent/spoken intros and fade-outs)
  * the outgoing track ramps its tempo to the incoming BPM over 16 bars, then is
    time-mapped beat-for-beat onto the incoming track's beats during the blend (rubberband)
  * blend = 16 bars (8 on key clash / big tempo jump): highs crossfade, bass swaps at the midpoint
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfiltfilt

SR = 44100
PHRASE = 32  # beats per phrase-grid step (8 bars)
RAMP_BEATS = 64  # 16 bars of tempo ramp before each blend
BASS_HZ = 150
TARGET_LUFS = -9.0
CUT_JUMP = 0.08  # tempo jumps above this get an echo-out cut instead of a beat-matched blend
ECHOES = 4

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "beats"
RUBBERBAND = ROOT / "tools" / "rubberband"


def ffmpeg() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        [ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def beats_of(name: str, mono: np.ndarray) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha1(name.encode()).hexdigest()[:16] + ".npy")
    if f.exists():
        return np.load(f)
    import essentia.standard as es

    _, beats, _, _, _ = es.RhythmExtractor2013(method="multifeature")(mono.astype(np.float32))
    b = np.asarray(beats, dtype=np.float64)
    np.save(f, b)
    return b


@dataclass
class Track:
    name: str
    meta: dict[str, Any]
    audio: np.ndarray  # (n, 2) gain-matched
    beats: np.ndarray  # seconds
    start_b: int  # first musically active beat
    end_b: int  # last musically active beat
    phase: int  # phrase-grid offset in beats
    kick: np.ndarray  # bool per beat: kick drum on the beat

    def kick_frac(self, a: int, b: int) -> float:
        k = self.kick[max(0, a) : max(0, b)]
        return float(k.mean()) if len(k) else 0.0

    def kick_start(self) -> int | None:
        """First phrase-grid beat after which the kick runs for a phrase."""
        g = self.grid_at_or_after(self.start_b)
        while g + PHRASE <= self.end_b:
            if self.kick_frac(g, g + PHRASE) >= 0.6:
                return g
            g += PHRASE
        return None

    def kick_end(self) -> int | None:
        """Last beat before which the kick ran for a phrase."""
        b = self.end_b
        while b - PHRASE >= self.start_b:
            if self.kick_frac(b - PHRASE, b) >= 0.6:
                return b
            b -= 4
        return None

    def grid_at_or_after(self, b: int) -> int:
        return b + (self.phase - b) % PHRASE

    def grid_at_or_before(self, b: int) -> int:
        return b - (b - self.phase) % PHRASE

    def frame(self, b: int) -> int:
        return round(self.beats[b] * SR)


_KICK = butter(4, 120, btype="low", fs=SR, output="sos")


def kick_mask(low: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Beat has a kick if low-band energy right on the beat clearly beats the off-beat."""
    p = float(np.median(np.diff(beats)))
    w = int(0.05 * SR)

    def energy(times: np.ndarray) -> np.ndarray:
        return np.array([low[int(x * SR) : int(x * SR) + w].mean() if int(x * SR) + w <= len(low) else 0.0
                         for x in times])

    ratio = energy(beats) / (energy(beats + p / 2) + 1e-9)
    return np.convolve(ratio > 1.4, np.ones(4) / 4, mode="same") >= 0.5


def regular_grid(beats: np.ndarray, kick: np.ndarray, duration: float) -> np.ndarray:
    """Constant-tempo beat grid fitted to the kick beats (tracker output wanders in beatless parts)."""
    b = beats[kick] if kick.sum() >= 64 else beats
    p = float(np.median(np.diff(beats)))
    t0 = float(b[0])
    for _ in range(3):
        k = np.round((b - b[0]) / p)
        (p, t0), *_ = np.linalg.lstsq(np.vstack([k, np.ones_like(k)]).T, b, rcond=None)
        res = b - (t0 + p * k)
        b = b[np.abs(res) < 0.03]
    if len(b) < 32 or np.std(res[np.abs(res) < 0.03]) > 0.02:
        return beats  # tempo really drifts: keep tracker beats
    t0 -= np.floor(t0 / p) * p
    return t0 + p * np.arange(int((duration - t0) / p))


def snap_to_kicks(beats: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Shift a regular grid so it sits on the kick attacks (trackers sometimes lock between kicks)."""
    p = float(np.median(np.diff(beats)))
    w = int(0.01 * SR)
    env = np.convolve(low, np.ones(w) / w, mode="same")
    onset = np.maximum(0, np.diff(env, prepend=env[0]))
    onset = np.convolve(onset, np.ones(2 * w), mode="same")  # tolerate +-10 ms
    offsets = np.arange(0, p, 0.004)
    scores = []
    for d in offsets:
        idx = ((beats + d) * SR).astype(int)
        scores.append(onset[idx[idx < len(onset)]].sum())
    scores_a = np.array(scores)
    best = int(np.argmax(scores_a))
    if scores_a[best] < 1.3 * np.median(scores_a):
        return beats  # no clear kick: leave as is
    d = offsets[best] if offsets[best] <= p / 2 else offsets[best] - p
    shifted = beats + d
    return shifted[shifted >= 0]


def load_track(name: str, path: Path, meta: dict[str, Any]) -> Track:
    audio = decode(path) * 10 ** ((TARGET_LUFS - meta["lufs"]) / 20)
    mono = audio.mean(axis=1)
    low = np.abs(sosfilt(_KICK, mono))
    raw = beats_of(name, mono)
    beats = regular_grid(raw, kick_mask(low, raw), len(mono) / SR - 0.1)
    # grid at the analysed tempo: tracker may have locked to half-time / triplet pulse (drum & bass)
    p = float(np.median(np.diff(beats)))
    r = meta["bpm"] * p / 60
    if abs(r - 1) > 0.05:
        r = min((0.5, 2 / 3, 0.75, 4 / 3, 1.5, 2.0), key=lambda f: abs(f - r))
        beats = beats[0] + (p / r) * np.arange(int((len(mono) / SR - 0.1 - beats[0]) / (p / r)))
    beats = snap_to_kicks(beats, low)
    kick = kick_mask(low, beats)
    idx = (beats * SR).astype(int)
    rms = np.array([np.sqrt(np.mean(mono[a:b] ** 2) + 1e-12) for a, b in itertools.pairwise(idx)])
    active = rms > 0.35 * np.median(rms)

    def first_run(mask: np.ndarray) -> int:  # first index starting 8 consecutive active beats
        for i in range(len(mask) - 8):
            if mask[i : i + 8].all():
                return i
        return 0

    start_b = first_run(active)
    end_b = len(active) - first_run(active[::-1])  # index into beats (end of last active beat)

    # phrase phase: where bar-level energy changes most (elements enter/leave on phrase boundaries)
    bar = np.convolve(rms, np.ones(4) / 4, mode="full")[3:]  # bar[b] = mean rms[b-3..b]
    change = np.zeros_like(rms)
    change[4:] = np.abs(np.log(bar[4:] + 1e-9) - np.log(bar[:-4] + 1e-9))
    change = np.roll(change, -3)  # attribute to the bar's first beat
    scores = [change[p:end_b:PHRASE].sum() for p in range(PHRASE)]
    return Track(name, meta, audio, beats, start_b, end_b, int(np.argmax(scores)), kick)


def key_clash(a: str, b: str) -> bool:
    from djmat import key_cost

    return key_cost(a, b) >= 6


def stretch(seg: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    """Time-map `seg` so each source frame lands on its target frame (rubberband R3 engine)."""
    with tempfile.TemporaryDirectory() as td:
        src, dst, tm = Path(td, "in.wav"), Path(td, "out.wav"), Path(td, "map.txt")
        sf.write(src, seg, SR, subtype="FLOAT")
        inner, last = [], 0
        for s, t in anchors[1:-1]:  # rubberband wants strictly increasing source frames
            if s > last:
                inner.append((s, t))
                last = s
        tm.write_text("".join(f"{s} {t}\n" for s, t in inner))
        total = anchors[-1][1] / SR
        subprocess.run(
            [str(RUBBERBAND), "-q", "--fine", "--timemap", str(tm), "-D", f"{total:.6f}", str(src), str(dst)],
            check=True, capture_output=True,
        )
        out, _ = sf.read(dst, dtype="float32", always_2d=True)
    n = anchors[-1][1]
    return out[:n] if len(out) >= n else np.pad(out, ((0, n - len(out)), (0, 0)))


_LP = butter(2, BASS_HZ, btype="low", fs=SR, output="sos")


def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = sosfiltfilt(_LP, x, axis=0).astype(np.float32)
    return low, x - low


def blend_env(n: int, beat_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gain curves over a blend of n frames: (out_high, out_low, in_high, in_low)."""
    x = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)
    in_high = np.sin(np.pi / 2 * np.clip(x / 0.5, 0, 1))
    out_high = np.cos(np.pi / 2 * np.clip((x - 0.5) / 0.5, 0, 1))
    w = beat_frames / n  # bass swap takes one beat around the midpoint
    in_low = np.clip((x - 0.5 + w / 2) / w, 0, 1)
    return out_high, 1 - in_low, in_high, in_low


def echo_tail(seg: np.ndarray, beat_f: int) -> np.ndarray:
    """Echo-out: the last beat repeated ECHOES times, decaying, bass removed (for cut transitions)."""
    last = seg[-beat_f:].copy()
    f = min(len(last) // 4, int(0.005 * SR))
    ramp = np.linspace(0, 1, f, dtype=np.float32)[:, None]
    last[:f] *= ramp
    last[-f:] *= ramp[::-1]
    tail = np.zeros((ECHOES * beat_f, 2), dtype=np.float32)
    for k in range(ECHOES):
        tail[k * beat_f : k * beat_f + len(last)] += last * 0.55 ** (k + 1)
    _, hi = split(tail)
    return hi


def build_mix(order: list[str], files: dict[str, Path], analysis: dict[str, Any], out_dir: Path,
              bars: int = 16, log: Any = print) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks: list[Track] = []
    for i, n in enumerate(order, 1):
        log(f"[load {i}/{len(order)}] {files[n].name}")
        tracks.append(load_track(n, files[n], analysis[n]))

    # blend lengths and cue points
    blend_beats: list[int] = []  # 0 = cut (echo-out, no beat-matching)
    for a, b in itertools.pairwise(tracks):
        jump = abs(b.meta["bpm"] - a.meta["bpm"]) / a.meta["bpm"]
        short = key_clash(a.meta["camelot"], b.meta["camelot"]) or jump > 0.06
        blend_beats.append(0 if jump > CUT_JUMP else 4 * (bars // 2 if short else bars))
    # incoming: start so its kick arrives by the bass swap; outgoing: keep its kick until the swap
    cue_in = [tracks[0].start_b]
    for i, t in enumerate(tracks[1:], 1):
        ci = t.grid_at_or_after(t.start_b)
        ks = t.kick_start()
        if ks is not None:
            ci = min(max(ci, t.grid_at_or_before(ks - blend_beats[i - 1] // 2)), ci + 2 * PHRASE)  # skip <= 16 bars
        cue_in.append(ci)
    cue_out: list[int] = []
    for i, t in enumerate(tracks[:-1]):
        L = blend_beats[i]
        prev_L = blend_beats[i - 1] if i else 0
        ke = t.kick_end()
        last = t.end_b - L if ke is None else min(t.end_b - L, ke - L // 2)
        co = t.grid_at_or_before(last)
        if co < cue_in[i] + prev_L + PHRASE:  # very short track: give up phrase alignment
            co = max(cue_in[i] + prev_L + 4, t.end_b - L)
        cue_out.append(co)

    pieces: list[tuple[int, np.ndarray]] = []  # (mix position, rendered audio)
    marks: list[tuple[int, Track]] = []
    pos = 0
    for i, t in enumerate(tracks):
        s0 = t.frame(cue_in[i])
        if i == len(tracks) - 1:  # last track plays out to its natural end
            seg = t.audio[s0:].copy()
            marks.append((pos, t))
            pieces.append((pos, seg))
            break

        nxt, L = tracks[i + 1], blend_beats[i]
        co, ci_n = cue_out[i], cue_in[i + 1]
        if L == 0:  # cut: native tempo up to the phrase, then echo out under the incoming drop
            seg = t.audio[s0 : t.frame(co)]
            beat_f = t.frame(co) - t.frame(co - 1)
            log(f"[cut {i + 1}/{len(tracks)}] {t.meta['bpm']:.1f} -> {nxt.meta['bpm']:.1f} BPM, echo-out")
            marks.append((pos, t))
            pieces.append((pos, np.concatenate([seg, echo_tail(seg, beat_f)])))
            pos += len(seg)
            continue
        n_beats = nxt.beats[ci_n : ci_n + L + 1] - nxt.beats[ci_n]
        prev_L = blend_beats[i - 1] if i else 0
        r0 = max(cue_in[i] + prev_L, co - RAMP_BEATS)

        # anchors: identity -> tempo ramp -> locked to the incoming beats
        anchors = [(0, 0), (t.frame(r0) - s0, t.frame(r0) - s0)]
        p_out = np.median(np.diff(t.beats[co : co + L + 1]))
        p_in = np.median(np.diff(n_beats))
        tgt = float(t.frame(r0) - s0)
        span = max(1, co - r0)
        for k in range(r0 + 1, co + 1):
            ratio = 1 + (p_in / p_out - 1) * (k - r0) / span
            tgt += (t.beats[k] - t.beats[k - 1]) * SR * ratio
            anchors.append((t.frame(k) - s0, round(tgt)))
        T = anchors[-1][1]
        for k in range(1, L + 1):
            anchors.append((t.frame(co + k) - s0, T + round(n_beats[k] * SR)))
        seg = t.audio[s0 : t.frame(co + L)]
        log(f"[stretch {i + 1}/{len(tracks)}] {t.meta['bpm']:.1f} -> {nxt.meta['bpm']:.1f} BPM, blend {L // 4} bars")
        rendered = stretch(seg, anchors)
        marks.append((pos, t))
        pieces.append((pos, rendered))
        pos += T

    total = max(p + len(a) for p, a in pieces)
    mix = np.zeros((total, 2), dtype=np.float32)
    for i, (p, a) in enumerate(pieces):
        a = a.copy()
        beat_f = int(60 / tracks[i].meta["bpm"] * SR)
        if i > 0 and blend_beats[i - 1] == 0:  # after a cut: drop in, 20 ms fade against clicks
            f = int(0.02 * SR)
            a[:f] *= np.linspace(0, 1, f, dtype=np.float32)[:, None]
        elif i > 0:  # incoming side of the previous blend
            n = round((tracks[i].beats[cue_in[i] + blend_beats[i - 1]] - tracks[i].beats[cue_in[i]]) * SR)
            _, _, in_hi, in_lo = blend_env(n, beat_f)
            lo, hi = split(a[:n])
            a[:n] = lo * in_lo[:, None] + hi * in_hi[:, None]
        if i < len(pieces) - 1 and blend_beats[i] == 0:
            pass  # echo tail already shaped in echo_tail()
        elif i < len(pieces) - 1:  # outgoing side of the next blend
            n = pieces[i + 1][0] - p
            n_bl = len(a) - n
            out_hi, out_lo, _, _ = blend_env(n_bl, beat_f)
            lo, hi = split(a[n:])
            a[n:] = lo * out_lo[:, None] + hi * out_hi[:, None]
        else:  # fade the very end over 4 bars
            f = min(len(a), 16 * beat_f)
            a[-f:] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
        mix[p : p + len(a)] += a

    wav = out_dir / "mix.wav"
    sf.write(wav, mix, SR, subtype="FLOAT")
    mp3 = out_dir / "mix.mp3"
    title = f"djmat mix ({len(tracks)} tracks)"
    subprocess.run(
        [ffmpeg(), "-v", "error", "-y", "-i", str(wav), "-af", "volume=2dB,alimiter=limit=0.84:attack=5:release=80:level=false",
         "-c:a", "libmp3lame", "-b:a", "320k", "-metadata", f"title={title}", str(mp3)],
        check=True,
    )
    wav.unlink()

    # tracklist + cue sheet
    lines, cue = [], [f'TITLE "{title}"', f'FILE "{mp3.name}" MP3']
    for n, (p, t) in enumerate(marks, 1):
        s = p / SR
        lines.append(f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}  {t.meta['bpm']:.0f} BPM  "
                     f"{t.meta['camelot']:>3}  {t.name.removesuffix('.mp3')}")
        cue += [f"  TRACK {n:02d} AUDIO", f'    TITLE "{t.name.removesuffix(".mp3")}"',
                f"    INDEX 01 {int(s // 60):02d}:{int(s % 60):02d}:{int(s % 1 * 75):02d}"]
    (out_dir / "tracklist.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "mix.cue").write_text("\n".join(cue) + "\n")
    (out_dir / "mix.json").write_text(json.dumps(
        [{"pos": n, "start": round(p / SR, 2), "file": t.name} for n, (p, t) in enumerate(marks, 1)], indent=1,
        ensure_ascii=False))
    log(f"Mix: {total / SR / 60:.1f} min -> {mp3.relative_to(ROOT)}")
    return mp3
