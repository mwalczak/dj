"""djmat: analyze BPM/key/genre of mp3s, sort them into a harmonic set, rename files.

Usage:
    python djmat.py analyze            # writes analysis.json + analysis.csv
    python djmat.py sort               # prints proposed order, writes order.csv
    python djmat.py rename [--apply]   # dry-run by default; renames in set order
    python djmat.py undo               # restores original filenames
    python djmat.py web                # generates public/index.html
    python djmat.py zip                # bundles the set into mp3/dj-set.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent
MP3_DIR = ROOT / "mp3"
MODELS = ROOT / "models"
ANALYSIS = ROOT / "analysis.json"
ORDER_CSV = ROOT / "order.csv"
RENAMES = ROOT / "renames.json"  # {original_name: current_name}

# Camelot wheel: (key, scale) -> code
_MAJOR = ["B", "F#", "C#", "G#", "D#", "A#", "F", "C", "G", "D", "A", "E"]
_MINOR = ["G#", "D#", "A#", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
_ENHARMONIC = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def camelot(key: str, scale: str) -> str:
    key = _ENHARMONIC.get(key, key)
    table = _MAJOR if scale == "major" else _MINOR
    return f"{table.index(key) + 1}{'B' if scale == 'major' else 'A'}"


def fold_bpm(bpm: float, lo: float = 100.0, hi: float = 160.0) -> float:
    """Fix half/double-tempo detections for dance music."""
    while bpm < lo:
        bpm *= 2
    while bpm >= hi:
        bpm /= 2
    return bpm


def load_renames() -> dict[str, str]:
    return json.loads(RENAMES.read_text()) if RENAMES.exists() else {}


def current_files() -> dict[str, Path]:
    """original_name -> current path for every mp3 in MP3_DIR."""
    by_current = {v: k for k, v in load_renames().items()}
    return {by_current.get(p.name, p.name): p for p in sorted(MP3_DIR.glob("*.mp3"))}


# ---------------------------------------------------------------- analyze


def analyze_track(path: Path, genre_models: tuple[Any, Any, list[str]]) -> dict[str, Any]:
    import essentia.standard as es
    import numpy as np

    audio = es.MonoLoader(filename=str(path), sampleRate=44100)()

    bpm, _beats, beats_conf, _, _ = es.RhythmExtractor2013(method="multifeature")(audio)
    key, scale, key_strength = es.KeyExtractor(profileType="edma")(audio)

    stereo, sr, *_ = es.AudioLoader(filename=str(path))()
    _, _, lufs, _ = es.LoudnessEBUR128(sampleRate=sr)(stereo)

    # Energy proxy: mean RMS of the loudest half of 1s frames (ignores quiet intros)
    frames = np.array([es.RMS()(f) for f in es.FrameGenerator(audio, frameSize=44100, hopSize=44100)])
    energy = float(np.mean(np.sort(frames)[len(frames) // 2 :]))

    embed_model, head_model, classes = genre_models
    audio16 = es.MonoLoader(filename=str(path), sampleRate=16000, resampleQuality=4)()
    preds = head_model(embed_model(audio16)).mean(axis=0)
    top = np.argsort(preds)[::-1][:3]
    styles = [(classes[i].split("---")[-1], round(float(preds[i]), 3)) for i in top]

    return {
        "bpm_raw": round(float(bpm), 2),
        "bpm": round(fold_bpm(float(bpm)), 1),
        "bpm_conf": round(float(beats_conf), 2),
        "key": f"{key} {scale}",
        "camelot": camelot(key, scale),
        "key_strength": round(float(key_strength), 2),
        "lufs": round(float(lufs), 1),
        "energy": round(energy, 4),
        "genre": styles[0][0],
        "genres_top3": styles,
        "duration": round(len(audio) / 44100, 1),
    }


def cmd_analyze(args: argparse.Namespace) -> None:
    import essentia.standard as es

    meta = json.loads((MODELS / "genre_discogs400-discogs-effnet-1.json").read_text())
    models = (
        es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODELS / "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1"
        ),
        es.TensorflowPredict2D(
            graphFilename=str(MODELS / "genre_discogs400-discogs-effnet-1.pb"),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        ),
        meta["classes"],
    )

    results: dict[str, Any] = json.loads(ANALYSIS.read_text()) if ANALYSIS.exists() else {}
    files = current_files()
    for i, (orig, path) in enumerate(files.items(), 1):
        if orig in results and not args.force:
            continue
        print(f"[{i}/{len(files)}] {orig}", file=sys.stderr)
        try:
            results[orig] = analyze_track(path, models)
        except Exception as e:  # noqa: BLE001 - keep going on a single bad file
            print(f"  ! failed: {e}", file=sys.stderr)
            continue
        ANALYSIS.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    with (ROOT / "analysis.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "bpm", "bpm_raw", "bpm_conf", "camelot", "key", "key_strength", "lufs", "energy", "genre", "top3"])
        for name, r in sorted(results.items(), key=lambda kv: kv[1]["bpm"]):
            top3 = "; ".join(f"{g} {p}" for g, p in r["genres_top3"])
            w.writerow([name, r["bpm"], r["bpm_raw"], r["bpm_conf"], r["camelot"], r["key"],
                        r["key_strength"], r["lufs"], r["energy"], r["genre"], top3])
    print(f"Analyzed {len(results)} tracks -> {ANALYSIS.name}, analysis.csv")


# ---------------------------------------------------------------- sort


def key_cost(a: str, b: str) -> float:
    """0 = same key, 1 = Camelot neighbour (±1 or A/B), 2 = energy boost (+2), else clash."""
    na, la, nb, lb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    d = (nb - na) % 12
    if la == lb:
        return {0: 0.0, 1: 1.0, 11: 1.0, 2: 2.0, 7: 2.5}.get(d, 6.0)
    return 1.0 if d == 0 else 6.0


def transition_cost(a: dict[str, Any], b: dict[str, Any]) -> float:
    pct = (b["bpm"] - a["bpm"]) / a["bpm"] * 100
    bpm_c = abs(pct) * (0.6 if pct >= 0 else 1.5)  # prefer rising tempo
    if pct < -2:
        bpm_c += 8  # set should build, not sag
    if abs(pct) > 6:
        bpm_c += 10  # beyond clean time-stretch range
    return key_cost(a["camelot"], b["camelot"]) * 2 + bpm_c


def best_order(tracks: dict[str, dict[str, Any]], window: int = 5, beam: int = 2000) -> list[str]:
    """Rising-tempo arc: slot k may take any track whose BPM rank is within `window` of k.

    Keeps the set building in tempo (no orphans dumped at the end) while
    reshuffling neighbours for the best harmonic flow. Beam search over total cost.
    """
    by_bpm = sorted(tracks, key=lambda n: tracks[n]["bpm"])
    rank = {n: i for i, n in enumerate(by_bpm)}
    paths: list[tuple[float, list[str]]] = [(0.0, [])]
    for k in range(len(by_bpm)):
        nxt: list[tuple[float, list[str]]] = []
        for cost, path in paths:
            used = set(path)
            overdue = [n for n in by_bpm[: max(0, k - window + 1)] if n not in used]
            cands = overdue or [n for n in by_bpm[: k + window] if n not in used]
            for n in cands:
                step = transition_cost(tracks[path[-1]], tracks[n]) if path else 0.0
                nxt.append((cost + step, path + [n]))
        nxt.sort(key=lambda x: x[0])
        # dedupe states (same set used + same last track) to keep the beam diverse
        seen: set[tuple[frozenset[str], str]] = set()
        paths = []
        for c, p in nxt:
            sig = (frozenset(p), p[-1])
            if sig not in seen:
                seen.add(sig)
                paths.append((c, p))
            if len(paths) >= beam:
                break
    assert all(abs(rank[n] - i) < 2 * window for i, n in enumerate(paths[0][1]))
    return paths[0][1]


def cmd_sort(args: argparse.Namespace) -> list[str]:
    tracks: dict[str, dict[str, Any]] = json.loads(ANALYSIS.read_text())
    present = current_files()
    tracks = {k: v for k, v in tracks.items() if k in present}
    order = best_order(tracks)

    with ORDER_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#", "bpm", "camelot", "genre", "file", "bpm_jump_%", "key_cost"])
        prev = None
        for i, n in enumerate(order, 1):
            t = tracks[n]
            jump = f"{(t['bpm'] - prev['bpm']) / prev['bpm'] * 100:+.1f}" if prev else ""
            kc = key_cost(prev["camelot"], t["camelot"]) if prev else ""
            w.writerow([i, t["bpm"], t["camelot"], t["genre"], n, jump, kc])
            if not args.quiet:
                print(f"{i:02d}  {t['bpm']:6.1f}  {t['camelot']:>3}  {jump:>6}  {kc!s:>4}  {t['genre'][:22]:22}  {n}")
            prev = t
    return order


# ---------------------------------------------------------------- rename

_BAD = '<>:"/\\|?*'


def new_name(idx: int, orig: str, t: dict[str, Any]) -> str:
    genre = "".join(c for c in t["genre"] if c not in _BAD)
    return f"{idx:02d} - {round(t['bpm'])} BPM - {t['camelot']} - {genre} - {orig}"


def cmd_rename(args: argparse.Namespace) -> None:
    tracks = json.loads(ANALYSIS.read_text())
    args.quiet = True
    order = cmd_sort(args)
    files = current_files()
    renames = load_renames()

    plan = [(files[o], new_name(i, o, tracks[o]), o) for i, o in enumerate(order, 1)]
    for src, dst, _ in plan:
        print(f"{src.name}\n   -> {dst}")
    if not args.apply:
        print("\nDry run. Re-run with --apply to rename.")
        return

    # two-phase to avoid collisions between old/new names
    tmp = [(src.rename(src.with_name(f".djmat_tmp_{i}.mp3")), dst, o) for i, (src, dst, o) in enumerate(plan)]
    for t, dst, o in tmp:
        t.rename(MP3_DIR / dst)
        renames[o] = dst
    RENAMES.write_text(json.dumps(renames, indent=2, ensure_ascii=False))
    print(f"Renamed {len(plan)} files. Undo with: python djmat.py undo")


def cmd_undo(_: argparse.Namespace) -> None:
    renames = load_renames()
    for orig, cur in renames.items():
        p = MP3_DIR / cur
        if p.exists():
            p.rename(MP3_DIR / orig)
    RENAMES.unlink(missing_ok=True)
    print(f"Restored {len(renames)} filenames.")


# ---------------------------------------------------------------- web

_NOISE = re.compile(
    r"\s*[\(\[]?\b(official\s+(music\s+)?(video|visuali[sz]er)|visuali[sz]er)\b[\)\]]?", re.IGNORECASE
)


def split_artist_title(orig: str) -> tuple[str, str]:
    name = _NOISE.sub("", Path(orig).stem.replace(" ●● ", " - ")).strip()
    parts = name.split(" - ")
    return (parts[-2], parts[-1]) if len(parts) > 1 else ("", name)


def cmd_web(args: argparse.Namespace) -> None:
    from datetime import date

    tracks = json.loads(ANALYSIS.read_text())
    args.quiet = True
    order = cmd_sort(args)
    files = current_files()
    energies = [tracks[o]["energy"] for o in order]
    lo, hi = min(energies), max(energies)

    rows = []
    for i, o in enumerate(order, 1):
        t = tracks[o]
        artist, title = split_artist_title(o)
        rows.append({
            "pos": i, "file": files[o].name, "artist": artist, "title": title,
            "bpm": t["bpm"], "camelot": t["camelot"], "key": t["key"],
            "genre": t["genre"], "genres": [g for g, _ in t["genres_top3"]],
            "energy": round((t["energy"] - lo) / (hi - lo), 3) if hi > lo else 0.5,
            "duration": t["duration"],
        })
    data = {
        "tracks": rows, "mp3Base": args.mp3_base, "generated": date.today().isoformat(),
        "zip": ZIP_NAME, "totalBytes": sum(files[o].stat().st_size for o in order),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (ROOT / "web" / "template.html").read_text().replace("/*__DATA__*/", payload)
    out = ROOT / "public" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} tracks, mp3 base '{args.mp3_base}')")


# ---------------------------------------------------------------- zip

ZIP_NAME = "dj-set.zip"


def cmd_zip(args: argparse.Namespace) -> None:
    """Bundle the set into mp3/dj-set.zip (stored, not deflated: mp3 doesn't compress)."""
    import zipfile

    args.quiet = True
    order = cmd_sort(args)
    files = current_files()
    out = MP3_DIR / ZIP_NAME
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for o in order:
            z.write(files[o], f"DJ Set/{files[o].name}")
    tmp.replace(out)
    print(f"Wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB, {len(order)} tracks)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--force", action="store_true", help="re-analyze cached tracks")
    s = sub.add_parser("sort")
    s.add_argument("--quiet", action="store_true")
    r = sub.add_parser("rename")
    r.add_argument("--apply", action="store_true")
    sub.add_parser("undo")
    sub.add_parser("zip")
    w = sub.add_parser("web")
    w.add_argument("--mp3-base", default="mp3/", help="URL prefix for mp3 links")
    args = ap.parse_args()
    {"analyze": cmd_analyze, "sort": cmd_sort, "rename": cmd_rename, "undo": cmd_undo, "web": cmd_web, "zip": cmd_zip}[args.cmd](args)


if __name__ == "__main__":
    main()
