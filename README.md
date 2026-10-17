# dj

Analyzes a folder of mp3s (BPM, key, genre), orders them into a DJ set with a rising tempo arc and
harmonic (Camelot) transitions, and publishes the tracklist with play/download links at
https://dj.walczaki.com.

## Analyze & sort (local)

```bash
python3 -m venv .venv && .venv/bin/pip install essentia-tensorflow mutagen numpy
mkdir -p models && cd models
curl -LO https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json
cd ..

.venv/bin/python djmat.py analyze           # mp3/*.mp3 -> analysis.json / analysis.csv
.venv/bin/python djmat.py sort              # print set order -> order.csv
.venv/bin/python djmat.py rename --apply    # "01 - 120 BPM - 5A - Deep House - <original>.mp3"
.venv/bin/python djmat.py undo              # restore original names
.venv/bin/python djmat.py web               # -> public/index.html (commit it)
.venv/bin/python djmat.py zip               # -> mp3/dj-set.zip ("Download all" button)
.venv/bin/python djmat.py mix [--tracks 1-4] # -> mix/mix.mp3 + mix.cue + tracklist.txt ("Download mix" button)
```

`mix` needs the Rubber Band CLI. Without root it can be unpacked locally into `tools/`
(`apt-get download rubberband-cli librubberband2 ...` + `dpkg -x`, wrapper `tools/rubberband`);
ffmpeg comes from the `imageio-ffmpeg` pip package. Extra pip deps: `scipy soundfile imageio-ffmpeg`.

## Deploy

Every push to `main` builds an nginx image with `public/`, pushes it to GHCR and applies `k8s.yaml`
on the k3s server (same workflow as `ip`). Secrets: `VPS_HOST`, `VPS_SSH_KEY`.

MP3s are not in the image or the repo. They are served from `/srv/dj/mp3` on the server (hostPath):

```bash
ssh debian@$VPS_HOST 'sudo mkdir -p /srv/dj/mp3 && sudo chown debian /srv/dj/mp3'
.venv/bin/python djmat.py zip
scp mp3/*.mp3 mp3/dj-set.zip debian@$VPS_HOST:/srv/dj/mp3/
scp mix/mix.mp3 debian@$VPS_HOST:/srv/dj/mp3/dj-mix.mp3
```

After the first run make the GHCR package public (Packages → dj → Package settings → Change visibility),
then re-run the workflow.
