# SignLang

Real-time ASL fingerspelling and word-sign recognition from your webcam, running
locally on CPU. Nothing leaves your machine and no model is shipped pre-trained:
the classifier learns *your* hand, your angle, and your lighting.

MediaPipe supplies 21 hand landmarks per frame, a small MLP classifies them into
a sign, and a dwell timer turns a held sign into a character.

## Install

<details open>
<summary><b>Linux / macOS</b></summary>

```bash
git clone https://github.com/vaibhavsingh-shekhawat/signlang.git
cd signlang
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
bash scripts/fetch_models.sh
```

</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
git clone https://github.com/vaibhavsingh-shekhawat/signlang.git
cd signlang
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python scripts\fetch_models.py
```

</details>

Then record, train, and recognise:

```bash
signlang collect
signlang train
signlang live
```

## Requirements

- Python 3.11 or newer
- A webcam
- Linux, macOS, or Windows

## More install detail

The commands at the top of this page are all you need. Notes on the pieces that
catch people out:

`scripts\fetch_models.py` is the Windows equivalent of `fetch_models.sh` and
uses only the standard library, so it runs before `pip install` if you prefer
that order. No Git Bash or WSL needed.

If PowerShell blocks the activate script, either run the activation with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation
and call `.\.venv\Scripts\python.exe -m signlang.cli live` instead.

Setting the environment variables on Windows is the same as on Linux, but for
one session:

```powershell
$env:SIGNLANG_CAMERA = "1"
.\.venv\Scripts\python.exe -m signlang.cli live
```

The download step fetches the MediaPipe hand landmarker (~7.8 MB) into
`models/`. It is the only prerequisite that pip does not install.

## Moving between machines

Useful when you want to train on a faster laptop than the one you demo on. A
bundle carries your recordings and, unless you say otherwise, the trained
model.

```bash
signlang export                  # writes signlang-transfer.tar.gz
```

Copy that file over (USB stick, network share, email — it is under 1 MB) and on
the other machine:

```bash
signlang import                  # merges it into this project
```

The import **appends** recordings rather than replacing them, so you can pool
data from several machines and train once on the lot. Importing the same
bundle twice will double your counts, so keep one copy per machine.

Useful flags:

```bash
signlang export --no-model            # recordings only, leave weights behind
signlang export --out=backup.tar.gz   # write somewhere specific
signlang import --dry-run             # list what is in it, write nothing
signlang import --from=backup.tar.gz  # import a bundle not in the project root
```

What travels: `data/samples/*.npy` (including the `.sources.npy` sidecars) and
`models/signs_mlp.pt` + `models/labels.json`.

What does **not**: the hand landmarker. It is 7.8 MB of a fixed third-party
file, so both machines fetch it with `scripts/fetch_models.sh` (or
`fetch_models.py` on Windows). If a bundle somehow contains one, import skips
it and says so.

Samples are normalised landmark coordinates, not images, so a bundle is small
and nothing photographic is shared. They are also already wrist-centred and
palm-scaled, which is what lets them transfer between different cameras.

## Quick start

The order matters: record samples, train on them, then recognise.

```bash
signlang collect    # 1. record training samples
signlang train      # 2. train the classifier
signlang live       # 3. recognise
```

### 1. Record samples

```bash
signlang collect
```

Hold each sign clearly in view and press **SPACE** to capture a burst. Aim for
**4 bursts per sign**, varying position and angle slightly between them. Move
through the alphabet with **N** / **B**, press **R** to clear a sign, **Q** to
quit and save.

Capture only the letters you want; the trainer uses whatever is present. To
record a subset:

```bash
signlang collect --only=A,B,C,D,E
```

### 2. Train

```bash
signlang train
```

Writes `models/signs_mlp.pt` and `models/labels.json`. Useful flags:

```bash
signlang train --epochs=120 --lr=1e-3
signlang train --no-augment     # disable rotation/scale jitter
```

Roughly **40+ samples per sign** is a practical floor. Fewer and the model will
guess between similar letters (A and E especially).

### 3. Recognise

```bash
signlang live
```

| Key | Action |
| --- | --- |
| hold a sign | appends the letter after the dwell time |
| swipe **right** | space |
| swipe **left** | delete last letter |
| `C` | clear the transcript |
| `Q` or `Esc` | quit |

Tuning for your hand:

```bash
signlang live --dwell=450          # faster confirm, 200-2500 ms
signlang live --sensitivity=0.4    # lower = more willing to accept
```

Lower `--dwell` if letters are being missed, raise it if they fire while you are
still adjusting your hand.

## Other commands

```bash
signlang cameras   # list detected video devices
signlang check     # verify a source works: fps and hand detection rate
```

`signlang check` is the first thing to run when a sign will not detect. It
reports frames per second and how often a hand was found.

## Configuration

All optional, via environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SIGNLANG_CAMERA` | `0` | device index |
| `SIGNLANG_SOURCE` | — | any OpenCV-readable source, overrides the index |
| `SIGNLANG_WIDTH` / `SIGNLANG_HEIGHT` | `640` / `480` | capture size |
| `SIGNLANG_DETECT_HEIGHT` | same as capture | landmark resolution |
| `SIGNLANG_MIRROR` | `1` | set `0` to disable selfie mirroring |
| `SIGNLANG_MAX_HANDS` | `1` | hands to track |

An HTTP MJPEG source works too, which is useful for a phone pointed at yourself:

```bash
SIGNLANG_SOURCE=http://192.168.1.5:8080/video signlang live
```

## How it works

```
webcam -> HandPipeline (mirror, MediaPipe landmarks) -> 21x3 points
       -> feature extraction + augmentation -> MLP -> sign + confidence
       -> dwell timer -> transcript
```

Capture happens on a background thread with a small queue, so the newest frame
always wins and a slow consumer cannot stall the camera.

Recognition is deliberately **hold-to-confirm** rather than instant. A letter is
only committed once the same sign has been the top prediction for the dwell
window, which is what stops the output flickering between similar letters while
your hand is still moving.

The dwell timer is wall-clock based and runs on every captured frame, so frame
rate affects responsiveness but not what gets recognised.

**Recognition is worse on the exhibition laptop than on my own.** The model
learns your hand shape to some degree. Collect a few extra bursts on that
machine's camera, import them on the training machine, and retrain.

**Model recognises only some letters.** It can only predict signs present at
training time. `cat models/labels.json` shows what it knows. Collect the
missing ones and retrain.

## Troubleshooting

**Nothing is detected.** Run `signlang check`. If it reports 0% hands while your
hand is in frame, the problem is lighting or framing, not the model. Your hand
should fill roughly a third of the view.

**Letters come out wrong or flip between two options.** Not enough data. Collect
more bursts for those signs and retrain. Confusable pairs: A/E, M/N, U/V/R.

**Words render without spaces.** Each word sign (`HELLO`, `THANKYOU`, …) expands
to its text and a trailing space; letter signs do not. Use a right swipe
between letter groups.

**Low frame rate.** Landmark detection costs roughly 90 ms per frame on a small
CPU, capping the loop near 10 fps. Lowering `SIGNLANG_DETECT_HEIGHT` will not
help much; it is already the dominant cost. Closing other CPU-heavy work helps.

**Wrong camera.** `signlang cameras` lists indices, then set `SIGNLANG_CAMERA=1`.

**Windows: keyboard shortcuts do nothing in `collect`.** You need to click the
video window once so it has focus, then SPACE and the other keys work there.
Terminal key reading uses the Windows console API and works as well, but only
when the console window itself has focus.

**Windows: camera opens but the image is black.** Another app is holding it
(Teams, Zoom, Camera). Close those first.

## Data layout

```
data/samples/A.npy          recorded landmarks for sign A
data/samples/A.sources.npy  which camera each sample came from
models/signs_mlp.pt         trained weights (not committed)
models/hand_landmarker.task MediaPipe landmarker (fetched by script)
scripts/fetch_models.py     the download step for Windows
```

Recordings are plain NumPy arrays, so they are easy to back up, copy between
machines, or inspect:

```python
import numpy as np
a = np.load("data/samples/A.npy")     # shape: (n_samples, 63)
```

`.npy` arrays hold normalised landmark vectors, not images, so recordings are
small and no photographs of your hand are stored.

## License

MIT
