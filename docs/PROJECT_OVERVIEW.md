# SignLang — Project Description

A complete orientation document for a model picking up this codebase cold.

> **Note:** the models in this project have no image-input capability. Every visual
> claim in this document was verified **geometrically** (via intercepted `cv2` draw
> calls and pixel assertions), never visually. Screenshots exist but cannot be
> inspected by the model.

---

## 1. What this is

**SignLang** is a local, CPU-only American Sign Language (ASL) fingerspelling and
word-sign translator. It runs entirely on the user's machine — no network calls, no
cloud inference.

The user shows a letter to their webcam. MediaPipe detects 21 hand landmarks per
frame. A small PyTorch MLP classifies the handshape into one of 25 signs. A dwell
timer converts a *held* sign into a character. Output is a growing text transcript.

There are two front ends over the same model:

- `signlang live` — an OpenCV terminal/window overlay
- `signlang game` — a browser-based "shadow-play" Chrome-dino-style game where you
  spell letters to jump over cacti

**Location:** `/home/vaibhav/Projects/signlang`

---

## 2. Critical constraints — read before editing

Two files are **protected**. Do not modify them without explicit permission from
the user:

- `src/signlang/hands.py` — camera capture + MediaPipe landmark detection
- `src/signlang/engine.py` — dwell timer, smoothing, swipe gesture recognition

Everything else (`live.py`, `game/*`, `config.py`, `train.py`, etc.) is fair game.

**Never run the camera-backed app yourself.** The user runs it. Provide commands as
text. Synthesizing frames and testing pure functions is fine.

**Nothing is committed.** Six commits exist on the branch; all recent work is
uncommitted in the working tree.

---

## 3. Environment

This is a **low-resource machine** — relevant to any optimization work:

| | |
|---|---|
| CPU | 2 cores |
| RAM | 3.7 GB total, ~1.1 GB available |
| OS | Arch Linux, Omarchy, **Hyprland on Wayland** (XWayland fallback) |
| Python | 3.11.16 in `.venv/` |
| torch | 2.14.0+**cpu** |
| opencv | 4.11.0 |
| numpy | 1.26.4 |
| mediapipe | 0.10.21 |

Critical: **CPU-only torch**, 2 cores, little free RAM. The GPU is an AMD Radeon
iGPU that torch is *not* using.

### Measured performance (real, with camera on)

```
pipeline.read()   median = 133 ms   ->  7.6 fps ceiling
live overlay      ~7.7 ms per frame
```

**MediaPipe detection is 94% of the frame budget.** Lowering
`SIGNLANG_DETECT_HEIGHT` has *no* measurable effect (tested 480/320/240 → all
133 ms). The log confirms `Created TensorFlow Lite XNNPACK delegate for CPU`.

Do not attempt overlay micro-optimization expecting a responsiveness win — it is
capped at ~6% of frame time. The only real levers are:

1. GPU delegate for MediaPipe (in `hands.py` — protected)
2. A lighter hand model (in `hands.py` — protected)
3. Running detection every other frame (possible in `live.py`; the dwell timer is
   wall-clock based so accuracy is unaffected)

---

## 4. Architecture

```
webcam
  └─ HandPipeline (hands.py, background thread, queue maxsize=2)
       └─ MediaPipe HandLandmarker, RunningMode.VIDEO, max 1 hand
  └─ normalize_hand()  — mirror-fix, wrist-centre, palm-scale
  └─ landmark_features() — 63 floats
  └─ Predictor (MLP) — 25-way softmax
  └─ DwellRecognizer (engine.py) — smoothing, stability, dwell, swipe
  └─ text buffer
       ├─ live.py    — OpenCV overlay
       └─ game/*     — HTTP server + browser game
```

### Module map

| File | Lines | Role |
|---|---|---|
| `config.py` | 91 | All tunables, env vars, label definitions |
| `cli.py` | 98 | Command dispatch, help text |
| `hands.py` | 152 | **PROTECTED** Capture + MediaPipe + landmark drawing |
| `engine.py` | 201 | **PROTECTED** Dwell/smoothing/gesture state machine |
| `features.py` | 130 | Normalization, 63-dim features, augmentation |
| `model.py` | 87 | `SignMLP`, checkpoint save/load, `Predictor` |
| `dataset.py` | 122 | `.npy` sample storage, append, load |
| `collect.py` | 326 | Interactive sample recording |
| `train.py` | 221 | Training loop, class weighting, confusion matrix |
| `live.py` | 690 | OpenCV overlay UI |
| `test_live_layout.py` | 467 | Layout/geometry regression tests |
| `selftest.py` | 105 | Rotation-invariance robustness checks |
| `check.py` | 58 | Camera diagnostic |
| `cameras.py` | 67 | Enumerate video devices |
| `transfer.py` | 293 | export/import tarballs for moving between machines |
| `game/pool.py` | 102 | Which letters are safe to grade |
| `game/scoring.py` | 248 | Pure game state machine, no camera |
| `game/detector.py` | 210 | Background classifier thread |
| `game/server.py` | 264 | stdlib HTTP server, JSON state API |
| `game/static/game.js` | 763 | Canvas dino game |
| `eval_burst.py` | 148 | **BROKEN** — see §9 |

---

## 5. Feature pipeline

`features.py`:

**`normalize_hand(points, handedness)`** — flips x for "Left" handedness, subtracts
the wrist, divides by wrist→middle-MCP distance. This is why recordings transfer
between cameras and people.

**`landmark_features(p)`** → 63 floats:

- 63 raw coords (21 × 3, x/y/z)
- 5 finger extensions (tip→MCP distance / palm)
- pinch (thumb-tip → index-tip / palm)
- spread (adjacent fingertip gaps)
- mean fingertip depth
- palm normal (cross product)

**`augment()`** — roll ±11°, pitch ±26°, yaw ±20°, finger jitter 0.014, noise 0.006,
re-normalized after every step.

---

## 6. Dwell recognizer (`engine.py` — PROTECTED)

State machine. Per frame:

1. Probabilities from the model
2. Exponential moving average over a 420 ms window
3. `agreement` = fraction of window frames whose argmax equals the current top label
4. Candidate changes → reset dwell clock
5. Dwell starts only if: `agreement >= 0.7` **and** `movement <= 26px` **and**
   `conf >= threshold (0.72)` **and** `conf - runner >= margin*0.5`
6. Emit when `t - dwell_start >= dwell_ms`

Key constants (`config.py`):

```python
DWELL_MS = 1000          # was 620, changed to 1000 by user request
SMOOTH_MS = 420
CONFIRM_THRESHOLD = 0.72
STABLE_MARGIN = 0.18
REPEAT_COOLDOWN_MS = 900
SWIPE_MIN_PX = 110
SWIPE_MAX_MS = 420
```

`DWELL_MS = 1000` is the current value — the user complained letters fired "too
fast" and asked for a 1 second timer. `--dwell=` overrides it; `set_dwell` clamps
to 200–2500.

**Swipe gestures:** right swipe = space, left swipe = backspace. Requires ≥110px
horizontal, ≤420ms, ≥1.8× more horizontal than vertical, ≥4 trail points, 1100ms
cooldown.

---

## 7. Model

`SignMLP`: `in_dim → 192 → 112 → 25`, each hidden layer
`Linear + LayerNorm + GELU + Dropout(0.28)`.

Checkpoint (`models/signs_mlp.pt`) stores state_dict, labels, mean/std,
temperature, meta. Normalization is applied at inference: `(x - mean) / std`.

### Current training data — 10,234 samples, 25 labels (A–Y, no Z)

```
A 210   B 672   C 616   D 616   E 616   F 392   G 168   H 168
I 336   J 336   K 560   L 560   M 504   N 504   O 336   P 336
Q 224   R 840   S 280   T  56   U 784   V 280   W 280   X 280
Y 280
```

**Counts are wildly uneven** — 840 for R vs 56 for T. `train.py` compensates with
class weighting. T at 56 samples is roughly 4 recording bursts, far too few to
generalise.

The model was trained on **one person's hands**. It is known to be worse on other
people.

### Train

```bash
.venv/bin/signlang train --epochs=70 --batch=64 --lr=2e-3 --val-frac=0.2
```

`train.py` splits rows **at random**, which is a known weakness — see §9.

### Collect

```bash
.venv/bin/signlang collect --only=R,U
```

Keys: SPACE capture burst, N/B next/prev, R clear sign, Q quit+save. Each burst is
14 frames (`BURST_FRAMES`, defined in `collect.py` — **not** in `config.py`).
Recording **appends** to existing `.npy` files, so `--only=R,U` adds to R and U
rather than replacing them.

---

## 8. Front end A — `signlang live`

```bash
.venv/bin/signlang live                       # fullscreen, 1s dwell
.venv/bin/signlang live --windowed            # opt out of fullscreen
.venv/bin/signlang live --dwell=1600 --sensitivity=0.4
```

Keys: `F` toggle fullscreen, `Q`/`Esc` quit, `C` clear, Backspace delete.

### Overlay layout

All geometry derives from `u = max(2.0, min(w,h) / 40.0)` so it scales identically
from a 240×320 thumbnail to a 1366×768 screen.

- **Top bar** — wordmark, "live" tag, A–Y alphabet strip (highlighted cell = current
  candidate). Only drawn when cells are wide enough.
- **Reading card** (top-left) — big letter in a dwell ring, countdown (`0.7s`)
  counting *down*, confidence %, progress bar across the foot, status note.
- **Runners-up panel** — top 4 alternatives with probability bars. Turned on/off by
  available width.
- **Bottom panel** — recent-letters strip (newest first, green + underlined),
  transcript (last 2 wrapped lines), gesture chip, fps, help line.

### Critical bug already fixed — understand this

`cv2.getWindowImageRect()` **does not return the window size.** It returns the size
of the *image last drawn into it*. A fullscreen window created empty measures
98×28, so composing to that size crammed the entire overlay into the top-left
corner.

Fix: `_screen_size()` in `live.py` asks Hyprland for the focused monitor via
`hyprctl monitors all`, dividing by `scale:` (a scaled monitor reports native
resolution, then the compositor scales down). Fallbacks → capture size. On this
machine it returns `(1366, 768)`.

Do not "simplify" this back to `getWindowImageRect`.

### Performance work already done

- `_blend()` uses a bounded cache of solid images + `cv2.addWeighted`. The strided
  copy-in/copy-back was **removed** after verifying `addWeighted` writes through a
  strided dst correctly.
- LUT-based blending was tested and is **10× slower** than `addWeighted`. `cv2.LUT`
  also slower. Don't retry these.
- Cached text sprites: not a win, alpha-compositing costs as much as rasterizing.

---

## 9. Known issues

**`eval_burst.py` is broken.** It does `from signlang.config import BURST_FRAMES`
but `BURST_FRAMES` lives in `collect.py`. Running it raises `ImportError`.

It exists because of a real methodological problem: **`train.py` splits validation
rows at random, but rows arrive in bursts of 14 near-identical frames.** A random
split puts near-duplicate neighbours on both sides, so validation scores ~1.00 and
is meaningless. `eval_burst.py` holds out whole bursts. Fixing the import is a
one-line move of the constant.

**Game pool is not evidence-based.** `game/pool.py` excludes I, P, S, L, U, X as
"too confusable" and T, Z as "too thin" — but `eval_burst.py` was never run, so
these are guesses. U is excluded, yet the user just collected 784 new U samples
specifically because U is confusing. These lists should be revisited after fixing
and running the burst evaluation.

**`game/scoring.py` comment is stale** — says "620ms dwell in engine.py"; it is now
1000 ms.

**`en_US-lessac-medium.onnx` (63 MB) sits unused in the project root.**
`config.py` defines `TTS_VOICE` but nothing references it. There is no TTS feature.

**No GPU acceleration.** Torch is CPU-only; MediaPipe uses XNNPACK CPU delegate.

---

## 10. Front end B — `signlang game`

```bash
.venv/bin/signlang game --port 8000 --host 0.0.0.0   # 0.0.0.0 to reach from a phone
```

Deliberately **stdlib-only** — `http.server` + threads, no FastAPI/uvicorn, so
installing on an exhibition machine cannot fail on a web framework.

### No video stream

The browser polls a small JSON blob. The camera belongs to the server process (only
one process can open a webcam), so the corner picture is a single JPEG fetched from
`/api/preview.jpg` a few times a second — not a long-lived stream that could wedge
the page.

Endpoints: `GET /`, `/static/*`, `/api/state`, `/api/preview.jpg`, `/api/pool`,
`/api/config`; POST for start/apply/shutdown.

### Threading contract

`tick_loop` runs at ~60 Hz for animation and phase timing. `LetterReader` runs at
~10 fps and publishes the latest letter into its own slot. **They never wait on each
other.** `detector.py` holds its lock for microseconds and publishes plain scalars.

A failed detection leaves the *previous* reading in place rather than reporting
blank — a dropped frame must not look like the player leaving the frame.

### Game phases

```
idle → ready (2200ms, NOT scored) → answer (2000ms, scored)
     → correct (900ms) → ready …
     → wrong (run over)
```

The `ready` phase exists because a scored window starting the instant the letter
appears punishes reaction time rather than a wrong letter.

A wrong shape held long enough costs time but **does not end the run** — only the
answer window closing does. A low-confidence detection resets the hold but is not a
strike: ending a run on the model's own uncertainty would be punishing the player
for it.

Hold 420 ms, minimum confidence 0.62.

### Game canvas

Chrome-dino homage drawn on `<canvas>`. Constants: `GROUND=0.80`, `DINO_X=0.12`,
`CACTUS_START=1.15`, `CACTUS_CLOSE=0.34`, `JUMP_MS=620`. Sprites are ASCII bitmaps
compiled to horizontal runs at load, then `fillRect`-ed. Optimized to be
**pixel-identical** across 300 frames at DPR 1 and 2.

---

## 11. Tests

Four suites, all passing as of the last run:

```bash
.venv/bin/python -m signlang.selftest              # rotation invariance, renormalisation
.venv/bin/python -m signlang.test_live_layout      # 84 overlay render cases
PYTHONPATH=src .venv/bin/python -m signlang.game.selftest
PYTHONPATH=src .venv/bin/python -m signlang.game.selftest_server
```

### `test_live_layout.py` — the important one

Intercepts every `cv2` draw call and asserts geometry: nothing outside the frame,
no transcript/status collision, alphabet never overlaps the wordmark, card never
touches the bottom panel, dwell ring stays inside the card, every glyph is backed by
a panel, no more alternative rows than fit.

Current coverage: **6 sizes × 14 states = 84 cases** (1280×720, 1024×576, 640×480,
480×640, 320×240, 240×320), plus blend parity, letterbox aspect, canvas-fill, and
screen-size resolution.

States: nohand, unclear, holding, gesture, long, nospaces, verylong, plus
recent-full, recent-one, flash-start/mid/end/faint, flash-norecent.

**Established practice here:** every new check must be verified by *injecting the
regression it is supposed to catch*. Several checks in this file were wrong on
first attempt and passed while broken:

- the alphabet check filtered on the wrong tuple index, so it only ever matched one
  letter
- the letterbox check measured canvas aspect instead of content aspect
- the progress-bar check was so loose it matched the runners-up bars, so deleting
  the dwell bar still passed
- the blend check was too permissive about strided write-back

Do the same. A green suite that never went red proves nothing.

### Bug history worth remembering

- **Strided ROI copy-back**: an optimization pass introduced a bug where the blend
  result stayed in a temporary and was never written to the frame. Caught by test,
  not by eye.
- **Font scale**: OpenCV `fontScale` is not a pixel size — a capital letter is
  ~22×scale px tall. All sizes go through `_scale(px_height)` with
  `_PX_PER_SCALE = 22.0`.
- **Test blindness**: filtering boxes on `b[1]` (x0) instead of `b[2]` (y0) left the
  entire alphabet strip untested.

---

## 12. Working tree state

```
 M README.md                        dwell docs updated
 M data/samples/R.npy               840 samples (user collected more)
 M data/samples/R.sources.npy
 M data/samples/U.npy               784 samples
 M data/samples/U.sources.npy
 M models/signs_mlp.pt              retrained
 M src/signlang/cli.py              game command dispatch
 M src/signlang/config.py           DWELL_MS 620 -> 1000
 M src/signlang/live.py             fullscreen, countdown, progress bar, UI rework
?? backups/                         user backup of signs_mlp.pt + labels.json
?? docs/                            this file
?? eval_burst.py                    BROKEN (ImportError)
?? src/signlang/game/               entire game subsystem, untracked
?? src/signlang/test_live_layout.py untracked
```

`git log`: 6 commits, `6d000b5` most recent.

---

## 13. Recent user requests and what was done

1. **"everything gets ushed into a small corner on the top left corner"** →
   root-caused to `getWindowImageRect` returning image size not window size; fixed
   with Hyprland query.
2. **"add a 1 second timer, and drastically improve the UI"** → `DWELL_MS = 1000`;
   added countdown (`0.7s`, counts down) and a progress bar; enlarged letter and
   card.
3. **"optimize it, don't change the visuals"** → removed strided copy in `_blend()`
   (9.09 → 7.70 ms). Rejected LUT (10× slower), sprite caching, and `LINE_8`
   (changes visuals). Reported honestly that detection is 94% of the frame.
4. **Training R and U** → the user collected 840 R and 784 U samples and retrained
   themselves. Model already updated in the tree.
5. **Open follow-up:** R/U confusion is a known trio with V (`game/pool.py`
   documents U as "V with fingers closed; only the gap differs"). If U/R still swap
   after the 1 s dwell, V samples are likely needed too.

---

## 14. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SIGNLANG_CAMERA` | `0` | Device index |
| `SIGNLANG_SOURCE` | — | Any OpenCV-readable source, overrides index |
| `SIGNLANG_WIDTH` / `SIGNLANG_HEIGHT` | `640` / `480` | Capture size |
| `SIGNLANG_DETECT_HEIGHT` | = capture | Landmark resolution — **measured to have no effect on speed** |
| `SIGNLANG_MIRROR` | `1` | Selfie mirroring |
| `SIGNLANG_MAX_HANDS` | `1` | Hands to track |
| `SIGNLANG_READ_TIMEOUT` | `3.0` | Seconds |

---

## 15. Things a previous model got wrong — do not repeat

- **Claimed a screenshot had been visually verified.** The model has no image input.
  Every visual claim was geometric, never visual. If asked about appearance, say you
  cannot see images.
- **Ran `hyprctl monitors` without confirming it exists.** It does on Hyprland, but
  the code falls back cleanly when absent. Same for `tkinter` (available) vs
  PyQt5/PySide (not installed).
- **Trusted `getWindowImageRect`.** Verified the bug with a standalone probe before
  changing anything, after the user reported the symptom.
- **Wrote a test that passed while the code was broken.** Two separate instances.
  Always inject the regression.

---

## 16. Quick reference

```bash
# diagnosis first, always
.venv/bin/signlang check
.venv/bin/signlang cameras

# workflow
.venv/bin/signlang collect --only=R,U
.venv/bin/signlang train
.venv/bin/signlang live

# game
.venv/bin/signlang game --host 0.0.0.0 --port 8000

# move between machines
.venv/bin/signlang export
.venv/bin/signlang import

# tests
.venv/bin/python -m signlang.selftest
.venv/bin/python -m signlang.test_live_layout
PYTHONPATH=src .venv/bin/python -m signlang.game.selftest
PYTHONPATH=src .venv/bin/python -m signlang.game.selftest_server
```

Undo for a retrain: `cp backups/signs_mlp.pt backups/labels.json models/` or
`git checkout data/samples/R.npy data/samples/U.npy`.

External `curl` to localhost is sandbox-blocked; use the in-process HTTP test
instead.

---

## 17. Recommended next steps

1. **Fix `eval_burst.py`** — move `BURST_FRAMES` to `config.py` (or import it from
   `collect`), then run it. Its result should drive the game pool exclusions and
   tell you whether the R/U retraining actually helped.
2. **Decide on `hands.py`** — a GPU delegate is the only real path to better than
   7.6 fps. Requires explicit user permission.
3. **Reconsider `game/pool.py` exclusions** once real burst-held-out accuracy
   numbers exist.
