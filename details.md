# SignLang — Complete Project Reference

> A self-contained technical description of the repository, written so any LLM (or
> engineer) can understand the codebase cold without reading the source first.
> Verified against the working tree on 2026-10-07 (20 commits, plus uncommitted
> work described in §14).

---

## 1. What the project is

**SignLang** is a local, CPU-only American Sign Language (ASL) recognizer that runs
entirely on the user's machine — no cloud, no network calls. It has **two
independent recognition subsystems** sharing one repo and one dispatcher:

| | **Letters mode** | **Words mode** |
|---|---|---|
| Recognizes | ASL fingerspelling (static handshapes) | Holistic word-signs (motion sequences) |
| Model | Small PyTorch **MLP** (supervised, trained) | **DTW + kNN** (instance-based, no training loop) |
| Vision | MediaPipe **HandLandmarker** (21 pts/hand) | MediaPipe **Holistic** (9 body + 42 hand pts) |
| Input | Single frame → 74-dim feature vector | 30-frame sequence → 51×3 landmarks |
| Package | `src/signlang/` (installed as `signlang`) | `words/` (standalone, never imported by letters) |
| Entry | `signlang live` / `python main.py` | `python main.py --mode words` |

Two front ends exist over the letters model: an OpenCV window overlay
(`signlang live`) and a browser game (`signlang game` — a Chrome-dino-style
"shadow-play" where you spell letters to jump cacti).

Privacy property: recordings are **normalized landmark coordinates, not images**,
so nothing photographic is ever stored or shared.

---

## 2. Repository layout

```
signlang/
├── main.py                    # Dispatcher: --mode letters|words (enforces isolation)
├── pyproject.toml             # Package metadata; entry point: signlang = signlang.cli:main
├── README.md                  # User-facing install/usage docs
├── details.md                 # This file
├── docs/PROJECT_OVERVIEW.md   # Earlier orientation doc (partly stale — see §14)
├── eval_burst.py              # Burst-held-out CV eval for the letters model
│
├── src/signlang/              # ===== LETTERS SUBSYSTEM (installed package) =====
│   ├── config.py              # All tunables + env vars (BURST_FRAMES=14 lives here)
│   ├── cli.py                 # Subcommand dispatch: collect/train/live/game/cameras/check/export/import
│   ├── hands.py               # [PROTECTED] Camera capture thread + MediaPipe HandLandmarker
│   ├── engine.py              # [PROTECTED] DwellRecognizer: smoothing/dwell/swipe state machine
│   ├── features.py            # normalize_hand, landmark_features (74-dim), augment
│   ├── model.py               # SignMLP, checkpoint IO, Predictor
│   ├── dataset.py             # .npy sample storage (one file per sign)
│   ├── collect.py             # Interactive burst recorder (SPACE/N/B/R/Q keys)
│   ├── train.py               # Training loop, class weighting, confusion matrix
│   ├── live.py                # OpenCV live overlay UI (~690 lines)
│   ├── check.py / cameras.py  # Diagnostics
│   ├── transfer.py            # export/import tarballs between machines
│   ├── game/                  # Browser game subsystem (stdlib HTTP server)
│   │   ├── server.py          # http.server + JSON state API + static files
│   │   ├── scoring.py         # Pure game state machine (idle/ready/answer/…)
│   │   ├── detector.py        # Background classification thread
│   │   ├── pool.py            # Which letters the game is allowed to quiz
│   │   ├── static/            # index.html, style.css, game.js (canvas dino)
│   │   ├── selftest.py        # 52 assertions on the state machine
│   │   └── selftest_server.py # In-process HTTP API tests
│   ├── selftest.py            # Rotation-invariance robustness checks
│   ├── test_live_layout.py    # 84 overlay-geometry regression cases
│   └── test_debounce.py       # SPACE hold-once debounce tests (pytest)
│
├── words/                     # ===== WORDS SUBSYSTEM (isolated package) =====
│   ├── __init__.py            # words.main(source, extra_args): routes record/eval/live
│   ├── config.py              # 25-word vocabulary + all thresholds
│   ├── normalize.py           # Masked extraction, imputation, shoulder-relative norm
│   ├── classifier.py          # Vectorized DTW + calibrated kNN (DTWKNNClassifier)
│   ├── dataset.py             # Per-word .npy storage, provenance sidecars, synthetic gen
│   ├── eval.py                # Provenance hold-out + stratified CV CLI
│   ├── record.py              # Webcam recorder / synthetic generator CLI
│   ├── live.py                # Real-time recognition with motion-energy segmentation
│   ├── data/samples/          # <WORD>.npy + <WORD>.sources.npy (87 samples, 16 words)
│   ├── models/words_classifier.npz  # Stored training set + calibration scales
│   └── tests/                 # 6 pytest files (~53 tests)
│
├── data/samples/              # Letters dataset: A.npy…Y.npy + SPACE.npy (+ .sources.npy)
├── models/                    # signs_mlp.pt, labels.json, hand_landmarker.task (7.8 MB, fetched)
├── scripts/
│   ├── fetch_models.sh/.py    # Downloads MediaPipe hand landmarker (only non-pip dep)
│   ├── record_space.py        # Record the custom SPACE gesture samples
│   ├── rebuild_words.py       # Rebuild words dataset+classifier (keeps real samples)
│   └── bench_landmarker.py    # Camera-free landmarker CPU/GPU cost benchmark
├── backups/                   # User backups of model/labels + timestamped archives
├── en_US-lessac-medium.onnx   # 63 MB Piper TTS voice (used by src/signlang/tts.py for live speech feedback)
└── signlang-transfer.tar.gz   # Example export bundle from `signlang export`
```

---

## 3. Environment and dependencies

| | |
|---|---|
| Python | ≥3.11 (venv at `.venv/`, 3.11.16) |
| Platform | Arch Linux, Hyprland/Wayland, low-resource: **2 CPU cores, 3.7 GB RAM** |
| torch | 2.14.0+**cpu** (AMD iGPU not used by torch) |
| opencv | 4.11 · numpy 1.26 · mediapipe 0.10.21 |
| Install | `pip install -e .` then `bash scripts/fetch_models.sh` |

Dependencies (pyproject.toml): `mediapipe>=0.10.21,<0.11`, `opencv-python>=4.8,<5`,
`numpy>=1.24,<2`, `torch>=2.2`. No web framework — the game uses stdlib
`http.server` only, deliberately, so installs on exhibition machines can't fail.

Console script: `signlang` → `signlang.cli:main`.

---

## 4. Letters subsystem — the pipeline

```
webcam → HandPipeline (hands.py, background thread, queue maxsize=2, newest wins)
       → MediaPipe HandLandmarker (VIDEO mode, 1 hand, GPU delegate optional)
       → normalize_hand(): mirror-fix for "Left" handedness → wrist-centre → palm-scale
       → landmark_features(): 74 floats
       → SignMLP: 74 → 192 → 112 → 26 (LayerNorm+GELU+Dropout 0.28 each hidden layer)
       → softmax (temperature-scaled) → probabilities
       → DwellRecognizer (engine.py): EMA smoothing → agreement → dwell → emit
       → transcript (text buffer) → OpenCV overlay (live.py) or HTTP game (game/*)
```

### 4.1 Feature extraction (`features.py`)

- **`normalize_hand(points, handedness)`** — flips x if handedness is "Left",
  subtracts the wrist, divides by wrist→middle-MCP distance. This translation+scale
  normalization is why recordings transfer across cameras and people.
- **`landmark_features(p)`** → **74 floats** (verified at runtime):
  63 raw coords (21×3) + derived geometric features — finger extensions
  (tip→MCP / palm), pinch (thumb-tip→index-tip / palm), fingertip spread, mean
  fingertip depth, palm normal (cross product).
- **`augment()`** — roll ±11°, pitch ±26°, yaw ±20°, finger jitter 0.014, noise
  0.006, re-normalized after every step. Disabled with `train --no-augment`.

### 4.2 Model (`model.py`)

`SignMLP(in_dim, n_classes, hidden=(192, 112), dropout=0.28)`; each hidden layer is
`Linear → LayerNorm → GELU → Dropout`. Checkpoint `models/signs_mlp.pt` stores:
`state_dict`, `labels`, `in_dim`, `n_classes`, `mean`/`std` (feature normalization
applied at inference: `(x-mean)/std`), `temperature`, `meta`.

Bundled checkpoint facts: **in_dim=74, n_classes=26, 10,570 samples, 20 epochs,
best_epoch=5, temperature=0.53, reported val_acc 0.9995** — but see §12: that
figure is from a *random row split* over burst-correlated data and is therefore
misleading.

`Predictor` wraps load + `probs()` for inference.

### 4.3 Dataset (letters)

- One `.npy` per sign in `data/samples/`, shape `(n, 21, 3)` float32, already
  wrist-centred and palm-scaled; a `.sources.npy` sidecar records which camera
  each sample came from.
- **10,570 samples across 26 labels: A–Y (no Z) + SPACE.**
  Counts are uneven: R 840, U 784, B/C/D/E 616 … T only **56** (~4 bursts).
  `train.py` compensates with per-class weights.
- Bursts: one SPACE press records `BURST_FRAMES = 14` near-identical frames
  (`BURST_FRAMES` is defined in `config.py:58`; `collect.py` imports it).
  **This burst structure is the root of the validation problem in §12.**

### 4.4 Dwell recognizer (`engine.py` — PROTECTED)

State machine per frame:

1. Probabilities from the model
2. Exponential moving average over `SMOOTH_MS = 420` ms
3. `agreement` = fraction of window frames whose argmax equals the current top label
4. Candidate change → reset dwell clock
5. Dwell starts only if: `agreement ≥ 0.7` **and** `movement ≤ 26px` **and**
   `conf ≥ CONFIRM_THRESHOLD (0.72)` **and** `conf - runner ≥ STABLE_MARGIN*0.5 (0.09)`
6. Emit when `t - dwell_start ≥ dwell_ms`

Key constants (`config.py`):

```python
DWELL_MS = 1000          # letters hold-to-confirm; --dwell= clamps 200–2500
SPACE_DWELL_MS = 800     # custom open-palm SPACE gesture
SMOOTH_MS = 420
CONFIRM_THRESHOLD = 0.72
STABLE_MARGIN = 0.18
REPEAT_COOLDOWN_MS = 900 # hold-once debouncing: one hold = one character
SWIPE_MIN_PX = 110       # right swipe = space, left swipe = backspace
SWIPE_MAX_MS = 420       # (needs ≥4 trail pts, ≥1.8× horizontal vs vertical, 1100 ms cooldown)
```

Deliberate design: recognition is **hold-to-confirm**, never instant, so output
doesn't flicker between confusable letters (A/E, M/N, U/V/R) while the hand moves.
The dwell timer is wall-clock based, so frame rate affects responsiveness but not
what gets recognized.

### 4.5 Capture & performance (`hands.py` — PROTECTED)

- Capture on a background thread with a small queue → newest frame always wins.
- `SIGNLANG_GPU=1` (default): MediaPipe runs on the **OpenGL/EGL GPU delegate**
  (~182 → ~70 ms/frame ≈ 2.6× on the AMD iGPU; auto-falls back to CPU; works
  headlessly via Mesa surfaceless EGL).
- `SIGNLANG_DETECT_EVERY=N`: run the detector every Nth frame, reuse landmarks
  in between (dwell timing unaffected). Measured: CPU every=1 → 5.5 fps,
  every=2 → ~11 fps, GPU+every=1 → ~14 fps.
- Landmark detection is ~94% of the frame budget; lowering `SIGNLANG_DETECT_HEIGHT`
  has **no** measured effect (tested 480/320/240 — all ~133 ms). Benchmark tool:
  `scripts/bench_landmarker.py` (camera-free).

### 4.6 Live overlay (`live.py`)

- All geometry derives from `u = max(2.0, min(w,h)/40.0)` so it scales identically
  from a 240×320 thumbnail to 1366×768.
- Panels: top bar (wordmark + A–Y alphabet strip), reading card (big letter, dwell
  ring, countdown, confidence, progress bar), runners-up panel (top-4 with bars),
  bottom panel (recent letters, transcript, gesture chip, fps, help).
- **Known trap:** `cv2.getWindowImageRect()` returns the size of the last *image
  drawn*, not the window — a fullscreen empty window measures 98×28 and the whole
  overlay collapses into the corner. The fix (`_screen_size()`) queries Hyprland
  via `hyprctl monitors all` (dividing by `scale:`), falling back to capture size.
  **Do not "simplify" this back.**
- Blending uses a bounded cache of solid images + `cv2.addWeighted`. LUT-based
  blending was tested and is **10× slower** — don't retry. Cached text sprites:
  not a win either.
- OpenCV `fontScale` is not pixels — a capital is ~22×scale px tall; everything
  goes through `_scale(px_height)` with `_PX_PER_SCALE = 22.0`.

### 4.7 Game (`src/signlang/game/`)

```bash
signlang game --host 0.0.0.0 --port 8000   # 0.0.0.0 to reach from a phone
```

- **No video stream.** The browser polls JSON (`GET /api/state`); the camera
  corner is a single JPEG from `GET /api/preview.jpg` a few times/second — a long
  stream could wedge the page. Other endpoints: `/`, `/static/*`, `/api/pool`,
  `/api/config`; POST for start/apply/shutdown.
- **Threading contract:** `tick_loop` ~60 Hz (animation/phase timing) and
  `LetterReader` ~10 fps never wait on each other; the detector lock is held for
  microseconds and publishes plain scalars. A dropped frame **keeps the previous
  reading** rather than reporting blank (a dropout must not look like the player
  leaving).
- **Phases:** `idle → ready (2200 ms, NOT scored) → answer (2000 ms, scored) →
  correct (900 ms) → ready … → wrong (run over)`. `ready` exists because a scored
  window starting the instant the letter appears punishes reaction time. A wrong
  shape costs time but doesn't end the run; only the answer window closing does.
  A low-confidence detection resets the hold but is **not** a strike (never punish
  the player for the model's own uncertainty).
- Hold requirement **420 ms**, min confidence **0.62** (`scoring.py`).
- Canvas dino game in `static/game.js` (~763 lines): constants `GROUND=0.80`,
  `DINO_X=0.12`, `CACTUS_START=1.15`, `CACTUS_CLOSE=0.34`, `JUMP_MS=620`; ASCII
  bitmap sprites compiled to horizontal runs and `fillRect`-ed; optimized to be
  pixel-identical across 300 frames at DPR 1 and 2.
- `game/pool.py` excludes some letters as "too confusable"/"too thin" — these are
  **guesses**, never backed by evaluation (see §12).

---

## 5. Words subsystem — the pipeline

```
webcam → MediaPipe Holistic (model_complexity=0, 1 hand… full body)
       → extract_landmarks_masked(): 51 landmarks (9 body + 21 left + 21 right) + validity mask
       → impute_invalid(): dropped hand anchored to its wrist / body wrist / body mean
       → motion energy (centroid displacement, EMA-smoothed)
       → segmentation state machine: REST → SIGNING → CONFIRMED
       → resample segment to 30 frames → normalize_sequence(per_frame=True)
       → DTWKNNClassifier.predict_single() → word + confidence, or rejected with a reason
       → transcript + HUD
```

### 5.1 Geometry & normalization (`normalize.py`)

- **Landmark layout (51×3):** block 0–8 = body `[NOSE, L_SHOULDER, R_SHOULDER,
  L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP]`; 9–29 = left hand (wrist at
  9); 30–50 = right hand (wrist at 30). `TOTAL_LANDMARKS=51`, `FEATURE_DIM=153`.
- **Validity:** pose landmarks valid if visibility ≥ 0.5 (and seen = |x|>1e-4 or
  |y|>1e-4); hands valid if any coordinate magnitude > 1e-4.
- **`impute_invalid`:** body → mean of tracked body points; each hand → own wrist
  if valid, else body wrist, else body mean. Prevents zeros from poisoning
  normalization (regression-tested).
- **`normalize_landmarks`:** origin = midpoint of the two shoulders, scale =
  inter-shoulder distance (degenerate `<1e-6` → 1.0):
  `(pts − midpoint) / ‖L−R‖`. Guarantees translation + scale invariance; shoulder
  midpoint → origin, inter-shoulder distance → 1.0. Per-frame by default;
  `per_frame=False` uses median midpoint/scale across valid frames.

### 5.2 Dataset (`dataset.py`)

- Storage: `words/data/samples/<WORD>.npy` shape `(M, 30, 51, 3)` float32 +
  `<WORD>.sources.npy` object array of provenance tags (`"synthetic"`,
  `"webcam0"`, …) in the same order.
- `save_word_samples()` accepts `(T,D)`, `(N,T,D)`, `(T,N,C)`, `(B,T,N,C)` and
  normalizes to `(N,30,51,3)`; appends by default (so recording never overwrites).
- **Synthetic generator:** `_generate_synthetic_sequence_for_word()` hand-builds a
  torso template + word-specific wrist trajectories + hand states
  (0=open, 1=fist, −0.5=pinch) for **every one of the 25 vocabulary words**, then
  jitters amplitude (0.90–1.10), adds drift N(0,0.02), per-landmark noise
  N(0,0.005), time-warps (`warp_factor ∈ (0.85,1.15)` via `np.interp`), and
  finally shoulder-normalizes per frame.
- **Current data: 87 samples total, only 16 of 25 words have any data**
  (HELLO 12; every other present word 5 — all synthetic except recent HELLO
  webcam additions). The 9 words with **zero** samples: GOOD_MORNING, MORNING,
  AFTERNOON, NIGHT, HOW, WELCOME, HAVE, WATER, FOOD.

### 5.3 Vocabulary (`words/config.py`)

`WORDS` has **25** entries (the comment saying "24-word" is stale):
`HELLO, THANKYOU, PLEASE, YES, NO, HELP, SORRY, GOOD, BAD, NAME, MORE, LOVE,
EAT, DRINK, WHERE, FINISHED, GOOD_MORNING, MORNING, AFTERNOON, NIGHT, HOW,
WELCOME, HAVE, WATER, FOOD` — plus `WORD_GLOSS` mapping each to display text.

### 5.4 Classifier (`classifier.py`) — DTW + calibrated kNN

`DTWKNNClassifier(BaseEstimator, ClassifierMixin)` — scikit-learn compatible,
instance-based (the "model file" is literally the training set).

- **Distance:** Dynamic Time Warping on framewise Euclidean cost, normalized as
  `D(N,M)/(N+M)`. Optional Sakoe–Chiba band: `w = max(window, |N−M|)`.
- **`dtw_distances_to_set(query, refs, window, chunk=64)`** — the hot path:
  vectorized DP across all references in batches of 64. Pairwise costs via the
  expansion `‖a−b‖² = ‖a‖² + ‖b‖² − 2ab` computed in **float64** (avoids
  cancellation residuals) then cast to float32; DP loop sequential over query time
  with preallocated rows and in-place `np.minimum`/`np.add`. Results identical to
  per-pair DTW.
- **Calibration (`_calibrate`, at `fit` time):** leave-one-out nearest-neighbour
  distances → `intra_scale_` = median distance to nearest *same-class* sample,
  `inter_scale_` = median distance to nearest *different-class* sample.
- **Prediction (`predict_single`):**
  1. Distances to all stored references.
  2. Top-k (k=3) inverse-distance weighted vote → `vote_prob` per class.
  3. **Winner chosen by class-level minimum distance** (not vote share);
     `separation = clip((runner_up − best)/runner_up, 0, 1)`.
  4. **`confidence = vote_prob[best] × separation`** — scale-free, so it doesn't
     collapse when absolute distances are large or small.
  5. **Two rejection gates:** (a) absolute gate — reject if
     `best_dist > MAX_ACCEPT_DISTANCE_RATIO (8.0) × inter_scale_` →
     `"too far from all references"`; (b) reject if
     `confidence < CONFIDENCE_THRESHOLD (0.18)` → `"low confidence"`.
     Returns `(None, conf, details)` with `reject_reason` on rejection.
- **`details` dict:** `raw_label, vote_prob, best_dist, runner_up_dist, separation,
  intra_scale_, inter_scale_, top_indices/distances/labels, class_probs,
  reject_reason` — surfaced in the live HUD diagnostics panel.
- **Persistence:** compressed `.npz` (`words/models/words_classifier.npz`) with
  keys `X, y, classes, n_neighbors, confidence_threshold, distance_scale, window,
  intra_scale, inter_scale, max_accept_distance_ratio` (sentinel −1 for None).
  Loading an old model without scales recalibrates with a printed warning.

This design (separation-based confidence + absolute distance gate + LOO scales)
was introduced to fix a real bug: confidence used to be crushed by absolute
distance magnitude, rejecting valid signs. Regression tests: `tests/test_detection.py`.

### 5.5 Segmentation state machine (`words/live.py`)

Runs per frame (landmark detection on every `FRAME_SKIP = 2`nd frame):

- **Motion energy** = max displacement of **hand centroids** between frames
  (centroid-based, not averaged over all 44 points — averaging diluted wrist
  motion below threshold; regression-tested). EMA: `smooth = 0.7·smooth + 0.3·inst`.
  A hand counts as present only with ≥ `MIN_PRESENT_HAND_LANDMARKS (10)` tracked
  points.
- **States:**
  - `REST`: when `smooth_energy ≥ ENERGY_ACTIVE_THRESHOLD (0.020)` → `SIGNING`,
    seed `active_segment` with the last **5** sliding-window frames, reset
    `peak_energy`/`dropped_frames`.
  - `SIGNING`: append frames, track `peak_energy`. Triggers:
    1. `motion_stopped`: energy `< 0.010` (`ENERGY_QUIET_THRESHOLD`) and len ≥ 10
    2. `velocity_valley`: len ≥ 14, energy `< 0.5 × peak`, peak ≥ 0.020
    3. `buffer_full`: len ≥ `SEQUENCE_LENGTH (30)`
  - On trigger: if `dropout_frac > MAX_DROPOUT_FRAME_FRACTION (0.25)` (frames
    with missing shoulders) → discard segment, reset. Else resample to 30 frames
    (`np.interp`), normalize per-frame, classify.
    - Accepted (`pred not None and conf ≥ threshold`) → append to transcript,
      `CONFIRMED`, `cooldown = COOLDOWN_FRAMES (15)`, reset.
    - Rejected via `motion_stopped`/`buffer_full` → print diagnostics, back to `REST`.
    - Rejected via `velocity_valley` only → **keep accumulating** (the sign may
      still be evolving). This asymmetry is deliberate.
  - `CONFIRMED`: cooldown counts down; at 0 → `REST` (hold-once semantics).
- Keys: `Q`/`Esc` quit, `C` clear transcript, `D` toggle diagnostics panel.
- Warns at startup if the dataset is >50% synthetic (`_warn_if_synthetic_heavy`).

### 5.6 Recording (`words/record.py`)

```bash
python main.py --mode words record              # interactive webcam
python main.py --mode words record --synthetic  # generate synthetic samples
```

Flags: `--camera` (default), `--synthetic/--generate`, `--samples N` (default 10),
`--word=HELLO,YES` (subset), `--source` (camera index or file), `--list`
(per-word counts), `--seed` (default 42).

Flow: IDLE → 2 s COUNTDOWN → RECORDING until 30 frames captured → normalize
per-frame → `save_word_samples(..., source="webcam0", append=True)`.
Keys: SPACE (capture), N/B (next/previous word), Q/ESC (quit).
Note: record flips the frame horizontally; live does not.

### 5.7 Evaluation (`words/eval.py`)

```bash
python main.py --mode words eval [--cv 5] [--k 3] [--threshold X] [--window N] [--save-model]
```

Two protocols:

1. **`evaluate_holdout_by_source`** — train on `synthetic`, test on all `real`
   samples. This is the honest estimate of live performance (prevents
   synthetic-template leakage). Reports accuracy, coverage (fraction accepted),
   per-sample `true->pred` strings, calibration scales.
2. **`evaluate_cross_validation`** — stratified k-fold (`random_state=42`,
   k clamped to the smallest class), reports mean±std accuracy, **mean per-sample
   latency (ms)**, full `classification_report`, confusion matrix including an
   `UNKNOWN` column for rejections.

Both print provenance counts and warn loudly when synthetic dominates real data
(CV becomes optimistic — it's mostly template recall).

---

## 6. Dispatcher & isolation guarantee (`main.py`)

```bash
python main.py                       # letters live (default)
python main.py --mode words          # words live
python main.py --mode words record
python main.py --mode words eval
python main.py --mode letters --source 1 <subcommand>
```

- Parses `--mode {letters,words}` (default `letters`), `--source`.
- **Letters mode NEVER imports `words`:** it purges `words` from `sys.modules` if
  present, sets `SIGNLANG_SOURCE`/`SIGNLANG_CAMERA` env, then routes to
  `signlang.cli.main` (known subcommands: live/game/train/collect/cameras/check/
  export/import/help) or falls through to `signlang.live.main`.
- Words mode dynamically imports `words` and calls
  `words.main(source=..., extra_args=...)`, whose first positional token selects
  `record` / `eval` / default `live`.
- Isolation is enforced by tests: `words/tests/test_isolation.py` spawns
  subprocesses and asserts `words` never appears in `sys.modules` when letters
  code runs.

---

## 7. Letters CLI reference

```bash
signlang collect [--only=A,B,C]   # record bursts: SPACE capture, N/B nav, R clear, Q quit+save
signlang train [--epochs=120] [--lr=1e-3] [--batch=64] [--val-frac=0.2] [--no-augment]
signlang live [--dwell=1000] [--sensitivity=0.4] [--windowed]
signlang game [--host H] [--port P] [--source S] [--letters] [--no-camera]
signlang cameras                   # list video devices
signlang check                     # fps + hand-detection rate diagnostic
signlang export [--no-model] [--out=F] / signlang import [--dry-run] [--from=F]
```

Live keys: hold letter → char after dwell; hold open-palm SPACE gesture → one
space; right swipe → space; left swipe → backspace; `C` clear; `Q`/`Esc` quit;
`F` fullscreen toggle.

Environment variables (letters):

| Variable | Default | Meaning |
|---|---|---|
| `SIGNLANG_CAMERA` | `0` | device index |
| `SIGNLANG_SOURCE` | — | any OpenCV-readable source (incl. HTTP MJPEG), overrides index |
| `SIGNLANG_WIDTH`/`HEIGHT` | 640/480 | capture size |
| `SIGNLANG_DETECT_HEIGHT` | = capture | landmark resolution (measured: no speed effect) |
| `SIGNLANG_MIRROR` | `1` | selfie mirroring |
| `SIGNLANG_MAX_HANDS` | `1` | hands tracked |
| `SIGNLANG_GPU` | `1` | GPU delegate for detection (~2.6×; auto-fallback) |
| `SIGNLANG_DETECT_EVERY` | `1` | detect every Nth frame (dwell unaffected) |
| `SIGNLANG_READ_TIMEOUT` | `3.0` | seconds |

Transfer bundles (`signlang export`) carry `data/samples/*.npy` (+ sources) and
`models/signs_mlp.pt` + `labels.json`; import **appends** recordings (importing
twice doubles counts). The 7.8 MB landmarker never travels — both machines fetch
it. Bundles are <1 MB because they hold normalized coordinates, not images.

---

## 8. Words config constants (quick table)

| Constant | Value | Meaning |
|---|---|---|
| `SEQUENCE_LENGTH` | 30 | frames per classified segment |
| `FRAME_SKIP` | 2 | detect every 2nd frame |
| `FEATURE_DIM` / `TOTAL_LANDMARKS` | 153 / 51 | 51×3 |
| `CONFIDENCE_THRESHOLD` | 0.18 | accept gate (separation-based confidence) |
| `MAX_ACCEPT_DISTANCE_RATIO` | 8.0 | absolute gate: best_dist ≤ 8 × inter_scale |
| `KNN_NEIGHBORS` | 3 | inverse-distance vote count |
| `ENERGY_ACTIVE_THRESHOLD` | 0.020 | REST → SIGNING |
| `ENERGY_QUIET_THRESHOLD` | 0.010 | "motion stopped" trigger |
| `COOLDOWN_FRAMES` | 15 | post-confirm refractory period |
| `MAX_DROPOUT_FRAME_FRACTION` | 0.25 | discard segment above this |
| `MIN_PRESENT_HAND_LANDMARKS` | 10 | hand "present" for centroids |
| `POSE_VISIBILITY_MIN` | 0.5 | pose validity |
| `MODEL_COMPLEXITY` | 0 | MediaPipe Holistic lightest model |

---

## 9. Tests and how to run them

```bash
# pytest (words suite + letters debounce) — currently 61 pass, 1 FAIL (stale, §14)
.venv/bin/python -m pytest

# letters-side self-tests (run and passing as of this doc)
.venv/bin/python -m signlang.selftest                # rotation invariance, renormalisation
.venv/bin/python -m signlang.test_live_layout        # 84 overlay render cases
.venv/bin/python -m signlang.test_debounce           # SPACE hold-once (9 tests, unittest)
PYTHONPATH=src .venv/bin/python -m signlang.game.selftest         # 52 assertions
PYTHONPATH=src .venv/bin/python -m signlang.game.selftest_server  # HTTP API in-process
```

**Test inventory:**

- `words/tests/test_classifier.py` (12) — DTW identity/symmetry/non-negativity,
  time-warp behaviour, batch matrix, fit/predict, threshold rejection, save/load,
  empty-query edge cases, windowed symmetry.
- `words/tests/test_normalize.py` (9) — shoulder midpoint/distance invariants,
  translation/scale invariance, degenerate shoulders, per-frame vs median
  normalization.
- `words/tests/test_detection.py` (17) — the regression suite for the calibration
  rewrite (confidence independent of absolute distance scale, distance gate
  default, noise/ambiguity rejection, class-level distances) **and** motion-energy
  correctness (centroids not diluted, static=no energy, one-handed triggers,
  hand-appearing produces no spike, validity masks), plus masking/imputation tests.
- `words/tests/test_recognition.py` (6) — resampling, simulated multiclass
  recognition (acc ≥ 0.90 on held-out synthetic), noise rejection, headless
  `run_live` on a dummy video.
- `words/tests/test_dataset.py` (3) — accepted shapes, invalid shape raises, empty
  dataset shapes.
- `words/tests/test_isolation.py` (6) — subprocess isolation, dispatcher routing,
  help, words standalone. **One assert is stale (§14).**
- `src/signlang/test_live_layout.py` — intercepts every `cv2` draw call and
  asserts geometry: nothing out of frame, no transcript/status collision, alphabet
  never overlaps the wordmark, dwell ring inside the card, etc. 6 sizes × 14
  states = 84 cases + blend parity/letterbox/canvas/screen-size checks.
- `src/signlang/selftest.py` — classification accuracy invariant under rotation
  about each axis (300 renormalised samples).

**Established practice in this repo:** every new geometry/regression check must be
validated by *injecting the regression it is supposed to catch*. Several checks
were written wrong on the first attempt and passed while broken (alphabet filter
used the wrong tuple index; letterbox measured canvas aspect not content aspect;
progress-bar check matched the runners-up bars). A green suite that never went
red proves nothing.

---

## 10. Key algorithms — one-paragraph summaries

- **Wrist-centre/palm-scale (letters) & shoulder-centre/shoulder-scale (words):**
  normalize away translation and absolute scale so landmarks are comparable
  across cameras, distances, and people. Words mode additionally masks and
  imputes dropped landmarks before normalizing, so a lost hand doesn't create a
  huge spurious offset.
- **Dwell-to-confirm:** a prediction only emits after holding top position with
  high agreement/confidence/low movement for a wall-clock window — trades latency
  for stability against confusable classes.
- **DTW:** optimal alignment between two time series of possibly different
  lengths; normalized path cost makes segments of different durations comparable.
  Banded (Sakoe–Chiba) and batch-vectorized for real-time use.
- **kNN with calibrated, separation-based confidence:** prediction quality judged
  by the *relative* gap to the runner-up class scaled by vote share, with an
  absolute sanity gate anchored to the training set's inter-class distance scale.
  Designed to reject out-of-distribution input instead of guessing.
- **Motion-energy segmentation:** self-segmenting the video stream — detect the
  start of a sign by hand-centroid motion crossing a threshold, end it when motion
  stops / hits a valley / hits the buffer cap — so the user doesn't press a key.

---

## 11. Architecture decision record (why things are this way)

- **Two isolated subsystems in one repo:** fingerspelling (frame-wise, static) and
  word-signs (sequence, dynamic) have fundamentally different ML needs; forcing
  one model would be worse. Isolation is test-enforced because letters mode must
  stay importable without the words package (and vice versa).
- **Instance-based (DTW/kNN) for words:** only tens of samples per class —
  training a sequence network would overfit. kNN on DTW is data-efficient and its
  "training set" is human-inspectable.
- **Stdlib HTTP game server:** exhibition installs must not fail on a web-framework
  dependency.
- **Camera in one process, browser polls JSON:** only one process can own a
  webcam; avoids long-lived streaming failure modes.
- **Committed model + recordings:** a clone works immediately (`signlang live`).
- **No images stored:** privacy, and bundles stay tiny.

---

## 12. Known issues & methodological caveats

**Measured results (verified 2026-10-08 by actually running the evals):**

| Eval | Command | Result |
|---|---|---|
| Letters, burst-held-out (honest) | `.venv/bin/python eval_burst.py` | **100.0%** over 10,570 held-out frames / 755 bursts; worst letters Q 99.6%, W 99.6%, P 99.7%; only confusions W→J, R→X, Q↔P (1 each) |
| Letters, random row split (inflated) | inside `signlang train` | val_acc 0.9995 — meaningless (burst leakage) |
| Words, train synthetic → test real | `main.py --mode words eval` | **0.0% accuracy, 0.0% coverage** — all 7 real webcam HELLO samples rejected |
| Words, 5-fold CV (mostly synthetic) | same | 98.89% ± 2.22, ~20 ms/sample — optimistic (template recall) |

1. **Letters random-split validation is inflated** — `train.py` splits rows at
   random, but rows arrive in bursts of 14 near-duplicate frames, so its
   val_acc ≈ 0.999 means nothing. **`eval_burst.py` is the honest tool** (holds
   out whole bursts; `BURST_FRAMES = 14` lives in `signlang/config.py:58`) and
   has now been run: **100.0% on unseen bursts** (table above). Quote that
   number instead — with the caveat that it is same-person, same-camera.
2. **Game letter pool is contradicted by evidence.** `game/pool.py` excludes
   I, P, S, L, U, X ("too confusable") and T ("too thin") based on guesses.
   `eval_burst.py` grades **all 26 signs "safe"** (T=56 samples included). The
   exclusion lists should be replaced with the eval's recommendation: pool = all.
3. **Fixed:** the stale `test_isolation.py` assertion (hardcoded 16-word
   vocabulary) now checks consistency with `NUM_WORDS`. Suite: **62/62 pass**.
4. **Words model does not transfer to real signs (the critical issue).**
   Trained on 80 synthetic + 7 real sequences, it accepts **0 of 7** real
   webcam samples (rejected by the distance gate / confidence). 9 of 25
   vocabulary words have zero samples at all. Until real recordings exist for
   each word, live words mode will reject nearly everything — this is a data
   problem, not a code problem. Fix: `main.py --mode words record --camera
   --samples 20` for every word, then `eval --save-model` and check that the
   real-hold-out line rises above 0%.
5. **Model trained on one person's hands** (letters and words). Accuracy on other
   people/cameras is known to be worse; the workflow is collect → train → live.
6. **Confusable pairs:** A/E, M/N, U/V/R (letters); R/U/V trio also relevant for
   words.
7. ~~**`en_US-lessac-medium.onnx` (63 MB) is unused**~~ — TTS restored in `src/signlang/tts.py` using Piper ONNX voice and sounddevice. Speaks confirmed letters, spaces, and transcript read-aloud via 'R' key. Configured via `TTS_VOICE` and toggled via `SIGNLANG_TTS` environment variable.
8. **Stale documentation/comment nits:** ~~README says "A–H and K–N"~~ (fixed —
   now says A–Y + SPACE with measured numbers); `words/config.py` says
   "24-word" for a 25-word list; `signlang help` lists an `assess` command that
   `cli.py` doesn't implement (prints "unknown command"); `game/scoring.py`
   comment references the old 620 ms dwell (now 1000 ms); `docs/PROJECT_OVERVIEW.md`
   §12 claims "6 commits / eval_burst broken / game untracked" — all outdated.
9. **Performance ceiling:** MediaPipe detection dominates the frame budget; levers
   are the GPU delegate, `SIGNLANG_DETECT_EVERY`, and (protected code) a lighter
   model. Overlay optimization cannot materially improve fps.
10. ~~Uncommitted work~~ — everything was committed and pushed (see §14).
   `words/models/words_classifier.npz` was re-saved on 2026-10-08 with the
   current code (87 refs, calibrated scales, threshold 0.18); the previous file
   was an old-format model predating calibration (threshold 0.65, no scales).

---

## 13. Git history (24 commits, chronological, all pushed to GitHub)

```
6d000b5 Add recordings for I through Y and retrain the 25-sign model
3c6e759 Add MIT LICENSE
fd82c7d Add readme and license metadata to project table
de527cc Add a browser game, retrain on new R and U samples, rebuild the live overlay
d1446b7 Add custom SPACE gesture with hold-once debouncing and cooldown
855424c Add words config, shoulder-relative normalization, and invariant tests
67de60e Implement vectorized DTW distance and scikit-learn compatible kNN classifier
1ed2388 Implement words dataset management, synthetic sequence generator, and recorder
2651795 Add words cross-validation evaluation script and pre-trained model
b786e59 Implement words live recognition with motion energy detection and HUD overlay
2eca09d Add main.py dispatcher and tests for isolation and sequence recognition
51c3f00 Optimize DTW with 2-row DP, fix empty query edge cases, and correct dataset dimensions
5a164d2 Support words subcommands, filter landmark dropout, and enhance continuous sign segmentation
206cbac Add unit tests for dataset shapes, live video execution, and subcommand isolation
150f774 Support --camera and --generate flags in words record CLI
```

(The first 5 shown are the earliest; history was built up letters-first, then the
words subsystem, then isolation/tests.)

---

## 14. Change set shipped in PR #1 (was uncommitted; committed 2026-10-08)

Pushed to `github.com/RAZOR-NINJAS/SignLang` via PR #1 (merged into `main`).
Original diff — 12 files, **+1111 / −215**, plus 3 new files:

| File | Change |
|---|---|
| `words/config.py` | vocabulary 16 → **25** words; `CONFIDENCE_THRESHOLD` 0.65 → 0.18; added `MAX_ACCEPT_DISTANCE_RATIO=8.0`; energy thresholds 0.035/0.015 → 0.020/0.010; dropout/visibility constants added |
| `words/classifier.py` | LOO calibration (`intra_scale_`/`inter_scale_`), vectorized batch DTW with float64 cost expansion + Sakoe–Chiba band, class-level winner + separation-based confidence, absolute distance gate, richer `reject_reason` details, extended save/load |
| `words/normalize.py` | **new** (133 lines): masked extraction, invalid-landmark imputation, shoulder-relative normalization |
| `words/dataset.py` | provenance sidecars (`.sources.npy`), robust multi-shape save, synthetic generation for all 25 words |
| `words/eval.py` | `evaluate_holdout_by_source` (train-synthetic/test-real), provenance reporting + warnings |
| `words/live.py` | masked+imputed extraction, centroid motion energy, velocity_valley trigger, dropout rejection, calibration-aware HUD diagnostics, synthetic-heavy warning |
| `src/signlang/config.py` | **`BURST_FRAMES = 14` moved here** (fixes eval_burst import), `GPU_DELEGATE`, `DETECT_EVERY` |
| `src/signlang/hands.py` | GPU delegate support + detection frame-skip (+78 lines) |
| `README.md` | documents `SIGNLANG_GPU`, `SIGNLANG_DETECT_EVERY`, updated perf guidance |
| `words/data/samples/HELLO.*` | grew: more webcam samples recorded |
| `words/models/words_classifier.npz` | retrained/rebuilt |
| **untracked** | `scripts/bench_landmarker.py`, `scripts/rebuild_words.py`, `words/tests/test_detection.py` |

---

## 15. Rules for AI assistants working in this repo

1. **`src/signlang/hands.py` and `src/signlang/engine.py` are PROTECTED** — do not
   modify without explicit user permission. Everything else is fair game.
2. **Never run the camera-backed app yourself.** Provide commands as text; the
   user runs them. Synthesizing frames and testing pure functions is fine.
   External `curl` to localhost may be sandbox-blocked — use in-process HTTP tests.
3. **Verify tests by injecting the regression they should catch** (§9). Never
   trust a green test that was never observed failing.
4. Don't "simplify" the Hyprland screen-size workaround back to
   `cv2.getWindowImageRect` (§4.6).
5. Don't retry LUT blending or sprite caching for the overlay (measured losses).
6. Environment is low-resource (2 cores, 3.7 GB RAM, CPU-only torch) — prefer
   measured micro-benchmarks (`scripts/bench_landmarker.py`) over assumptions.

---

## 16. Suggested next steps

1. **Record real samples for words mode — this is the blocker.** The honest
   hold-out (train synthetic → test real) accepts **0 of 7** real samples
   (§12). Record ≥20 sequences per word with `main.py --mode words record
   --camera`, re-run `eval --save-model`, and confirm the real-hold-out line
   is no longer 0% before demoing words mode.
2. **Update `game/pool.py` exclusions** — `eval_burst.py` grades all 26 signs
   "safe" (§12), contradicting the hardcoded exclude list.
3. ~~Run `eval_burst.py`~~ done — 100.0% burst-held-out (§12).
4. ~~Fix stale test assert~~ done — 62/62 (§12).
5. ~~Commit and push the pending work~~ done — PR #1 merged (§14).
6. Record more bursts for the thin letters (T=56) if you plan to demo with
   hands other than your own.
