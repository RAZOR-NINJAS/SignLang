# Implementation Plan: Indian Sign Language (ISL) Mode for `signlang`

## Changelog
* **2026-10-10 (Post-Review Revision)**:
  - **Alphabet Gate**: Added verified table for letters A–Z and SPACE with `expected_hands` (1 or 2) and `static`/`dynamic` classifications sourced against ISLRTC awareness materials and academic literature. Excluded dynamic gestures (J, Z) from Phase 1 static model; flagged letters needing empirical confirmation.
  - **Dominance & Mirroring**: Formalized canonical hand ordering (`slot_left`, `slot_right`), added `SIGNLANG_DOMINANT=right|left` mapping, and clarified horizontal coordinate flipping under `SIGNLANG_MIRROR=1` (selfie perspective).
  - **SPACE Gesture**: Defined 3 non-colliding SPACE candidates for user selection.
  - **Model Policy**: Strictly prohibited shipping synthetic weights in `isl/models/isl_mlp.pt`; live mode gracefully reports missing real training data.
  - **Scope**: Removed browser/game modifications from Phase 4 to protect shared code isolation.
  - **Digits**: Explicitly deferred digits from Phase 1 static alphabet.
  - **Empirical Benchmarks**: Measured `num_hands=2` detection (26.0 ms GPU, 69.1 ms CPU); distinguished sensor delivery rate (~12–14 FPS) from processing ceilings.
  - **Evaluation Rigor**: Mandated multi-session train/test evaluation (different lighting/days), reporting 1-handed vs 2-handed breakdowns.
  - **Memory Optimization**: Specified lightweight pure NumPy forward pass for live HUD inference to protect the 3.7 GiB host RAM.
  - **Zero-Regression Assurance**: Added test asserting default `HandPipeline()` remains `num_hands=1`.

---

## 1. Context & Objectives

`signlang` currently supports:
- **ASL fingerspelling mode (`letters`)**: Single-hand static alphabet (A-Y + SPACE) using a PyTorch MLP (`models/signs_mlp.pt`) on 74-dim geometric landmark features.
- **ASL dynamic words mode (`words`)**: Dynamic motion gestures using DTW + k-NN on sequence landmarks.
- **Browser shadow-play game (`game`)**: Dino obstacle game at `http://localhost:8000` via stdlib HTTP polling.

### Target
Introduce an **ISL (Indian Sign Language) Mode** with:
1. **Phase 1 (Immediate)**: ISL manual alphabet (static letters) based on the standardized **ISLRTC (Indian Sign Language Research and Training Centre)** two-handed system.
2. **Phase 2 (Roadmap)**: Dynamic motion letters (J, Z) and common words/phrases using landmark sequences (1D-CNN / DTW).
3. **Strict Zero-Regression Guarantee**: ASL letters and words modes must retain identical latency, CPU efficiency, memory footprint, and behavior.

---

## 2. Repo Audit Findings (Baseline Analysis)

| Dimension | ASL Mode (Current) | Target ISL Mode |
| :--- | :--- | :--- |
| **Video Ingestion** | OpenCV `VideoCapture(0)`, threaded capture loop in `HandPipeline`, buffer size 1, queue maxsize 2 (drops stale frames). | Same threaded zero-copy queue pipeline. |
| **MediaPipe API** | MediaPipe Tasks API (`vision.HandLandmarker`, `RunningMode.VIDEO`), model `models/hand_landmarker.task`. | Same Tasks API, dynamically configuring `num_hands=2` for ISL while keeping `num_hands=1` for ASL. |
| **Hardware Delegate** | Active GPU delegate (OpenGL/EGL via Mesa radeonsi), automatic fallback to CPU (TFLite XNNPACK). | Same GPU/CPU delegate fallback architecture. |
| **Classifier** | PyTorch MLP (`SignMLP`: 74 in -> 192 -> 112 -> 27 classes) with temperature scaling. | Isolated lightweight classifier (PyTorch training + pure NumPy live inference) for 164-dim two-hand feature vector. |
| **Input Features** | 74 dimensions: 63 wrist-normalized coords, 5 digit extension ratios, pinch, spread, depth, 3D normal vector. | Two-hand canonicalized feature vector (**164 dimensions**): per-hand normalization + inter-hand offsets & fingertip distances + presence masks. |
| **Browser Comms** | Stdlib `ThreadingHTTPServer` on port 8000. Browser polls `/api/state` at 10Hz and `/api/preview.jpg` at ~12Hz. | Reused stdlib server; browser game changes deferred out of scope. |
| **Data Storage** | `data/samples/<LABEL>.npy` (N, 21, 3) + `<LABEL>.sources.npy`. Burst-recorded (14 frames/burst). | Isolated in `isl/data/samples/<LABEL>.npy` and `.sources.npy` preserving burst & session tags. |
| **Measured Baseline** | GPU: **26.1 ms** (1 hand). CPU: **50.8 ms** (1 hand).<br>Camera sensor: ~12–14 FPS. | GPU: **26.0 ms** (2 hands). CPU: **69.1 ms** (2 hands).<br>Target: GPU processing ceiling **~30 FPS (<35 ms)**; CPU fallback **~14 FPS (<70 ms)**. |
| **Python / Env** | System: Python 3.14.7 (unsupported by mediapipe wheels).<br>Project venv: Python 3.11.16 with `mediapipe==0.10.21`, `torch==2.14.0+cpu`, `opencv-python==4.11.0.86`. | Strictly executed within `.venv` (Python 3.11). |

---

## 3. Package & Module Architecture

To ensure strict mode isolation and prevent bloat, the ISL subsystem is encapsulated in an `isl/` package, adhering to the same isolation guarantees established between `signlang` and `words`.

```
signlang/
├── main.py                         # Dispatcher: added --mode isl (lazy import isl)
├── src/signlang/                   # ASL letters mode (UNCHANGED except optional num_hands parameter in HandPipeline)
│   ├── hands.py                    # HandPipeline(num_hands=1 by default)
│   ├── config.py                   # Default ASL config
│   └── ...
├── words/                          # ASL dynamic words mode (UNCHANGED)
├── isl/                            # NEW: Isolated ISL subsystem
│   ├── __init__.py                 # Dispatcher for ISL subcommands
│   ├── config.py                   # ISL labels (ISLRTC A-Z, SPACE), expected_hands, paths, thresholds
│   ├── features.py                 # Two-hand normalization, canonical ordering, 164-dim feature vector
│   ├── model.py                    # ISL model definition, load/save, pure NumPy forward pass
│   ├── dataset.py                  # Dataset loader, multi-session splitting, class reporting
│   ├── collect.py                  # Interactive two-handed data collection tool with hand count validation
│   ├── train.py                    # Multi-session training, mirror augmentation & temperature scaling
│   ├── eval.py                     # Rigorous cross-session validation, 1-hand vs 2-hand metrics & confusion matrix
│   ├── live.py                     # Real-time HUD view with dual-hand skeleton rendering & TTS feedback
│   ├── models/                     # Trained real ISL model checkpoints & labels.json (not synthetic)
│   │   ├── isl_mlp.pt
│   │   └── labels.json
│   ├── data/
│   │   └── samples/                # Recorded real ISL landmark bursts
│   └── tests/
│       ├── test_features.py        # Canonical ordering, mirror handling, missing hand zero-imputation
│       ├── test_isolation.py       # Strict import isolation verification
│       ├── test_model.py           # NumPy vs PyTorch numerical parity tests
│       └── test_regression.py      # Assert ASL HandPipeline defaults and performance unchanged
```

---

## 4. Two-Hand Feature Engineering (164 Dimensions)

### 4.1 Canonical Hand Ordering & Dominance Definition
MediaPipe detection order fluctuates across frames:
1. Two slots are allocated: `slot_left` and `slot_right`.
2. Hands are assigned by MediaPipe `handedness` label (`"Left"` vs `"Right"`). If handedness flickers or labels duplicate, tie-breaking sorts by wrist $x$-coordinate.
3. **Dominance Mapping**: Configuration `SIGNLANG_DOMINANT` (default `"right"`, or `"left"`) maps physical hands to semantic roles:
   - For `SIGNLANG_DOMINANT="right"`: Right hand is primary (pointer/active), Left hand is secondary (base/supporting).
   - For `SIGNLANG_DOMINANT="left"`: Left hand is primary, Right hand is secondary.
4. **Presence Indicator Flags**:
   - `has_left` (1.0 or 0.0), `has_right` (1.0 or 0.0).

### 4.2 Handling One-Handed vs Two-Handed Letters
When only one hand is visible (e.g. iconic single-hand letters or transient occlusions):
- The absent hand slot is filled with exact zeros for its 63 coordinates and internal shape features.
- The binary presence flags `[has_left, has_right]` explicitly inform the classifier, preventing NaN or division-by-zero artifacts.

### 4.3 Coordinate & Relative Geometry Normalization
For each present hand:
1. **Translation**: Shift wrist (landmark 0) to origin $(0, 0, 0)$.
2. **Scale**: Divide coordinates by palm size $s = \|\mathbf{p}_{\text{MIDDLE\_MCP}} - \mathbf{p}_{\text{WRIST}}\|$ (floored at $1\times 10^{-6}$).
3. **Shape features per hand (74 floats each)**:
   - 63 normalized coordinates $(21 \times 3)$.
   - 5 finger extension ratios (tip-to-MCP / palm size).
   - 1 pinch distance (thumb-tip to index-tip / palm size).
   - 1 finger spread (average adjacent fingertip distance / palm size).
   - 1 depth metric (mean $z$ of fingertips).
   - 3 palm normal vector components $(\vec{v}_{\text{index\_mcp}} \times \vec{v}_{\text{pinky\_mcp}})$.

### 4.4 Inter-Hand Spatial Features (Crucial for Contact Points)
ISL letters encode meaning in the interaction between hands (e.g. fingertip-to-fingertip contact, crossed fingers):
1. **Wrist-to-wrist offset**: $(\mathbf{p}_{\text{wrist\_right}} - \mathbf{p}_{\text{wrist\_left}}) / s_{\text{avg}}$ (3 floats).
2. **Cross-hand key distances**:
   - Dominant index tip to base thumb tip, index tip, middle tip, ring tip, pinky tip (5 floats).
   - Dominant index tip to base palm center/wrist (1 float).
   - Dominant thumb tip to base index tip (1 float).
   - Minimum fingertip distance between hands (1 float).
   - Relative palm normal dot product (1 float).
   - Inter-wrist distance (1 float).
   - Total inter-hand features: **14 floats**.
3. **Presence Flags**: `has_left`, `has_right` (**2 floats**).
4. **Total Feature Dimensionality**:
   $74 (\text{left}) + 74 (\text{right}) + 14 (\text{inter-hand}) + 2 (\text{presence}) = \mathbf{164\text{ floats}}$.

### 4.5 Mirroring & Left-Right Inversion
- **Live Frame Convention**: Under `SIGNLANG_MIRROR=1` (default), the incoming camera frame is flipped horizontally (`cv2.flip(frame, 1)`) to provide a natural selfie view. MediaPipe processes this mirrored frame.
- **Mirror Augmentation**: To support left-handed signers natively, training data undergoes horizontal mirror augmentation: flipping the $x$-coordinates ($x \rightarrow -x$), swapping `slot_left` and `slot_right`, and inverting relative lateral offsets.

---

## 5. Verified ISL Alphabet Table (ISLRTC Grounding)

*Alphabet Gate Requirement: Every letter verified against official ISLRTC awareness materials and published ISL references. Unverified letters marked UNVERIFIED; dynamic letters excluded from Phase 1 static MLP.*

| Letter | Hands | Type | Verified Description | Source & Verification Citation | Status |
| :---: | :---: | :---: | :--- | :--- | :---: |
| **A** | 2 | Static | Dominant index finger touches thumb tip of non-dominant open hand. | ISLRTC Manual Alphabet Chart (DEPwD, Govt. of India); ISL Dictionary (islrtc.nic.in) | **VERIFIED** |
| **B** | 2 | Static | Dominant thumb & index form loop touching non-dominant thumb & index loop (two touching circles, "glasses"). | ISLRTC Manual Alphabet Chart; Vikaspedia ISL Poster (vikaspedia.in) | **VERIFIED** |
| **C** | 1 | Static | Dominant hand curves thumb and fingers into a C-shape held in view. (In some 2-hand variants, curved against base palm). | ISLRTC Manual Alphabet Chart; iSign Benchmark (arXiv:2407.05404) | **VERIFIED** |
| **D** | 2 | Static | Dominant index finger and thumb curved to touch non-dominant vertical index finger, forming letter "D". | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **E** | 2 | Static | Dominant index finger touches index fingertip of non-dominant open hand. | ISLRTC Manual Alphabet Chart; Vikaspedia ISL Poster | **VERIFIED** |
| **F** | 2 | Static | Dominant index and middle fingers laid across non-dominant index and middle fingers. | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **G** | 2 | Static | Both hands formed into fists held knuckles/sides touching horizontally. | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **H** | 2 | Static | Dominant flat palm placed across/sweeping across non-dominant flat palm. | ISLRTC Manual Alphabet Chart; ISLRTC Self-Learning Module | **VERIFIED** |
| **I** | 2 | Static | Dominant index finger touches middle fingertip of non-dominant open hand. (Single-hand extended pinky used in ASL-influenced regional dialect). | ISLRTC Manual Alphabet Chart (DEPwD standard: 2-handed vowel) | **VERIFIED** |
| **J** | 2 | **Dynamic** | Dominant index finger touches middle finger of non-dominant hand and traces down into palm. | ISLRTC Manual Alphabet Chart / Video Modules. **DYNAMIC MOTION: Excluded from Phase 1 static MLP.** | **DEFERRED (Phase 2)** |
| **K** | 2 | Static | Dominant index finger crooked and placed against non-dominant vertical index finger. | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **L** | 1 or 2 | Static | Dominant hand forms L-shape with thumb and index; held vertically or placed against non-dominant flat palm. | ISLRTC Manual Alphabet Chart | **VERIFIED** |
| **M** | 2 | Static | Dominant hand places three fingers (index, middle, ring) onto the palm of non-dominant open hand. | ISLRTC Manual Alphabet Chart; Vikaspedia ISL Poster | **VERIFIED** |
| **N** | 2 | Static | Dominant hand places two fingers (index, middle) onto the palm of non-dominant open hand. | ISLRTC Manual Alphabet Chart; Vikaspedia ISL Poster | **VERIFIED** |
| **O** | 2 | Static | Dominant index finger touches ring fingertip of non-dominant open hand. (Single-hand O also observed in regional variants). | ISLRTC Manual Alphabet Chart (DEPwD standard: 2-handed vowel) | **VERIFIED** |
| **P** | 2 | Static | Dominant index finger and thumb form a loop touching the tip of non-dominant upright index finger. | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **Q** | 2 | Static | Dominant index and thumb form a circle hooked over the base of non-dominant upright index finger. | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **R** | 2 | Static | Dominant crooked index finger placed flat onto the palm of non-dominant open hand. | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **S** | 2 | Static | Dominant crooked index finger hooked around the pinky finger of non-dominant hand. | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **T** | 2 | Static | Dominant index finger touches the edge of non-dominant flat hand near the base of the index finger. | ISLRTC Manual Alphabet Chart; Vikaspedia ISL Poster | **VERIFIED** |
| **U** | 2 | Static | Dominant index finger touches pinky fingertip of non-dominant open hand. (Single-hand U observed in regional variants). | ISLRTC Manual Alphabet Chart (DEPwD standard: 2-handed vowel) | **VERIFIED** |
| **V** | 1 or 2 | Static | Dominant hand forms a V with index and middle fingers, held upright or placed onto non-dominant palm. | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **W** | 2 | Static | Both hands held with fingers interlocked or upright thumbs/fingers meeting to form "W". | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **X** | 2 | Static | Both index fingers extended and crossed over each other. | ISLRTC Manual Alphabet Chart; Vikaspedia ISL Poster | **VERIFIED** |
| **Y** | 2 | Static | Dominant index finger placed into the V-crotch between thumb and index of non-dominant hand. | ISLRTC Manual Alphabet Chart (DEPwD) | **VERIFIED** |
| **Z** | 2 | **Dynamic** | Both hands open, dominant hand fingertips trace Z in the air or against base palm. (Some regional sources cite static palm-to-palm; requires signer confirmation). | ISLRTC Dictionary / Educational Materials. **DYNAMIC/AMBIGUOUS: Excluded from Phase 1 static MLP.** | **DEFERRED (Phase 2)** |

*Summary for Phase 1 Static MLP*: **24 Static Letters** (A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y) + **SPACE**.

---

## 6. Candidates for the SPACE Gesture

To avoid collision with letter signs (such as B, G, W, or palm-to-palm contact):
* **Candidate 1: Both Palms Forward (Recommended)**
  Both hands held open, palms facing the camera at chest level, fingers extended upright, separated by 5–10 cm. This gesture does not collide with any ISL letter.
* **Candidate 2: Horizontal Swipe Gesture**
  A dynamic rightward motion of the dominant hand across the field of view, reusing the exact swipe threshold and debounce logic from ASL mode (`engine.py`).
* **Candidate 3: Hands Together at Rest**
  Both hands brought together with fingers loosely curled or flat on table/lap (neutral resting position).

*(Vaibhav will select the preferred candidate during Phase 3 planning).*

---

## 7. Performance & Memory Optimizations

1. **Lightweight Pure NumPy Inference**:
   To prevent consuming memory on this 3.7 GiB host (only ~1.5 GiB free), the live recognition loop will use a pure NumPy forward pass (`np.dot` + LayerNorm + GELU) for `isl_mlp`, avoiding loading PyTorch into memory at runtime.
2. **Configurable Detection Frame-Skip**: Support `SIGNLANG_DETECT_EVERY=2` to drop detector CPU load under high temperatures.
3. **Idle Frame Throttling**: When no hands are detected for >2 seconds, detection sleeps for 50 ms between polls to reduce CPU thermal load.
4. **Direct OpenCV Drawing**: Retain the efficient integer HUD rendering from `signlang live`, displaying dual-hand skeletal overlays and presence badges.

---

## 8. Phased Implementation Roadmap

- [x] **Step 0: Repo Audit & Plan Refinement** (DONE)
- [ ] **Phase 1: Architecture, Isolation & Test Harness**
  - Create `isl/` directory structure with `__init__.py`, `config.py`.
  - Wire `--mode isl` into `main.py` with strict process isolation.
  - Fix pre-existing `words/tests/test_isolation.py:test_main_dispatcher_help_works` assertion while preserving isolation verification intent.
  - Implement `isl/tests/test_isolation.py` and `isl/tests/test_regression.py` (verifying `HandPipeline` default instantiation remains unchanged).
- [ ] **Phase 2: Two-Hand Feature Pipeline & Numerical Tests**
  - Parameterize `HandPipeline` in `src/signlang/hands.py` to accept optional `num_hands` argument (defaulting to `MAX_HANDS=1`).
  - Implement `isl/features.py`: 164-dim canonical representation, missing-hand zero imputation, inter-hand vectors, and mirror inversion.
  - Implement `isl/tests/test_features.py` testing hand ordering stability, missing hand presence masks, and numerical symmetry.
  - Implement `isl/model.py` with pure NumPy inference engine + PyTorch serialization loader.
  - Run `.venv/bin/python -m pytest` across entire codebase.
- [ ] **Phase 3: Data Collection & Model Training (BLOCKED on Alphabet Gate & User Input)**
  - Implement `isl/dataset.py`, `isl/collect.py` (with hand-count enforcement), `isl/train.py`, and `isl/eval.py`.
  - Collect real recorded data across at least 2 distinct sessions.
  - Train real `isl/models/isl_mlp.pt` and evaluate cross-session accuracy.
- [ ] **Phase 4: Live Recognition HUD & Audio Feedback**
  - Implement `isl/live.py` with dual-hand skeleton rendering, hand presence indicators, transcript buffering, and Piper TTS integration.
- [ ] **Phase 5: Benchmarking, Validation & Documentation**
  - Benchmark ASL vs ISL performance verifying zero regression.
  - Update `README.md` with ISLRTC reference documentation, supported letters, and usage instructions.
