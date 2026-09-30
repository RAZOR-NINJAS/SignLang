import sys
import time

import cv2
import numpy as np

from .engine import DwellRecognizer
from .features import normalize_hand
from .hands import HandPipeline, draw_landmarks
from .model import Predictor

WINDOW = "signlang - live"


def _wrap(text, width=34):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def main(argv=None):
    argv = list(argv or [])
    pipe = None
    try:
        predictor = Predictor()
    except FileNotFoundError as exc:
        print(exc)
        return 1

    rec = DwellRecognizer(predictor)
    for a in argv:
        if a.startswith("--dwell="):
            rec.set_dwell(int(a.split("=", 1)[1]))
        elif a.startswith("--sensitivity="):
            rec.set_threshold(1.0 - float(a.split("=", 1)[1]))

    print("=" * 58)
    print(f"  SIGNLANG LIVE   signs: {' '.join(predictor.labels)}")
    print(f"  dwell {rec.dwell_ms}ms   threshold {rec.threshold:.2f}")
    print("=" * 58)
    print("  Hold a letter still to confirm it. Swipe LEFT to erase,")
    print("  swipe RIGHT for a space.  Q quits, C clears.")
    print("=" * 58)

    pipe = HandPipeline()
    pipe.started.wait(timeout=12)
    if pipe.error is not None:
        print(f"camera error: {pipe.error}")
        return 1

    fps_t, fps_n, fps_v = time.time(), 0, 0.0
    try:
        while True:
            frame, landmarks, handedness = pipe.read(timeout=3.0)
            if frame is None:
                continue
            fps_n += 1
            if time.time() - fps_t > 0.5:
                fps_v = fps_n / (time.time() - fps_t)
                fps_t, fps_n = time.time(), 0

            h, w = frame.shape[:2]
            norm = None
            if landmarks is not None:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], np.float32)
                draw_landmarks(frame, landmarks)
                norm = normalize_hand(pts, handedness)

            st = rec.update(norm, handedness, w, h)

            disp = frame
            bar = 118
            cv2.rectangle(disp, (0, h - bar), (w, h), (18, 18, 22), -1)

            cand = st.get("candidate")
            conf = st.get("confidence", 0.0)
            dwell = st.get("dwell", 0.0)
            good = cand is not None and conf >= rec.threshold

            col = (120, 235, 255) if good else (140, 140, 150)
            cv2.putText(disp, cand or "-", (16, h - bar + 52), cv2.FONT_HERSHEY_SIMPLEX,
                        1.7, col, 3, cv2.LINE_AA)
            cv2.putText(disp, f"{conf:.2f}", (150, h - bar + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

            cv2.rectangle(disp, (240, h - bar + 26), (240 + 260, h - bar + 40), (50, 50, 56), -1)
            bw = int(260 * dwell)
            bcol = (90, 230, 140) if dwell > 0 else (50, 50, 56)
            cv2.rectangle(disp, (240, h - bar + 26), (240 + bw, h - bar + 40), bcol, -1)
            cv2.putText(disp, "dwell", (240, h - bar + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (130, 130, 140), 1, cv2.LINE_AA)

            lines = _wrap(st.get("buffer", ""))
            for i, ln in enumerate(lines[-2:]):
                cv2.putText(disp, ln, (16, h - 34 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (235, 235, 240), 1, cv2.LINE_AA)

            cv2.putText(disp, f"{fps_v:4.1f} fps", (w - 108, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 120), 1, cv2.LINE_AA)
            if st.get("gesture"):
                cv2.putText(disp, st["gesture"].upper(), (w - 108, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 230, 140), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k in (ord("c"), ord("C")):
                rec.clear()
            elif k == 8:
                rec.backspace()
    except KeyboardInterrupt:
        pass
    finally:
        if pipe is not None:
            pipe.close()
        cv2.destroyAllWindows()

    print(f"\n  transcript: {rec.buffer!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
