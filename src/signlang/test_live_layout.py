"""Layout checks for the live overlay, so the OpenCV window is not eyeballed.

A cv2 window cannot be unit tested for looks, but the failure mode that matters
is measurable: text running off the frame, panels colliding, and labels painted
straight onto the camera image where they inherit whatever contrast the room
happens to have. This intercepts every draw call and asserts its geometry.

Run: .venv/bin/python -m signlang.test_live_layout
"""
import re
import subprocess
import sys

import cv2
import numpy as np

from .config import DWELL_MS
from .live import (ALT_ROWS, FLASH_MS, HELP, RECENT_MAX, _blend,
                   _draw_overlay, _fit_to_window, _layout)

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXY")
SIZES = [(1280, 720), (1024, 576), (640, 480), (480, 640), (320, 240), (240, 320)]

CASES = {
    "nohand": {"gesture": None, "emitted": None, "candidate": None,
               "confidence": 0.0, "dwell": 0.0, "buffer": "", "top": []},
    "unclear": {"gesture": None, "emitted": None, "candidate": "A",
                "confidence": 0.44, "dwell": 0.0, "buffer": "",
                "top": [{"label": "A", "prob": .44}, {"label": "S", "prob": .31},
                        {"label": "E", "prob": .12}, {"label": "M", "prob": .06},
                        {"label": "N", "prob": .03}]},
    "holding": {"gesture": None, "emitted": "B", "candidate": "B",
                "confidence": 0.91, "dwell": 0.62, "buffer": "HI",
                "top": [{"label": "B", "prob": .91}, {"label": "R", "prob": .05},
                        {"label": "P", "prob": .02}, {"label": "D", "prob": .01}]},
    "gesture": {"gesture": "space", "emitted": " ", "candidate": "L",
                "confidence": 0.88, "dwell": 0.15, "buffer": "HELLO WORLD",
                "top": [{"label": "L", "prob": .88}, {"label": "G", "prob": .06},
                        {"label": "D", "prob": .03}, {"label": "O", "prob": .02}]},
    "long": {"gesture": "backspace", "emitted": None, "candidate": "M",
             "confidence": 0.71, "dwell": 0.05,
             "buffer": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AGAIN",
             "top": [{"label": "M", "prob": .71}, {"label": "N", "prob": .14},
                     {"label": "W", "prob": .09}, {"label": "H", "prob": .03}]},
    # Fingerspelled text usually has no spaces, so the unbreakable-word path
    # has to hold up too.
    "nospaces": {"gesture": None, "emitted": None, "candidate": "V",
                 "confidence": 0.80, "dwell": 0.30, "buffer": "HANDLANGJUMPS",
                 "top": [{"label": "V", "prob": .80}, {"label": "W", "prob": .11}]},
    "verylong": {"gesture": None, "emitted": None, "candidate": "T",
                 "confidence": 0.66, "dwell": 0.0, "buffer": "A" * 240,
                 "top": [{"label": "T", "prob": .66}, {"label": "I", "prob": .2}]},
}

# The recent-letters strip and the confirmation flash are only drawn when the
# caller supplies them, so they need their own cases rather than a state flag.
RECENT = ["K", "B", "M", "A", "R", "D", "H", "S", "L", "O", "P"]
EXTRA = [
    ("recent-full", {}, RECENT, None),
    ("recent-one", {}, ["B"], 40.0),
    ("flash-start", {}, ["K"], 0.0),
    ("flash-mid", {}, ["B"], FLASH_MS / 2),
    ("flash-end", {}, ["M"], FLASH_MS - 1.0),
    ("flash-faint", {}, ["M"], FLASH_MS * 0.97),
    ("flash-norecent", {}, [], 100.0),
]

_boxes = []


def _install_probes():
    """Record the bounding box of every draw call, then draw for real."""
    real = {"putText": cv2.putText, "rectangle": cv2.rectangle,
            "circle": cv2.circle, "ellipse": cv2.ellipse}

    def putText(img, text, org, font, scale, color, thick=1, line=cv2.LINE_AA):
        (tw, th), _ = cv2.getTextSize(str(text), font, scale, thick)
        _boxes.append(("text", org[0], org[1] - th, org[0] + tw,
                       org[1] + th * 0.35, str(text)))
        return real["putText"](img, text, org, font, scale, color, thick, line)

    def rect(img, p0, p1, color, thick=1, line=cv2.LINE_AA):
        _boxes.append(("rect", min(p0[0], p1[0]), min(p0[1], p1[1]),
                       max(p0[0], p1[0]), max(p0[1], p1[1]), ""))
        return real["rectangle"](img, p0, p1, color, thick, line)

    def circle(img, c, r, color, thick=1, line=cv2.LINE_AA):
        _boxes.append(("circle", c[0] - r, c[1] - r, c[0] + r, c[1] + r, ""))
        return real["circle"](img, c, r, color, thick, line)

    def ellipse(img, c, ax, angle, s, e, color, thick=1, line=cv2.LINE_AA):
        rx, ry = max(ax), min(ax)
        _boxes.append(("ellipse", c[0] - rx, c[1] - ry, c[0] + rx, c[1] + ry, ""))
        return real["ellipse"](img, c, ax, angle, s, e, color, thick, line)

    cv2.putText, cv2.rectangle, cv2.circle, cv2.ellipse = (
        putText, rect, circle, ellipse)
    return real


def _restore(real):
    cv2.putText, cv2.rectangle, cv2.circle, cv2.ellipse = (
        real["putText"], real["rectangle"], real["circle"], real["ellipse"])


def _overlaps(a, b):
    return not (a[2] <= b[1] or b[2] <= a[1] or a[4] <= b[3] or b[4] <= a[3])


def _check(w, h, st, recent=(), flash_ms=None):
    """Return a list of layout failures for one frame size and state."""
    L = _layout(w, h)
    _boxes.clear()
    _draw_overlay(np.zeros((h, w, 3), np.uint8), st, LABELS, 27.5, 0.62,
                  recent, flash_ms)
    tag = f"{st['_name']} {w}x{h}"
    bad = []
    dwell = float(st.get("dwell", 0.0))

    # 1. nothing drawn outside the frame
    for kind, x0, y0, x1, y1, txt in _boxes:
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            bad.append(f"{tag}: {kind} {txt!r} outside frame "
                       f"({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")

    # 2. the transcript must not collide with the status row
    sy = h - L["u"] * 0.9
    status = [b for b in _boxes if b[0] == "text" and b[1] >= sy - 1]
    big = [b for b in _boxes if b[0] == "text" and b[4] <= sy - 1]
    for a in big:
        for b in status:
            if _overlaps(a, b):
                bad.append(f"{tag}: transcript {a[5]!r} overlaps status {b[5]!r}")

    # 3. the alphabet strip must not collide with the wordmark. Restricted to
    #    the top bar: single letters also appear in the recent strip and in
    #    the confirmation flash, which are allowed elsewhere.
    wm = [b for b in _boxes if b[5] == "SIGNLANG"]
    # Box layout is (kind, x0, y0, x1, y1, text): filter on y0 (index 2), the
    # one that says which band a glyph is in.
    letters = [b for b in _boxes if b[0] == "text" and len(b[5]) == 1
               and b[2] < L["top_h"] and b not in wm]
    for a in letters:
        for b in wm:
            if _overlaps(a, b):
                bad.append(f"{tag}: alphabet {a[5]!r} overlaps wordmark")

    # 4. the reading card and the bottom panel never overlap
    if L["card_box"][3] > L["bot"][1]:
        bad.append(f"{tag}: card bottom {L['card_box'][3]} into panel "
                   f"top {L['bot'][1]}")

    # 5. the dwell ring stays inside the card. The confirmation flash also
    #    draws a ring, deliberately centred on the frame, so only the rings
    #    that overlap the card region at all are held to this.
    cb = L["card_box"]
    rings = [b for b in _boxes if b[0] in ("circle", "ellipse")
             and not (b[3] <= cb[0] or b[1] >= cb[2]
                       or b[4] <= cb[1] or b[2] >= cb[3])]
    for r in rings:
        if r[1] < cb[0] or r[2] < cb[1] or r[3] > cb[2] or r[4] > cb[3]:
            bad.append(f"{tag}: dwell ring leaves the card")

    # 6. something was actually drawn
    if len(_boxes) < 8:
        bad.append(f"{tag}: only {len(_boxes)} draw calls")

    # 7. every glyph must sit on a panel. Text painted straight onto the camera
    #    image inherits whatever contrast the room happens to have, which is
    #    how secondary labels end up invisible.
    panels = [L["top"], L["card_box"], L["bot"]]
    if L["alt_box"] is not None:
        panels.append(L["alt_box"])
    for kind, x0, y0, x1, y1, txt in _boxes:
        if kind != "text" or not txt.strip():
            continue
        # The confirmation flash is a deliberate exception: it is drawn over
        # the live camera image, large and centred, so it cannot be expected
        # to sit on a panel. Identify it by position rather than by text, so
        # a real unbacked label cannot claim the exemption.
        if (flash_ms is not None and recent and txt == recent[0]
                and abs((x0 + x1) / 2 - w / 2) < L["u"]
                and abs((y0 + y1) / 2 - h / 2) < L["u"]):
            continue
        if not any(p[0] <= x0 and p[1] <= y0 and p[2] >= x1 and p[3] >= y1
                   for p in panels):
            bad.append(f"{tag}: text {txt!r} not backed by a panel")

    # 8. the runners-up panel must not ride over the reading card
    if L["alt_box"] is not None:
        ab, cb = L["alt_box"], L["card_box"]
        if ab[0] < cb[2] and ab[3] > cb[1]:
            bad.append(f"{tag}: alt panel overlaps card")

    # 9. no more alternatives than there are rows of room for
    if len(st["top"]) - 1 > ALT_ROWS and st["_rows"] > ALT_ROWS:
        bad.append(f"{tag}: drew {st['_rows']} rows for ALT_ROWS={ALT_ROWS}")

    # 10. the countdown must be inside the card, and it must count down to zero
    #     rather than up from zero: a bar that fills while the label climbs is
    #     ambiguous about how long is left.
    if dwell > 0.001:
        want = f"{DWELL_MS / 1000.0 * (1.0 - dwell):.1f}s"
        found = [b for b in _boxes if b[0] == "text" and b[5] == want]
        if not found:
            bad.append(f"{tag}: no {want!r} countdown while holding")
        for b in found:
            if b[1] < cb[0] or b[2] < cb[1] or b[3] > cb[2] or b[4] > cb[3]:
                bad.append(f"{tag}: countdown {want!r} outside the card")

    # 11. the dwell bar is a full-width track plus a fill that starts at the
    #     same left edge and is shorter by the fraction still to go. Matching
    #     on the track alone is what let a missing fill slip through, and
    #     requiring the fill to be wide enough would fail near the start of a
    #     dwell, where it is legitimately a sliver.
    def _in_card(b):
        return (b[0] == "rect" and b[4] - b[2] <= L["u"] * 0.8
                and cb[0] <= b[1] and cb[1] <= b[2]
                and b[3] <= cb[2] and b[4] <= cb[3])

    bars = [b for b in _boxes if _in_card(b) and b[3] - b[1] > L["u"] * 4]
    fills = [b for b in _boxes
             if _in_card(b) and b[1] == bars[0][1] and b[4] == bars[0][4]
             and b[3] < bars[0][3]] if bars else []
    if dwell > 0.001:
        if not bars:
            bad.append(f"{tag}: dwell bar track missing")
        elif not fills:
            bad.append(f"{tag}: dwell bar has no fill at {dwell:.0%}")
        else:
            got = (fills[0][3] - fills[0][1]) / (bars[0][3] - bars[0][1])
            if abs(got - dwell) > 0.06:
                bad.append(f"{tag}: dwell bar shows {got:.0%}, state says {dwell:.0%}")

    return bad


def _check_blend():
    """The panel blend must match its float32 formula on every pixel.

    Two things this guards: the cached-solid fast path agreeing with the
    original arithmetic, and strided boxes actually writing back. A box that
    does not start at (0, 0) is a strided numpy view, and blending a temporary
    without copying it back silently leaves the panel unpainted.
    """
    rng = np.random.default_rng(11)
    tint, alpha = (15, 18, 25), 0.66
    bad = []
    # (box, roi_shape): includes origin-anchored boxes (strided by the row
    # pitch), interior boxes, and a full-frame one.
    for box, sh in [((0, 0, 1280, 82), (82, 1280, 3)),
                    ((20, 20, 1300, 102), (82, 1280, 3)),
                    ((25, 107, 223, 305), (198, 198, 3)),
                    ((0, 605, 1280, 720), (115, 1280, 3)),
                    ((25, 107, 1320, 250), (143, 1295, 3)),
                    ((1, 1, 2, 2), (1, 1, 3)),
                    ((25, 100, 32, 140), (40, 7, 3))]:
        base = rng.integers(0, 256, sh, dtype=np.uint8).copy()
        img = np.zeros((box[3] + 5, box[2] + 5, 3), np.uint8)
        img[box[1]:box[1] + sh[0], box[0]:box[0] + sh[1]] = base
        view = img[box[1]:box[1] + sh[0], box[0]:box[0] + sh[1]]
        contiguous = view.flags["C_CONTIGUOUS"]
        _blend(img, box, tint, alpha)
        got = img[box[1]:box[1] + sh[0], box[0]:box[0] + sh[1]]
        t = np.array(tint, np.float32)
        ref = (alpha * t + (1.0 - alpha) * base.astype(np.float32)).astype(np.uint8)
        # cv2 rounds half away from zero, numpy truncates, so the two may
        # disagree by one step and nothing else.
        d = int(np.abs(got.astype(int) - ref.astype(int)).max()) if got.size else 0
        if d > 1:
            bad.append(f"blend box {box} (contiguous={contiguous}): "
                       f"max diff {d} > 1")

    # A uniform region must move: a blend that silently no-ops is the failure
    # this whole check exists to catch.
    flat = np.full((40, 60, 3), 200, np.uint8)
    img = np.zeros((45, 65, 3), np.uint8)
    img[5:45, 5:65] = flat
    _blend(img, (5, 5, 65, 45), tint, alpha)
    got = img[5:45, 5:65]
    if not np.all(got != 200):
        bad.append("blend left pixels untouched")
    return bad


def _check_letterbox():
    """Fullscreen must letterbox, never stretch, and never upscale.

    Stretching would squash a 4:3 capture on a 16:9 display, and upscaling
    costs milliseconds per frame while adding no detail.
    """
    bad = []
    rng = np.random.default_rng(3)

    for (w, h), (ww, wh) in [((1280, 720), (1920, 1080)),
                             ((640, 480), (1920, 1080)),
                             ((1280, 720), (1280, 720)),
                             ((640, 480), (1024, 1366)),
                             ((320, 240), (3840, 2160)),
                             ((640, 480), (0, 0))]:
        src = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

        out = _fit_to_window(src, ww, wh)
        if out.shape[:2] != (wh, ww):
            if ww or wh:      # a zero-size request is passed through on purpose
                bad.append(f"letterbox {w}x{h}->{ww}x{wh}: got "
                           f"{out.shape[1]}x{out.shape[0]}")
            continue
        if (w, h) == (ww, wh):
            continue
        nh, nw = out.shape[:2]
        # Locate the drawn region rather than trusting the arithmetic: a
        # stretch keeps the canvas aspect while making the content wrong.
        rows = np.where(out.reshape(nh, -1).max(axis=1) > 0)[0]
        cols = np.where(out.max(axis=(0, 2)) > 0)[0]
        if not len(rows) or not len(cols):
            bad.append(f"letterbox {w}x{h}->{ww}x{wh}: nothing drawn")
            continue
        got = (cols.max() - cols.min() + 1) / (rows.max() - rows.min() + 1)
        want = w / h
        if abs(got - want) > 0.03:
            bad.append(f"letterbox {w}x{h}->{ww}x{wh}: content aspect "
                       f"{got:.3f} != source {want:.3f} (stretched)")
        # Centred: equal black margin on both sides, and top/bottom.
        ml, mr = cols.min(), nw - 1 - cols.max()
        mt, mb = rows.min(), nh - 1 - rows.max()
        if abs(ml - mr) > 1 or abs(mt - mb) > 1:
            bad.append(f"letterbox {w}x{h}->{ww}x{wh}: off-centre "
                       f"(l{ml} r{mr} t{mt} b{mb})")
        # And it must never exceed the canvas it was given.
        if cols.max() - cols.min() + 1 > nw or rows.max() - rows.min() + 1 > nh:
            bad.append(f"letterbox {w}x{h}->{ww}x{wh}: content overflows canvas")
    return bad


def _check_fills_canvas():
    """The overlay must reach the edges of the composed canvas.

    Regression guard: composing at a stale or unlaid-out window size made the
    overlay land in a small top-left corner while every other check still
    passed, because they all composed at the frame size and never noticed the
    canvas was wrong.
    """
    bad = []
    st = dict(CASES["holding"], _name="fills")
    for win_w, win_h in [(1366, 768), (1920, 1080), (1024, 768), (1280, 720)]:
        cam = np.zeros((720, 1280, 3), np.uint8)
        shown = _fit_to_window(cam, win_w, win_h)
        _draw_overlay(shown, st, LABELS, 30.0, 0.62, list("KBMLARD"), 150.0)
        if (shown.shape[1], shown.shape[0]) != (win_w, win_h):
            bad.append(f"canvas {shown.shape[1]}x{shown.shape[0]} != {win_w}x{win_h}")
            continue
        rows = np.where(shown.reshape(win_h, -1).max(axis=1) > 0)[0]
        cols = np.where(shown.reshape(-1, win_w).max(axis=0) > 0)[0]
        # Full-width bars run to both edges, so ink must touch first and last.
        if cols.min() > 1 or cols.max() < win_w - 2:
            bad.append(f"{win_w}x{win_h}: overlay x span {cols.min()}..{cols.max()}"
                       " does not fill the canvas (drawn into a corner?)")
        if rows.min() > 1 or rows.max() < win_h - 2:
            bad.append(f"{win_w}x{win_h}: overlay y span {rows.min()}..{rows.max()}"
                       " does not fill the canvas (drawn into a corner?)")
    return bad


def _check_screen_size():
    """hyprctl output must resolve to the focused monitor's pixel size."""
    from signlang import live as L

    real = subprocess.run

    def fake(stdout):
        class R:
            pass
        r = R()
        r.stdout = stdout
        r.stderr = ""
        r.returncode = 0
        return lambda *a, **k: r

    cases = [
        ("1366x768@60.01600 at 0x0\n\tscale: 1\n\tfocused: yes\n", (1366, 768)),
        ("2560x1440@60 at 0x0\n\tscale: 2\n\tfocused: yes\n", (1280, 720)),
        ("1920x1080@60 at 0x0\n\tscale: 1.5\n\tfocused: yes\n", (1280, 720)),
        # Monitors are separated by a blank line; the unfocused one comes first.
        ("Monitor A:\n\t800x600@60 at 0x0\n\tscale: 1\n\tfocused: no\n\n"
         "Monitor B:\n\t3840x2160@60 at 1920x0\n\tscale: 2\n\tfocused: yes\n",
         (1920, 1080)),
    ]
    bad = []
    for block, want in cases:
        subprocess.run = fake(block)
        got = L._screen_size()
        if got != want:
            bad.append(f"_screen_size() on {_first_size(block)} -> {got}, want {want}")

    # Unreadable monitor size must degrade to the capture, never to a corner.
    subprocess.run = fake("Monitor X:\n\tno size here\n\tfocused: yes\n")
    if L._screen_size() is not None:
        bad.append("_screen_size() returned a size for unparseable output")

    def boom(*a, **k):
        raise FileNotFoundError()

    subprocess.run = boom
    if L._screen_size() is not None:
        bad.append("_screen_size() should be None when hyprctl is absent")
    fs = L._fullscreen
    try:
        L._fullscreen = True
        if L._target_size(1280, 720) != (1280, 720):
            bad.append("fullscreen should fall back to the capture size")
        L._fullscreen = False
        if L._target_size(1280, 720) != (1280, 720):
            bad.append("windowed should always use the capture size")
    finally:
        L._fullscreen = fs
        subprocess.run = real
    return bad


def _first_size(block):
    m = re.search(r"(\d+x\d+@)", block)
    return m.group(1) if m else block.splitlines()[0].strip()


def main():
    print("live overlay layout checks")
    print(f"  {len(SIZES)} frame sizes x {len(CASES) + len(EXTRA)} states\n")
    real = _install_probes()
    failures = []
    try:
        failures += _check_blend()
        print("  ok   panel blend matches the float32 formula")
        failures += _check_letterbox()
        failures += _check_fills_canvas()
        print("  ok   overlay fills the whole canvas")
        failures += _check_screen_size()
        print("  ok   display size resolves to the focused monitor")
        print("  ok   fullscreen letterboxes without stretching\n")
        for w, h in SIZES:
            for name, base in CASES.items():
                st = dict(base, _name=name)
                st["_rows"] = min(max(len(base["top"]) - 1, 0), ALT_ROWS)
                failures += _check(w, h, st)
            for name, patch, recent, flash in EXTRA:
                st = dict(CASES["holding"], **patch)
                st["_name"] = name
                st["_rows"] = min(max(len(st["top"]) - 1, 0), ALT_ROWS)
                failures += _check(w, h, st, recent, flash)
            n = len(CASES) + len(EXTRA)
            print(f"  ok   {w}x{h} across {n} states")
    finally:
        _restore(real)

    print()
    if failures:
        print("FAILURES:")
        for f in failures[:40]:
            print(f"  - {f}")
        print(f"\n{len(failures)} failure(s)")
        return 1
    print(f"ALL CHECKS PASSED "
          f"({len(SIZES) * (len(CASES) + len(EXTRA))} render cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())