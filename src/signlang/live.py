"""Live fingerspelling view: camera, hand skeleton, and what it reads.

The overlay is drawn straight onto the camera frame with OpenCV, so it works
on the exhibition laptop with no browser and no network. It reads the same
letters the model knows, uses the same dwell-to-confirm rule as the game, and
shows the transcript as you sign.

Run: signlang live [--dwell=1000] [--sensitivity=0.8]

hands.py and engine.py are used exactly as they are; this file only decides
what the picture looks like.
"""

import functools
import re
import subprocess
import time

import cv2
import numpy as np

from .config import DWELL_MS
from .engine import DwellRecognizer
from .features import normalize_hand
from .hands import HandPipeline, draw_landmarks
from .model import Predictor
from .tts import get_tts

WINDOW = "signlang live"

# Set by main() so _target_size() can report the layout size it will compose.
# Kept at module scope because _screen_size() is also useful on its own, and a
# function-local flag would force the layout helpers to take it as an argument.
_fullscreen = True

# ---- overlay style ----------------------------------------------------
# OpenCV draws in BGR, so colours are converted from the same hex values the
# web game uses. One palette across both front ends so they read as the same
# tool rather than two unrelated programs.

def _bgr(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# Refined color palette with better depth and visual hierarchy
C_PANEL     = _bgr("#080b10")   # Deeper midnight for better contrast
C_PANEL_HI  = _bgr("#0f1419")   # Slightly lifted panel variant
C_BORDER    = _bgr("#1a1f2e")   # Refined slate border
C_BORDER_HI = _bgr("#2d3748")   # Highlighted border for active elements
C_BORDER_GLOW = _bgr("#3b82f6") # Subtle blue glow accent
C_TRACK     = _bgr("#151a24")   # Darker track for progress elements
C_INK       = _bgr("#f1f5f9")   # Primary text - slightly softer white
C_BRIGHT    = _bgr("#ffffff")   # Pure white for emphasis
C_DIM       = _bgr("#8b9cb3")   # Secondary text - better readability
C_MUTED     = _bgr("#5a6a80")   # Tertiary text
C_GOOD      = _bgr("#22c55e")   # Emerald green - success/confirmed
C_GOOD_DIM  = _bgr("#16a34a")   # Dimmed emerald for tracks
C_WARN      = _bgr("#eab308")   # Amber - pending/caution
C_WARN_DIM  = _bgr("#ca8a04")   # Dimmed amber
C_ACCENT    = _bgr("#0ea5e9")   # Sky blue - interactive/live
C_ACCENT_HI = _bgr("#38bdf8")   # Bright sky for highlights
C_GOLD      = _bgr("#fbbf24")   # Warm gold - active selection
C_GOLD_DIM  = _bgr("#d97706")   # Dimmed gold
C_ERROR     = _bgr("#ef4444")   # Red - error/low confidence

F_BIG = cv2.FONT_HERSHEY_DUPLEX
F_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# OpenCV's fontScale is not a pixel size: getTextSize reports a capital letter
# about 22*scale pixels tall. Sizes are therefore written as pixel heights and
# converted, so the layout below reads in the same units as the boxes. A floor
# keeps text legible on a very small capture instead of shrinking to nothing.
_PX_PER_SCALE = 22.0


def _scale(px_height):
    return max(0.30, px_height / _PX_PER_SCALE)


def _thick(px_height):
    return 2 if px_height >= 30 else 1


def _fit(text, font, scale, thick, max_w):
    """Longest prefix of `text` that fits `max_w`, marked with an ellipsis.

    The help line is the one string that can outgrow a narrow capture, since it
    sits between the gesture chip and the frame counter.
    """
    if max_w <= 0:
        return ""
    if cv2.getTextSize(text, font, scale, thick)[0][0] <= max_w:
        return text
    for n in range(len(text) - 1, 0, -1):
        cut = text[:n].rstrip() + "..."
        if cv2.getTextSize(cut, font, scale, thick)[0][0] <= max_w:
            return cut
    return ""


# Text heights, in the same `u` units the layout uses.
T_WORDMARK = 0.90
T_SUB = 0.55
T_CELL = 0.62
T_LETTER = 3.40
T_PCT = 0.70
T_COUNT = 1.60
T_NOTE = 0.50
T_ALT = 0.70
T_ALTP = 0.55
T_TEXT = 1.15
T_STATUS = 0.50
T_RECENT = 1.30

# Runners-up: how many alternatives, and their row step, in `u` units.
ALT_ROWS = 4
ALT_DY = 1.9
ALT_TOP = 0.20

# Confirmation flash: how long the big letter stays up, and the ring that
# expands with it. Short enough to feel instant, long enough to be seen in
# peripheral vision while the user is watching their hand.
FLASH_MS = 420
T_FLASH = 5.2

# Recent letters strip in the bottom panel.
RECENT_MAX = 10

# Keys shown in the status row. Kept short so it never collides with the
# readout on a narrow camera.
HELP = "F fullscreen · Q quit · C clear · R read · swipe = space"


def _wrap(text, max_px, font, scale, thick=1):
    """Break `text` into lines that each fit `max_px` once drawn in `font`.

    Measured with getTextSize rather than a character count, because the font
    scales with the frame. Fingerspelled text often has no spaces at all
    ("HELLOWORLD"), so a word longer than the line is split too rather than
    being allowed to run off the edge.
    """
    def width(s):
        return cv2.getTextSize(s, font, scale, thick)[0][0]

    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}".strip()
        if cur and width(trial) > max_px:
            lines.append(cur)
            cur = word
        else:
            cur = trial
        while width(cur) > max_px and len(cur) > 1:
            cut = len(cur) - 1
            while cut > 1 and width(cur[:cut]) > max_px:
                cut -= 1
            lines.append(cur[:cut])
            cur = cur[cut:]
    if cur:
        lines.append(cur)
    return lines or [""]


# One solid image per (shape, colour) so the blend is a cv2 call instead of
# float maths over every panel pixel. The result is bit-identical to the
# float32 blend but ~22x faster, which is the difference between 17fps and a
# smooth preview on a 720p camera. The cache is keyed on shape, so a resized
# window just misses a few frames until the new entries land.
_solids = {}


def _solid(shape, color):
    key = (shape, tuple(color))
    img = _solids.get(key)
    if img is None:
        if len(_solids) > 32:          # bound the cache; resizes cycle shapes
            _solids.clear()
        img = np.empty(shape, np.uint8)
        img[:] = color
        _solids[key] = img
    return img


def _blend(img, box, color, alpha=0.70):
    """Fill a box with a translucent colour, touching only that box.

    This runs on every camera frame, so it blends the panel's own pixels
    rather than compositing the whole picture, which would cost far more
    than the translucent panel is worth.
    """
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    # A box that does not start at (0, 0) is a strided view. cv2.addWeighted
    # writes straight through a strided dst (verified: the blend lands in the
    # frame and neighbouring pixels are untouched), so the copy-in/copy-back
    # that used to guard this is dead weight on the two biggest panels.
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1.0 - alpha, _solid(roi.shape, color), alpha, 0,
                    dst=roi)


@functools.lru_cache(maxsize=16)
def _layout(w, h):
    """Pixel geometry for one frame size.

    Every measurement comes from `u`, which comes from the short side, so the
    same numbers hold for a 320x240 thumbnail and a 1280x720 window. The old
    layout used fixed pixels and ran its dwell bar off the right edge of any
    camera narrower than 500px.
    """
    u = max(2.0, min(w, h) / 40.0)
    pad = int(u * 1.4)
    top_h = int(u * 4.6)
    # Taller than it needs to be for the transcript alone: the recent-letters
    # strip sits above it, and two transcript lines plus a status row.
    bot_h = int(u * 9.2)
    # Wide enough to hold the letter, the countdown underneath it and the
    # progress bar, without the countdown crowding the ring above.
    card = int(u * 11.8)

    top = (0, 0, w, top_h)
    card_box = (pad, top_h + pad, pad + card, top_h + pad + card)
    bot = (0, h - bot_h, w, h)

    # Runners-up sit to the right of the ring. They get their own panel: the
    # labels are secondary grey, and grey text on an arbitrary camera
    # background has no guaranteed contrast. All of it is resolved here so the
    # panel and the rows can never disagree about where they are.
    ax = card_box[2] + pad
    pct_right = w - pad
    tx0 = ax + int(u * 2.2)
    tx1 = pct_right - int(u * 3.2)
    ay = card_box[1] + int(card * ALT_TOP)
    alt_box = None
    if tx1 - tx0 > 4:
        alt_box = (card_box[2], ay - int(u * 1.8), w,
                   ay + int(ALT_ROWS * u * ALT_DY))

    return {
        "u": u,
        "pad": pad,
        "top_h": top_h,
        "bot_h": bot_h,
        "card": card,
        "top": top,
        "card_box": card_box,
        "bot": bot,
        "ax": ax,
        "tx0": tx0,
        "tx1": tx1,
        "ay": ay,
        "pct_right": pct_right,
        "alt_box": alt_box,
    }


def _draw_overlay(img, st, labels, fps, threshold, recent=(), flash_ms=None,
                  dwell_ms=DWELL_MS):
    """Paint the whole overlay onto the camera frame, in place.

    `recent` is the newest-first list of confirmed letters, and `flash_ms` is
    how long ago the last one was confirmed (None when nothing just landed).
    Both are passed in rather than tracked here, so the caller owns the state
    and this stays a pure function of the frame.
    """
    h, w = img.shape[:2]
    L = _layout(w, h)
    u, pad = L["u"], L["pad"]

    candidate = st.get("candidate")
    conf = float(st.get("confidence", 0.0))
    dwell = float(st.get("dwell", 0.0))
    gesture = st.get("gesture")
    buffer = st.get("buffer", "")
    top = st.get("top") or []
    confirmed = candidate is not None and conf >= threshold

    # ---- top bar: name on the left, the alphabet on the right ----
    _blend(img, L["top"], C_PANEL, 0.75)
    # Gradient-style bottom border with double line for depth
    cv2.line(img, (0, L["top_h"] - 1), (w, L["top_h"] - 1), C_BORDER_HI, 1, cv2.LINE_AA)
    cv2.line(img, (0, L["top_h"]), (w, L["top_h"]), C_BORDER, 1, cv2.LINE_AA)

    brand_y = int(u * 3.0)
    ws = _scale(u * T_WORDMARK)
    cv2.putText(img, "SIGNLANG", (pad, brand_y), F_BIG, ws,
                C_INK, _thick(u * T_WORDMARK), cv2.LINE_AA)
    (bw, _), _ = cv2.getTextSize("SIGNLANG", F_BIG, ws, _thick(u * T_WORDMARK))

    # Studio broadcast pill badge: refined container with pulsing dot + label
    live_x = pad + bw + int(u * 0.9)
    pill_h = int(u * 1.4)
    pill_y = brand_y - int(u * 1.05)
    pill_w = int(u * 4.4)
    if pill_y > 0 and pill_y + pill_h < L["top_h"]:
        # Outer pill with subtle gradient effect
        cv2.rectangle(img, (live_x, pill_y), (live_x + pill_w, pill_y + pill_h),
                      C_TRACK, -1)
        # Inner highlight for 3D depth
        cv2.rectangle(img, (live_x, pill_y), (live_x + pill_w, pill_y + 1),
                      C_BORDER_HI, 1, cv2.LINE_AA)
        cv2.rectangle(img, (live_x, pill_y), (live_x + pill_w, pill_y + pill_h),
                      C_BORDER_HI, 1, cv2.LINE_AA)
        # Pulsing status dot with layered glow
        dot_r = max(2, int(u * 0.18))
        dot_cx = live_x + int(u * 0.75)
        dot_cy = pill_y + pill_h // 2
        # Outer soft glow
        cv2.circle(img, (dot_cx, dot_cy), dot_r + 3, C_GOOD_DIM, 1, cv2.LINE_AA)
        # Middle glow ring
        cv2.circle(img, (dot_cx, dot_cy), dot_r + 1, C_GOOD, 1, cv2.LINE_AA)
        # Core bright dot
        cv2.circle(img, (dot_cx, dot_cy), dot_r, C_BRIGHT, -1, cv2.LINE_AA)
        ls = _scale(u * 0.52)
        lt = _thick(u * 0.52)
        (lw, lh), _ = cv2.getTextSize("LIVE", F_SMALL, ls, lt)
        cv2.putText(img, "LIVE", (dot_cx + dot_r + int(u * 0.4), dot_cy + lh // 2),
                    F_SMALL, ls, C_INK, lt, cv2.LINE_AA)

    # The alphabet doubles as a lookup chart for a letter the player is not
    # sure of. It only appears when the cells are wide enough to read, and
    # only in the space left of the wordmark, so it never overlaps it.
    n = len(labels)
    avail = int(w - pad * 2 - u * 12)
    if n and avail > 0:
        cell = avail // n
        if cell >= u * 1.35:
            ch = int(u * 1.9)
            x0 = w - pad - cell * n
            y0 = int((L["top_h"] - ch) / 2)
            cs = _scale(u * T_CELL)
            ct = _thick(u * T_CELL)
            for i, lab in enumerate(labels):
                cx = x0 + cell * i
                on = lab == candidate
                if on:
                    cell_bg = C_GOOD if confirmed else C_GOLD
                    cell_border = C_GOOD_DIM if confirmed else C_WARN
                    cell_text = C_BRIGHT if confirmed else C_PANEL
                    cv2.rectangle(img, (cx, y0), (cx + cell - 1, y0 + ch),
                                  cell_bg, -1, cv2.LINE_AA)
                    cv2.rectangle(img, (cx, y0), (cx + cell - 1, y0 + ch),
                                  cell_border, 1, cv2.LINE_AA)
                    # Glowing pip above the active cell
                    pip_h = max(2, int(u * 0.18))
                    if y0 >= pip_h + 1:
                        pip_col = C_GOOD if confirmed else C_GOLD
                        cv2.rectangle(img, (cx + cell // 4, y0 - pip_h - 1),
                                      (cx + 3 * cell // 4, y0 - 1), pip_col, -1)
                else:
                    if i > 0:
                        cv2.line(img, (cx, y0 + int(ch * 0.25)), (cx, y0 + int(ch * 0.75)),
                                 C_BORDER, 1)
                    cell_text = C_DIM
                lab_text = "_" if lab == "SPACE" else lab
                (tw, _), _ = cv2.getTextSize(lab_text, F_SMALL, cs, ct)
                cv2.putText(img, lab_text, (cx + (cell - tw) // 2, y0 + int(ch * 0.72)),
                            F_SMALL, cs, cell_text,
                            ct, cv2.LINE_AA)

    # ---- reading card: the letter the camera is offering right now ----
    cb = L["card_box"]
    _blend(img, cb, C_PANEL, 0.78)
    # Refined glass border with double-line effect
    cv2.rectangle(img, (cb[0], cb[1]), (cb[2], cb[3]), C_BORDER, 1, cv2.LINE_AA)
    # Inner subtle highlight
    cv2.rectangle(img, (cb[0] + 1, cb[1] + 1), (cb[2] - 1, cb[3] - 1), C_BORDER_HI, 1, cv2.LINE_AA)

    # Corner brackets / reticles on the card for a refined HUD appearance
    cl = max(3, int(u * 0.85))
    # Color based on confidence state
    if confirmed:
        bracket_col = C_GOOD
        bracket_glow = C_GOOD_DIM
    elif candidate:
        bracket_col = C_WARN
        bracket_glow = C_WARN_DIM
    else:
        bracket_col = C_BORDER_HI
        bracket_glow = C_BORDER
    # Draw corner brackets with glow effect
    for corner in ["tl", "tr", "bl", "br"]:
        if corner == "tl":
            x, y = cb[0], cb[1]
            dx1, dy1 = cl, 0
            dx2, dy2 = 0, cl
        elif corner == "tr":
            x, y = cb[2] - 1, cb[1]
            dx1, dy1 = -cl, 0
            dx2, dy2 = 0, cl
        elif corner == "bl":
            x, y = cb[0], cb[3] - 1
            dx1, dy1 = cl, 0
            dx2, dy2 = 0, -cl
        else:  # br
            x, y = cb[2] - 1, cb[3] - 1
            dx1, dy1 = -cl, 0
            dx2, dy2 = 0, -cl
        # Outer glow
        cv2.line(img, (x + dx1, y), (x, y), bracket_glow, 3, cv2.LINE_AA)
        cv2.line(img, (x, y + dy2), (x, y), bracket_glow, 3, cv2.LINE_AA)
        # Inner bright line
        cv2.line(img, (x + dx1, y), (x, y), bracket_col, 2, cv2.LINE_AA)
        cv2.line(img, (x, y + dy2), (x, y), bracket_col, 2, cv2.LINE_AA)
    # Subtle top hairline across the card header for polished glass finish
    accent_bar = C_GOOD if confirmed else (C_ACCENT if candidate else C_BORDER)
    cv2.line(img, (cb[0] + cl, cb[1]), (cb[2] - 1 - cl, cb[1]), accent_bar, 1, cv2.LINE_AA)

    cx = cb[0] + L["card"] // 2
    cy = cb[1] + int(L["card"] * 0.42)
    r = int(L["card"] * 0.29)
    thick = max(2, int(u * 0.30))

    # Cardinal tick marks for a precision chronograph dial look
    tlen = max(2, int(u * 0.35))
    # Dynamic color based on state
    if confirmed:
        tcol = C_GOOD
    elif dwell > 0.001:
        tcol = C_ACCENT_HI
    else:
        tcol = C_BORDER_HI
    cv2.line(img, (cx, cy - r - 2), (cx, cy - r - 2 - tlen), tcol, 1, cv2.LINE_AA)
    cv2.line(img, (cx + r + 2, cy), (cx + r + 2 + tlen, cy), tcol, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + r + 2), (cx, cy + r + 2 + tlen), tcol, 1, cv2.LINE_AA)
    cv2.line(img, (cx - r - 2, cy), (cx - r - 2 - tlen, cy), tcol, 1, cv2.LINE_AA)

    # Diagonal micro tick marks for instrument HUD aesthetic
    d = int(r * 0.7071)
    d1 = d + 2
    d2 = d + 2 + max(1, tlen // 2)
    cv2.line(img, (cx + d1, cy - d1), (cx + d2, cy - d2), C_BORDER_HI, 1, cv2.LINE_AA)
    cv2.line(img, (cx + d1, cy + d1), (cx + d2, cy + d2), C_BORDER_HI, 1, cv2.LINE_AA)
    cv2.line(img, (cx - d1, cy + d1), (cx - d2, cy + d2), C_BORDER_HI, 1, cv2.LINE_AA)
    cv2.line(img, (cx - d1, cy - d1), (cx - d2, cy - d2), C_BORDER_HI, 1, cv2.LINE_AA)

    # Ambient subtle guide circle
    cv2.circle(img, (cx, cy), r + max(2, int(u * 0.22)), C_BORDER, 1, cv2.LINE_AA)

    # The ring is the dwell clock: it only fills once the letter is stable and
    # confident enough to be counting towards a confirmation. Reading the fill
    # next to the letter is quicker than checking a separate bar.
    cv2.circle(img, (cx, cy), r, C_TRACK, thick, cv2.LINE_AA)
    if dwell > 0.001:
        # Gradient-like effect: draw the progress arc with the main color
        cv2.ellipse(img, (cx, cy), (r, r), 0, -90.0, -90.0 + 360.0 * dwell,
                    C_GOOD, thick, cv2.LINE_AA)
        # Leading progress beacon bead with glow effect
        ang = np.deg2rad(-90.0 + 360.0 * dwell)
        hx = int(round(cx + r * np.cos(ang)))
        hy = int(round(cy + r * np.sin(ang)))
        # Outer glow
        cv2.circle(img, (hx, hy), thick + 3, C_GOOD_DIM, 1, cv2.LINE_AA)
        # Middle ring
        cv2.circle(img, (hx, hy), thick + 1, C_GOOD, 1, cv2.LINE_AA)
        # Bright center
        cv2.circle(img, (hx, hy), thick - 1, C_BRIGHT, -1, cv2.LINE_AA)

    glyph = candidate or "-"
    # Refined color logic for the letter
    if confirmed:
        gcol = C_BRIGHT
    elif dwell > 0.001:
        gcol = C_GOOD
    elif candidate:
        gcol = C_WARN
    else:
        gcol = C_DIM
    scale_factor = 0.42 if len(glyph) > 1 else 1.0
    gscale, gthick = _scale(u * T_LETTER * scale_factor), _thick(u * T_LETTER * scale_factor)
    (tw, th), _ = cv2.getTextSize(glyph, F_BIG, gscale, gthick)
    cv2.putText(img, glyph, (cx - tw // 2, cy + th // 2), F_BIG, gscale,
                gcol, gthick, cv2.LINE_AA)

    base = cb[3]
    pct = f"{conf * 100:.0f}%" if candidate else "--"
    ps = _scale(u * T_PCT)
    (pw, _), _ = cv2.getTextSize(pct, F_SMALL, ps, _thick(u * T_PCT))
    # Color the percentage based on confidence level
    if confirmed:
        pct_col = C_GOOD
    elif conf >= 0.7:
        pct_col = C_ACCENT_HI
    elif candidate:
        pct_col = C_DIM
    else:
        pct_col = C_MUTED
    cv2.putText(img, pct, (cx - pw // 2, base - int(u * 3.3)), F_SMALL, ps,
                pct_col, _thick(u * T_PCT), cv2.LINE_AA)

    # Countdown. The ring shows progress but not speed: with a one second dwell
    # the user needs to know how much longer, in real units, or the wait feels
    # arbitrary. Empty when nothing is counting, so it does not sit there
    # claiming the next letter is on its way.
    if dwell > 0.001:
        secs = dwell_ms / 1000.0 * (1.0 - min(1.0, dwell))
        ctxt = f"{secs:.1f}s"
        cs, ct = _scale(u * T_COUNT), _thick(u * T_COUNT)
        (cw, _), _ = cv2.getTextSize(ctxt, F_BIG, cs, ct)
        cv2.putText(img, ctxt, (cx - cw // 2, base - int(u * 1.4)), F_BIG, cs,
                    C_GOOD, ct, cv2.LINE_AA)
    elif confirmed:
        ctxt = "ready"
        cs, ct = _scale(u * T_COUNT), _thick(u * T_COUNT)
        (cw, _), _ = cv2.getTextSize(ctxt, F_BIG, cs, ct)
        cv2.putText(img, ctxt, (cx - cw // 2, base - int(u * 1.4)), F_BIG, cs,
                    C_DIM, ct, cv2.LINE_AA)

    # Progress bar across the foot of the card. Same information as the ring,
    # but linear and unmissable from further away than the ring is.
    bx0 = cb[0] + int(u * 1.2)
    bx1 = cb[2] - int(u * 1.2)
    by = base - int(u * 0.55)
    bh = max(3, int(u * 0.48))
    # Track with subtle inner shadow
    cv2.rectangle(img, (bx0, by), (bx1, by + bh), C_TRACK, -1)
    cv2.line(img, (bx0, by), (bx1, by), C_BORDER_HI, 1, cv2.LINE_AA)
    if dwell > 0.001:
        # Progress fill with gradient effect
        fill_width = int((bx1 - bx0) * min(1.0, dwell))
        if fill_width > 0:
            cv2.rectangle(img, (bx0, by), (bx0 + fill_width, by + bh), C_GOOD, -1)
            # Bright leading edge
            cv2.line(img, (bx0 + fill_width - 1, by), (bx0 + fill_width - 1, by + bh),
                     C_BRIGHT, 1, cv2.LINE_AA)

    if candidate is None:
        note, ncol = "show your hand", C_DIM
    elif dwell > 0.001:
        note, ncol = "holding", C_GOOD
    elif confirmed:
        note, ncol = "hold still", C_GOOD
    else:
        note, ncol = "not sure yet", C_WARN
    ns, nt = _scale(u * T_NOTE), _thick(u * T_NOTE)
    note = _fit(note, F_SMALL, ns, nt, L["card"] - int(u * 0.6))
    if note:
        (nw, _), _ = cv2.getTextSize(note, F_SMALL, ns, nt)
        cv2.putText(img, note, (cx - nw // 2, base - int(u * 5.2)), F_SMALL,
                    ns, ncol, nt, cv2.LINE_AA)

    # ---- runners-up: what the model nearly thought it saw ----
    # A stalled dwell usually means two letters are close together. Showing
    # the alternatives turns "it just will not confirm" into a usable clue.
    if L["alt_box"] is not None:
        ab = L["alt_box"]
        _blend(img, ab, C_PANEL, 0.76)
        cv2.rectangle(img, (ab[0], ab[1]), (ab[2], ab[3]), C_BORDER, 1, cv2.LINE_AA)
        cv2.rectangle(img, (ab[0] + 1, ab[1] + 1), (ab[2] - 1, ab[3] - 1), C_BORDER_HI, 1, cv2.LINE_AA)

        ax, tx0, tx1 = L["ax"], L["tx0"], L["tx1"]
        pct_right = L["pct_right"]
        ascl = _scale(u * T_ALT)
        aps = _scale(u * T_ALTP)
        y = L["ay"]

        for i, item in enumerate(top[1:ALT_ROWS + 1]):
            lab = item["label"]
            prob = float(item["prob"])
            cv2.putText(img, lab, (ax, int(y)), F_SMALL, ascl,
                        C_BRIGHT if i == 0 else C_DIM, _thick(u * T_ALT), cv2.LINE_AA)
            y0, y1 = int(y - u * 0.85), int(y - u * 0.45)
            cv2.rectangle(img, (tx0, y0), (tx1, y1), C_TRACK, -1)
            bar_fill = C_ACCENT_HI if i == 0 else C_BORDER_HI
            bw = tx0 + max(2, int((tx1 - tx0) * prob))
            cv2.rectangle(img, (tx0, y0), (bw, y1), bar_fill, -1)
            if i == 0 and bw > tx0 + 2:
                cv2.line(img, (bw - 1, y0), (bw - 1, y1), C_BRIGHT, 1, cv2.LINE_AA)
            txt = f"{prob * 100:.0f}"
            (tw2, _), _ = cv2.getTextSize(txt, F_SMALL, aps, _thick(u * T_ALTP))
            cv2.putText(img, txt, (pct_right - tw2, int(y)),
                        F_SMALL, aps, C_INK if i == 0 else C_DIM, _thick(u * T_ALTP), cv2.LINE_AA)
            if i < len(top[1:ALT_ROWS + 1]) - 1:
                div_y = int(y + u * 0.5)
                cv2.line(img, (ax, div_y), (pct_right, div_y), C_BORDER, 1)
            y += u * ALT_DY

    # ---- confirmation flash ----
    # A dwell that quietly completes is easy to miss, especially while the
    # user is watching their hand. This draws the letter that just landed
    # large and central, then hands the eye straight back to the card.
    if flash_ms is not None and recent:
        t = min(1.0, max(0.0, float(flash_ms) / FLASH_MS))
        fade = (1.0 - t) ** 2
        if fade > 0.02:
            fs = _scale(u * T_FLASH)
            ft = _thick(u * T_FLASH)
            (fw2, fh2), _ = cv2.getTextSize(recent[0], F_BIG, fs, ft)
            fx, fy = w // 2, h // 2
            col = tuple(int(c * fade + p * (1 - fade))
                        for c, p in zip(C_GOOD, C_INK))
            cv2.putText(img, recent[0], (fx - fw2 // 2, fy + fh2 // 2),
                        F_BIG, fs, col, ft, cv2.LINE_AA)
            # Expanding primary ring pulse
            rad = int(u * (5.0 + 5.0 * t))
            ring = tuple(int(c * fade) for c in C_GOOD)
            cv2.circle(img, (fx, fy), rad, ring, max(1, int(u * 0.28)),
                       cv2.LINE_AA)
            # Inner secondary ripple ring
            rad_inner = int(u * (3.0 + 3.0 * t))
            ring_inner = tuple(int(c * fade * 0.6) for c in C_ACCENT)
            cv2.circle(img, (fx, fy), rad_inner, ring_inner, 1, cv2.LINE_AA)

            # 4 HUD corner brackets expanding with confirmation
            b_sz = max(fw2, fh2) // 2 + int(u * (0.8 + 1.0 * t))
            b_len = max(3, int(u * 0.6))
            b_col = tuple(int(c * fade * 0.75) for c in C_GOOD)
            cv2.line(img, (fx - b_sz, fy - b_sz), (fx - b_sz + b_len, fy - b_sz), b_col, 1, cv2.LINE_AA)
            cv2.line(img, (fx - b_sz, fy - b_sz), (fx - b_sz, fy - b_sz + b_len), b_col, 1, cv2.LINE_AA)
            cv2.line(img, (fx + b_sz, fy - b_sz), (fx + b_sz - b_len, fy - b_sz), b_col, 1, cv2.LINE_AA)
            cv2.line(img, (fx + b_sz, fy - b_sz), (fx + b_sz, fy - b_sz + b_len), b_col, 1, cv2.LINE_AA)
            cv2.line(img, (fx - b_sz, fy + b_sz), (fx - b_sz + b_len, fy + b_sz), b_col, 1, cv2.LINE_AA)
            cv2.line(img, (fx - b_sz, fy + b_sz), (fx - b_sz, fy - b_sz + b_len), b_col, 1, cv2.LINE_AA)
            cv2.line(img, (fx + b_sz, fy + b_sz), (fx + b_sz - b_len, fy + b_sz), b_col, 1, cv2.LINE_AA)
            cv2.line(img, (fx + b_sz, fy + b_sz), (fx + b_sz, fy + b_sz - b_len), b_col, 1, cv2.LINE_AA)

    # ---- bottom: recent letters, then transcript, then the status row ----
    _blend(img, L["bot"], C_PANEL, 0.80)
    bot_top = L["bot"][1]
    # Double-line border for depth
    cv2.line(img, (0, bot_top), (w, bot_top), C_BORDER, 1, cv2.LINE_AA)
    cv2.line(img, (0, bot_top + 1), (w, bot_top + 1), C_BORDER_HI, 1, cv2.LINE_AA)
    # Accent highlight on the left
    cv2.line(img, (pad, bot_top), (pad + int(u * 6.0), bot_top), C_ACCENT_HI, 2, cv2.LINE_AA)

    # The recent strip answers "did that go in?" without making the user read
    # the transcript. Newest letter is the bright one on the left.
    ry = bot_top + int(u * 2.0)
    if recent:
        rs = _scale(u * T_RECENT)
        rt = _thick(u * T_RECENT)
        step = int(u * 2.3)
        limit = min(len(recent), RECENT_MAX)
        for i, lab in enumerate(recent[:limit]):
            newest = i == 0
            (rw, _), _ = cv2.getTextSize(lab, F_SMALL, rs, rt)
            x = pad + i * step
            if i > 0:
                # Subtle separator dot
                dot_x = pad + i * step - int(step * 0.45)
                cv2.circle(img, (dot_x, ry - int(u * 0.35)), max(1, int(u * 0.12)), C_BORDER_HI, -1, cv2.LINE_AA)
            if newest:
                # Badge background for newest letter
                badge_x0 = x - int(u * 0.35)
                badge_x1 = x + rw + int(u * 0.35)
                cv2.rectangle(img, (badge_x0, ry - int(u * 1.05)), (badge_x1, ry + int(u * 0.25)), C_TRACK, -1)
                cv2.rectangle(img, (badge_x0, ry - int(u * 1.05)), (badge_x1, ry + int(u * 0.25)), C_BORDER_HI, 1, cv2.LINE_AA)
                # Underline accent
                cv2.rectangle(img, (x - 1, ry + int(u * 0.35)),
                              (x + rw + 1, ry + int(u * 0.70)), C_GOOD, -1)
            cv2.putText(img, lab, (x, ry), F_SMALL, rs, C_BRIGHT if newest else C_DIM, rt, cv2.LINE_AA)

    ty = bot_top + int(u * 5.0)
    if buffer:
        bs, bt = _scale(u * T_TEXT), _thick(u * T_TEXT)
        lines = _wrap(buffer, w - pad * 2 - int(u * 1.5), F_BIG, bs, bt)[-2:]
        for idx, ln in enumerate(lines):
            is_last = (idx == len(lines) - 1)
            cv2.putText(img, ln, (pad, ty), F_BIG, bs, C_INK, bt, cv2.LINE_AA)
            if is_last:
                (lw, _), _ = cv2.getTextSize(ln, F_BIG, bs, bt)
                cur_x = pad + lw + int(u * 0.3)
                cur_y0 = ty - int(u * 0.85)
                cur_y1 = ty + int(u * 0.15)
                # Cursor with glow effect
                cv2.rectangle(img, (cur_x - 1, cur_y0), (cur_x + max(3, int(u * 0.40)), cur_y1), C_ACCENT_HI, -1)
                cv2.rectangle(img, (cur_x - 1, cur_y0), (cur_x + max(3, int(u * 0.40)), cur_y1), C_BRIGHT, 1, cv2.LINE_AA)
            ty += int(u * 1.95)
    else:
        es = _scale(u * T_STATUS)
        # More inviting empty state
        cv2.putText(img, "hold a letter still to start", (pad, ty), F_SMALL,
                    es, C_MUTED, _thick(u * T_STATUS), cv2.LINE_AA)

    sy = h - int(u * 0.9)
    # Hairline divider line above the status control row
    cv2.line(img, (pad, sy - int(u * 0.95)), (w - pad, sy - int(u * 0.95)), C_BORDER, 1, cv2.LINE_AA)

    sx = pad
    if gesture:
        gtxt = gesture.upper()
        gs = _scale(u * T_SUB)
        (gw, _), _ = cv2.getTextSize(gtxt, F_SMALL, gs, _thick(u * T_SUB))
        chip = gw + int(u * 1.4)
        # Gesture chip with enhanced styling
        cv2.rectangle(img, (sx, sy - int(u * 1.25)), (sx + chip, sy + int(u * 0.25)),
                      C_GOOD, -1)
        cv2.rectangle(img, (sx, sy - int(u * 1.25)), (sx + chip, sy + int(u * 0.25)),
                      C_BRIGHT, 1, cv2.LINE_AA)
        cv2.putText(img, gtxt, (sx + int(u * 0.7), sy), F_SMALL, gs,
                    C_PANEL, _thick(u * T_SUB), cv2.LINE_AA)
        sx += chip + int(u * 1.0)
    hs, ht = _scale(u * T_STATUS), _thick(u * T_STATUS)
    fps_txt = f"{fps:4.1f} fps"
    (fw, _), _ = cv2.getTextSize(fps_txt, F_SMALL, hs, ht)
    fps_x = w - pad - fw
    help_txt = _fit(HELP, F_SMALL, hs, ht, (fps_x - int(u * 0.9)) - sx)
    if help_txt:
        cv2.putText(img, help_txt, (sx, sy), F_SMALL, hs, C_DIM, ht, cv2.LINE_AA)

    # Performance monitor dot + text with refined colors
    if fps >= 20.0:
        fps_dot_col = C_GOOD
    elif fps >= 10.0:
        fps_dot_col = C_WARN
    else:
        fps_dot_col = C_ERROR
    # FPS indicator with glow
    cv2.circle(img, (fps_x - int(u * 0.45), sy - int(u * 0.22)), max(2, int(u * 0.16)),
               fps_dot_col, -1, cv2.LINE_AA)
    cv2.circle(img, (fps_x - int(u * 0.45), sy - int(u * 0.22)), max(3, int(u * 0.22)),
               fps_dot_col, 1, cv2.LINE_AA)
    cv2.putText(img, fps_txt, (fps_x, sy), F_SMALL, hs,
                C_DIM, ht, cv2.LINE_AA)


def _fit_to_window(frame, win_w, win_h):
    """Centre `frame` inside win_w x win_h without distorting it.

    Scales the camera capture to fill the available screen area while
    preserving its native aspect ratio, letterboxing only when necessary to
    avoid distorting the video.
    """
    h, w = frame.shape[:2]
    if (w, h) == (win_w, win_h):
        return frame
    if win_w <= 0 or win_h <= 0:
        return frame
    scale = min(win_w / w, win_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if (nw, nh) != (w, h):
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (nw, nh), interpolation=interp)
    canvas = np.zeros((win_h, win_w, 3), np.uint8)
    x0, y0 = (win_w - nw) // 2, (win_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = frame
    return canvas


def _screen_size():
    """Pixel size of the focused monitor, or None when it cannot be read.

    `cv2.getWindowImageRect` is useless here: it reports the size of the image
    last drawn, not the size of the window, so a fullscreen window created
    empty measures 98x28 and the overlay ends up crammed into that corner.
    Asking the compositor for the monitor size avoids the whole loop.

    Scale matters as much as size: Hyprland reports 1366x768 at scale 1, but
    a scaled monitor is described in its own resolution, and the compositor
    then scales the window back down, so the value has to be divided by the
    scale to get the pixels OpenCV actually draws into.
    """
    try:
        out = subprocess.run(
            ["hyprctl", "monitors", "all"], capture_output=True, text=True,
            timeout=2.0, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    # Blocks are separated by blank lines; the focused monitor is the one
    # marked `focused: yes`.
    for block in out.split("\n\n"):
        if "focused: yes" not in block:
            continue
        size = re.search(r"^\s*(\d+)x(\d+)@\s*[\d.]+", block, re.M)
        scale = re.search(r"^\s*scale:\s*([\d.]+)", block, re.M)
        if not size:
            continue
        w, h = int(size.group(1)), int(size.group(2))
        s = float(scale.group(1)) if scale else 1.0
        if s <= 0:
            s = 1.0
        return max(1, int(w / s)), max(1, int(h / s))
    return None


def _target_size(frame_w, frame_h):
    """Size to compose for: the monitor when fullscreen, else the capture.

    Falls back to the capture size whenever the monitor cannot be read, which
    keeps a windowed run working unchanged and makes fullscreen degrade to a
    normal window rather than to a squashed one.
    """
    if not _fullscreen:
        return frame_w, frame_h
    size = _screen_size()
    return size if size else (frame_w, frame_h)


def main(argv=None):
    argv = list(argv or [])
    pipe = None
    fullscreen = "--windowed" not in argv
    global _fullscreen
    _fullscreen = fullscreen
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
    print(f"  display: {'fullscreen' if fullscreen else 'windowed'}"
          "   (F toggles, Esc or Q quits)")
    print("=" * 58)
    print("  Hold a letter still to confirm it. Swipe LEFT to erase,")
    print("  swipe RIGHT for a space.  Q quits, C clears.")
    print("=" * 58)

    pipe = HandPipeline()
    pipe.started.wait(timeout=12)
    if pipe.error is not None:
        print(f"camera error: {pipe.error}")
        return 1

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if fullscreen:
            size = _screen_size()
            if size:
                cv2.resizeWindow(WINDOW, size[0], size[1])
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW, 1280, 720)
    except cv2.error as exc:
        # Some Wayland compositors refuse the fullscreen request. The window
        # still opens, so carry on rather than failing the whole run.
        print(f"  (window setup fell back to a normal window: {exc})")

    fps_t, fps_n, fps_v = time.time(), 0, 0.0
    recent = []
    last_emit, flash_t = None, 0.0
    cur_target, cur_fs = None, None
    tts = get_tts()
    if tts.enabled:
        print("  TTS: enabled (voice: lessac-medium)")
    else:
        print("  TTS: disabled (install piper-tts and sounddevice to enable)")
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
            norm, pts = None, None
            if landmarks is not None:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], np.float32)
                draw_landmarks(frame, landmarks)
                cx, cy = int(landmarks[0].x * w), int(landmarks[0].y * h)
                # Refined wrist anchor reticle
                cv2.circle(frame, (cx, cy), 3, C_ACCENT_HI, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 7, C_ACCENT, 1, cv2.LINE_AA)
                # Fingertip tracking halos on index 4, 8, 12, 16, 20
                for tip_idx in (4, 8, 12, 16, 20):
                    tx, ty = int(landmarks[tip_idx].x * w), int(landmarks[tip_idx].y * h)
                    cv2.circle(frame, (tx, ty), 4, C_GOOD, 1, cv2.LINE_AA)
                norm = normalize_hand(pts, handedness)

            st = rec.update(norm, handedness, w, h, pts)

            # Track confirmations here rather than in the recogniser, so
            # engine.py stays untouched. `emitted` repeats for 500ms, so only a
            # change of letter counts as a new confirmation.
            emitted = st.get("emitted")
            if emitted and emitted != last_emit:
                last_emit = emitted
                flash_t = time.time()
                recent.insert(0, "_" if emitted == " " else emitted)
                del recent[RECENT_MAX:]
                # Speak the confirmed letter/gesture
                if tts.enabled:
                    tts.say_letter(emitted)
            elif not emitted:
                last_emit = None

            now = time.time()
            flash_ms = (now - flash_t) * 1000.0
            if flash_ms > FLASH_MS:
                flash_ms = None

            # Cache the display resolution so we don't spawn a subprocess on
            # every single frame.
            if cur_target is None or cur_fs != fullscreen:
                cur_target = _target_size(w, h)
                cur_fs = fullscreen

            shown = _fit_to_window(frame, *cur_target)

            _draw_overlay(shown, st, predictor.labels, fps_v, rec.threshold,
                          recent, flash_ms, rec.dwell_ms)

            cv2.imshow(WINDOW, shown)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k in (ord("c"), ord("C")):
                rec.clear()
                recent.clear()
                last_emit = None
            elif k in (ord("r"), ord("R")):
                if tts.enabled and rec.buffer:
                    tts.read_transcript(rec.buffer)
            elif k == 8:
                rec.backspace()
            elif k in (ord("f"), ord("F")):
                # Toggle at runtime so a mis-typed flag needs no restart.
                fullscreen = not fullscreen
                _fullscreen = fullscreen
                cur_fs = None
                try:
                    if fullscreen:
                        size = _screen_size()
                        if size:
                            cv2.resizeWindow(WINDOW, size[0], size[1])
                        cv2.setWindowProperty(
                            WINDOW, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(
                            WINDOW, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(WINDOW, 1280, 720)
                except cv2.error:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        if pipe is not None:
            pipe.close()
        tts.close()
        cv2.destroyAllWindows()

    print(f"\n  transcript: {rec.buffer!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())