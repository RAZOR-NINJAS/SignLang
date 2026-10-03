"""Run state and phase timing for the shadow-play game.

Pure logic: no camera, no browser, no model. The server drives it with a
monotonic clock and feeds it whatever letter the detector last saw, which is
what lets the 60fps animation and the 10fps detector run independently.

Phases, in order:

    idle      waiting for the player to press Start
    ready     target shadow shown, NOT scored - the player may already be
              forming it, so nothing here can be failed
    answer    scored window; the player must hold the target shape
    correct   brief celebration, then back to travelling
    wrong     the run ended: the answer window closed without the target

The ready phase exists because a scored window that starts the instant the
letter appears punishes the player for reaction time rather than for forming
the wrong letter.
"""

import time

PHASES = ("idle", "ready", "answer", "correct", "wrong", "finished")

DEFAULT_READY_MS = 2200
DEFAULT_ANSWER_MS = 2000

# A detected letter must persist this long before it counts. Shorter than the
# 1000ms dwell in engine.py on purpose: this is a timed game, not transcription,
# and half a second of steady holding is plenty at 10fps.
DEFAULT_HOLD_MS = 420

# Ignore a detection this weak. Below this the model is guessing, and a guess
# would either score a letter the player did not mean or burn the answer window
# on noise.
DEFAULT_MIN_CONF = 0.62




class GameConfig:
    """Tunables for a run, all in milliseconds unless noted."""

    def __init__(
        self,
        ready_ms=DEFAULT_READY_MS,
        answer_ms=DEFAULT_ANSWER_MS,
        hold_ms=DEFAULT_HOLD_MS,
        min_conf=DEFAULT_MIN_CONF,
        max_cacti=0,
    ):
        self.ready_ms = int(ready_ms)
        self.answer_ms = int(answer_ms)
        self.hold_ms = int(hold_ms)
        self.min_conf = float(min_conf)
        # 0 means unlimited; otherwise the run ends after this many clears.
        self.max_cacti = int(max_cacti)

    def as_dict(self):
        return {
            "ready_ms": self.ready_ms,
            "answer_ms": self.answer_ms,
            "hold_ms": self.hold_ms,
            "min_conf": self.min_conf,
            "max_cacti": self.max_cacti,
        }


class GameState:
    def __init__(self, config=None, pool=None, now=None):
        self.config = config or GameConfig()
        self.pool = pool
        self.clock = now or time.monotonic
        self.phase = "idle"
        self.score = 0
        self.cleared = 0
        self.streak = 0
        self.best_streak = 0
        self.target = None
        self.plays = []            # letters cleared this run, in order
        self.misses = []           # letters missed this run
        self.wrong_attempts = 0    # wrong shapes shown; costs time, not the run
        self.last_seen = None      # the last wrong letter, for feedback
        self._phase_until = 0.0
        self._held_since = None    # when the current agreeing letter began
        self._held_letter = None
        self._message = ""

    # -- helpers ---------------------------------------------------------

    def _now(self):
        return self.clock()

    def _enter(self, phase, duration_ms=0):
        self.phase = phase
        self._phase_until = self._now() + duration_ms / 1000.0
        self._held_since = None
        self._held_letter = None

    @property
    def phase_remaining_ms(self):
        if self.phase in ("idle", "wrong", "finished"):
            return 0
        return max(0, int((self._phase_until - self._now()) * 1000))

    @property
    def phase_progress(self):
        """0..1 through the current timed phase, for the countdown ring."""
        if self.phase not in ("ready", "answer"):
            return 0.0
        total = self.config.ready_ms if self.phase == "ready" else self.config.answer_ms
        if total <= 0:
            return 0.0
        return min(1.0, max(0.0, 1.0 - self.phase_remaining_ms / total))

    # -- lifecycle -------------------------------------------------------

    def start(self):
        if self.pool is None or self.pool.empty:
            self._message = "no usable letters in the pool"
            self._enter("wrong")
            return False
        self.score = 0
        self.cleared = 0
        self.streak = 0
        self.best_streak = 0
        self.plays = []
        self.misses = []
        self.wrong_attempts = 0
        self.last_seen = None
        self._message = ""
        self._next_challenge()
        return True

    def _next_challenge(self):
        if self.config.max_cacti and self.cleared >= self.config.max_cacti:
            self._message = "you reached the end of the play"
            self._enter("finished")
            return
        self.target = self.pool.next()
        self._enter("ready", self.config.ready_ms)

    def _succeed(self):
        self.plays.append(self.target)
        self.score += 10 + 5 * min(self.streak, 4)
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)
        self.cleared += 1
        self._enter("correct", 900)

    def _time_out(self):
        """The answer window closed without the target being held.

        This is the only way a run ends. Showing the wrong shape is a normal
        part of learning a new alphabet, so it only costs the player time.
        """
        if self.target:
            self.misses.append(self.target)
        self.streak = 0
        self._message = "the light faded before the shape was ready"
        self._enter("wrong")

    # -- input -----------------------------------------------------------

    def observe(self, letter, confidence=0.0):
        """Feed the detector's latest reading in. Never blocks, never raises.

        A letter must be seen at hold_ms in a row, above min_conf, and must
        not change mid-hold, before it counts.
        """
        if self.phase not in ("ready", "answer"):
            return

        # A letter we are not confident enough to believe: ignore it and reset
        # the hold, but do NOT count it as a strike. The hand is clearly still
        # there, just ambiguous, and ending a run on a wobble would be
        # punishing the player for the model's uncertainty.
        if letter and confidence < self.config.min_conf:
            self._held_since = None
            self._held_letter = None
            return

        # No hand in frame. Let the clock run out rather than ending the run:
        # a blank reading is as likely to be bad lighting or the player still
        # getting into position as it is a deliberate walk-away, and at 10fps
        # a single dropped frame must not be fatal.
        if not letter:
            self._held_since = None
            self._held_letter = None
            return

        if letter != self._held_letter:
            self._held_letter = letter
            self._held_since = self._now()
            return

        if self._held_since is None:
            return
        if (self._now() - self._held_since) * 1000.0 < self.config.hold_ms:
            return

        if letter == self.target:
            # Only the scored window counts. A correct shape during the ready
            # beat is deliberately ignored.
            if self.phase == "answer":
                self._succeed()
            return

        # A different letter held long enough. This does NOT end the run: the
        # player gets another go until the window closes. Reset the hold so the
        # same wrong shape cannot burn the whole window on its own, and record
        # it so the page can show what the camera actually saw.
        if self.phase == "answer":
            self.wrong_attempts += 1
            self.last_seen = letter
            self._held_since = None
            self._held_letter = None

    def tick(self):
        """Advance the clock. Call once per animation frame."""
        now = self._now()
        if self.phase == "ready" and now >= self._phase_until:
            self._enter("answer", self.config.answer_ms)
        elif self.phase == "answer" and now >= self._phase_until:
            self._time_out()
        elif self.phase == "correct" and now >= self._phase_until:
            self._next_challenge()

    # -- output ----------------------------------------------------------

    def snapshot(self, detected=None, confidence=0.0):
        return {
            "phase": self.phase,
            "target": self.target,
            "score": self.score,
            "cleared": self.cleared,
            "streak": self.streak,
            "best_streak": self.best_streak,
            "plays": list(self.plays),
            "misses": list(self.misses),
            "wrong_attempts": self.wrong_attempts,
            "last_seen": self.last_seen,
            "remaining_ms": self.phase_remaining_ms,
            "progress": round(self.phase_progress, 3),
            "message": self._message,
            "detected": detected,
            "confidence": round(float(confidence), 3),
            "config": self.config.as_dict(),
        }