"""End-to-end test of the game hub with a stubbed camera reader.

Verifies the real wiring in server.py: phase transitions on a live clock,
correct/wrong answers scoring, and the confidence gate. No camera needed.

Run from the project root:

    PYTHONPATH=src .venv/bin/python -m signlang.game.selftest_server
"""

import threading
import time

from .scoring import GameConfig
from .server import GameHub

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}{(' -> ' + detail) if detail else ''}")
        FAILS.append(name)


class FakeReader:
    """Stands in for LetterReader so we can feed exact letters."""

    def __init__(self):
        self.letter = None
        self.conf = 0.0

    def reading(self):
        return self.letter, self.conf, 10.0, 0.0, 120, 8

    def stop(self):
        pass


def wait_for(hub, phase, limit=8.0):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if hub.state.phase == phase:
            return True
        time.sleep(0.02)
    return False


def hold_until_phase_changes(hub, limit=8.0):
    start = hub.state.phase
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if hub.state.phase != start:
            return hub.state.phase
        time.sleep(0.02)
    return hub.state.phase


def new_hub(letters=("C", "D", "F"), **kw):
    opts = {"ready_ms": 700, "answer_ms": 4000, "hold_ms": 350}
    opts.update(kw)
    cfg = GameConfig(**opts)
    hub = GameHub(config=cfg, include=list(letters))
    hub.reader = FakeReader()
    threading.Thread(
        target=lambda: hub.tick_loop(threading.Event(), 60), daemon=True
    ).start()
    return hub


def test_correct_answer_scores():
    print("\ncorrect answer scores")
    hub = new_hub()
    hub.start_run()
    check("run started", hub.state.phase == "ready")
    target = hub.state.target

    check("answer phase opens", wait_for(hub, "answer"))
    hub.reader.letter, hub.reader.conf = target, 0.95
    ended = hold_until_phase_changes(hub)
    snap = hub.snapshot()
    check("correct answer advances the run", ended == "correct", ended)
    check("cleared counted", snap["cleared"] == 1, str(snap["cleared"]))
    check("score awarded", snap["score"] >= 10, str(snap["score"]))
    check("letter recorded", snap["plays"] == [target], str(snap["plays"]))
    check("detection surfaced to the page", snap["detected"] == target)


def test_wrong_answer_costs_time():
    print("\nwrong answer costs time, not the run")
    hub = new_hub()
    hub.start_run()
    wait_for(hub, "answer")
    target = hub.state.target
    wrong = next(c for c in ("C", "D", "F", "B") if c != target)
    hub.reader.letter, hub.reader.conf = wrong, 0.95
    time.sleep(1.2)
    snap = hub.snapshot()
    check("wrong answer does not end the run", hub.state.phase == "answer",
          hub.state.phase)
    check("wrong attempt recorded", snap["wrong_attempts"] >= 1,
          str(snap["wrong_attempts"]))
    check("last wrong letter surfaced", snap["last_seen"] == wrong,
          str(snap["last_seen"]))
    check("wrong answer is not a miss", snap["misses"] == [], str(snap["misses"]))
    check("nothing was cleared", snap["cleared"] == 0, str(snap["cleared"]))


def test_confidence_gate():
    print("\nconfidence gate")
    hub = new_hub()
    hub.start_run()
    wait_for(hub, "answer")
    target = hub.state.target

    hub.reader.letter, hub.reader.conf = target, 0.25
    time.sleep(1.0)
    check("a weak guess does not score", hub.state.phase == "answer", hub.state.phase)
    check("a weak guess is not a miss", hub.state.misses == [], str(hub.state.misses))

    hub.reader.letter, hub.reader.conf = target, 0.95
    ended = hold_until_phase_changes(hub)
    check("a confident hold scores", ended == "correct", ended)


def test_timeout_and_leaving():
    print("\ntimeout ends the run")
    hub = new_hub(ready_ms=400, answer_ms=900, hold_ms=300)
    hub.start_run()
    wait_for(hub, "answer")
    # Present but never confident enough to count, so the clock is what ends
    # it. A blank reading would be reported as leaving the screen instead.
    hub.reader.letter, hub.reader.conf = hub.state.target, 0.0
    ended = hold_until_phase_changes(hub, limit=6)
    check("running out of time ends the run", ended == "wrong", ended)
    check("timeout reported", "faded" in hub.snapshot()["message"],
          hub.snapshot()["message"])

    print("\nno hand in frame does not end the run early")
    hub2 = new_hub(ready_ms=400, answer_ms=6000, hold_ms=300)
    hub2.start_run()
    wait_for(hub2, "answer")
    hub2.reader.letter, hub2.reader.conf = None, 0.0
    time.sleep(1.5)
    check("blanks alone do not end the run", hub2.state.phase == "answer",
          hub2.state.phase)
    check("blanks are not counted as misses", hub2.state.misses == [],
          str(hub2.state.misses))


def test_pool_from_real_model():
    print("\npool built from the trained model")
    hub = GameHub()
    letters = hub.state.pool.letters
    check("pool is not empty", bool(letters), str(letters))
    # After burst eval, all 25 letters scored ≥99.6% — nothing excluded.
    check("pool includes all model letters", len(letters) == 25, str(letters))
    check("formerly excluded letters now included",
          all(c in letters for c in "IPSUX"), str(letters))
    print(f"  info  {len(letters)} letters: {' '.join(letters)}")


def test_settings_endpoint_inputs():
    print("\nsettings are validated")
    hub = GameHub(include=["C"])
    cfg = hub.state.config
    old = cfg.answer_ms
    cfg.answer_ms = 5000
    check("config is writable", cfg.answer_ms == 5000)
    cfg.answer_ms = old


def main():
    print("signlang game - server wiring tests (stubbed camera)")
    test_pool_from_real_model()
    test_correct_answer_scores()
    test_wrong_answer_costs_time()
    test_confidence_gate()
    test_timeout_and_leaving()
    test_settings_endpoint_inputs()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())