"""Tests for the parts of the game that need no camera.

Run with:

    .venv/bin/python -m signlang.game.selftest
"""

import random
import sys

from .pool import DEFAULT_EXCLUDE, LetterPool, build_pool, model_labels
from .scoring import GameConfig, GameState

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}{(' -> ' + detail) if detail else ''}")
        FAILS.append(name)


class FakeClock:
    """Monotonic clock the test drives by hand, so timing is exact."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance_ms(self, ms):
        self.t += ms / 1000.0


def run_scored(state, clock, letter, conf=0.95, hold_ms=420):
    """Feed one letter repeatedly for longer than the hold window."""
    state.observe(letter, conf)
    clock.advance_ms(hold_ms + 60)
    for _ in range(2):
        state.observe(letter, conf)
        state.tick()


def test_pool():
    print("\npool")
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # After burst evaluation all 25 letters scored ≥99.6%, so nothing is excluded.
    pool = build_pool(labels=labels)
    check("pool includes all letters", len(pool) == 26, f"got {len(pool)}: {pool}")
    check("pool keeps a clear letter", "C" in pool, str(pool))
    check("pool includes formerly excluded letters", "I" in pool and "P" in pool and "U" in pool)
    check("pool is a subset of the model labels", set(pool) <= set(labels))

    forced = build_pool(labels=labels, exclude=set())
    check("exclude=set() overrides defaults", "I" in forced and "A" in forced)
    check("T included after burst eval", "T" in forced)

    only_c = build_pool(include=["C", "B"], labels=labels)
    check("include filters to the request", sorted(only_c) == ["B", "C"], str(only_c))

    bag = LetterPool(["A", "B", "C"], rng=random.Random(1))
    first = [bag.next() for _ in range(3)]
    check("bag draws each letter once per cycle", sorted(first) == ["A", "B", "C"])
    check("bag refills after a full cycle", bag.next() in "ABC")

    check("empty pool is reported as empty", LetterPool([]).empty)
    check("empty pool yields None", LetterPool([]).next() is None)

    print(f"  info  live pool from your model: {len(build_pool())} letters -> "
          f"{' '.join(build_pool())}")
    print(f"  info  model labels on disk: {len(model_labels())}")


def test_countdown_timing():
    print("\ntiming: get-ready beat is not scored")
    clock = FakeClock()
    cfg = GameConfig(ready_ms=2000, answer_ms=3000, hold_ms=400)
    pool = LetterPool(["C"])
    state = GameState(cfg, pool, now=clock)

    state.start()
    check("run starts in the ready phase", state.phase == "ready")
    check("target is drawn", state.target == "C")

    # Right answer during the get-ready beat must be ignored entirely. Advance
    # in small steps so tick() sees each one and cannot skip past the deadline.
    deadline = 2000
    consumed = 0
    while consumed < deadline - 100:
        state.observe("C", 0.95)
        clock.advance_ms(100)
        consumed += 100
        state.tick()
    check("correct letter during ready does not score", state.phase == "ready")
    check("still ready just before the beat ends", state.phase == "ready")

    state.observe("C", 0.95)
    clock.advance_ms(150)
    state.tick()
    check("answer phase opens after the ready beat", state.phase == "answer")

    print("\ntiming: scored window")
    state2 = GameState(GameConfig(ready_ms=1000, answer_ms=3000), LetterPool(["C"]), now=clock)
    state2.start()
    clock.advance_ms(1100)
    state2.tick()
    check("answer phase is live", state2.phase == "answer")
    check("remaining time is near 3s", 2500 < state2.phase_remaining_ms <= 3000,
          str(state2.phase_remaining_ms))
    check("progress starts near zero", state2.phase_progress < 0.2)


def test_hold_requirement():
    print("\nhold requirement")
    clock = FakeClock()
    state = GameState(GameConfig(ready_ms=100, answer_ms=5000, hold_ms=400),
                      LetterPool(["C"]), now=clock)
    state.start()
    clock.advance_ms(150)
    state.tick()
    check("answer phase live", state.phase == "answer")

    state.observe("C", 0.95)
    clock.advance_ms(150)
    state.observe("C", 0.95)
    check("a brief flick does not score", state.phase == "answer")

    clock.advance_ms(400)
    state.observe("C", 0.95)
    check("sustained hold scores", state.phase == "correct", state.phase)
    check("score awarded", state.score >= 10, str(state.score))
    check("cleared incremented", state.cleared == 1)

    print("\nconfidence gate")
    clock2 = FakeClock()
    s2 = GameState(GameConfig(ready_ms=100, answer_ms=5000, hold_ms=300, min_conf=0.7),
                   LetterPool(["C"]), now=clock2)
    s2.start()
    clock2.advance_ms(150)
    s2.tick()
    s2.observe("C", 0.40)
    clock2.advance_ms(400)
    s2.observe("C", 0.40)
    check("weak detection ignored", s2.phase == "answer")
    check("weak detection is not counted as a wrong answer", s2.misses == [])

    print("\nletter must stay stable")
    clock3 = FakeClock()
    s3 = GameState(GameConfig(ready_ms=100, answer_ms=5000, hold_ms=300),
                   LetterPool(["C"]), now=clock3)
    s3.start()
    clock3.advance_ms(150)
    s3.tick()
    s3.observe("C", 0.95)
    clock3.advance_ms(200)
    s3.observe("D", 0.95)     # changed mid-hold: resets the timer
    clock3.advance_ms(200)
    s3.observe("D", 0.95)
    check("switching letters restarts the hold", s3.phase == "answer")


def test_timeout_ends_run():
    print("\nwrong shape costs time, not the run")
    clock = FakeClock()
    # Single-letter pool so the target is known, not shuffled.
    state = GameState(GameConfig(ready_ms=100, answer_ms=5000, hold_ms=300),
                      LetterPool(["C"]), now=clock)
    state.start()
    clock.advance_ms(150)
    state.tick()

    state.observe("D", 0.95)
    clock.advance_ms(400)
    state.observe("D", 0.95)
    check("wrong letter does not end the run", state.phase == "answer", state.phase)
    check("wrong attempt recorded", state.wrong_attempts == 1,
          str(state.wrong_attempts))
    check("last wrong letter surfaced", state.last_seen == "D", str(state.last_seen))
    check("wrong letter is not a miss", state.misses == [], str(state.misses))

    # The player can still recover and clear the same target afterwards.
    state.observe("C", 0.95)
    clock.advance_ms(400)
    state.observe("C", 0.95)
    check("player can recover and score", state.phase == "correct", state.phase)

    print("\ntimeout ends the run")
    clock2 = FakeClock()
    s2 = GameState(GameConfig(ready_ms=100, answer_ms=1000, hold_ms=300),
                   LetterPool(["C"]), now=clock2)
    s2.start()
    clock2.advance_ms(150)
    s2.tick()
    clock2.advance_ms(1200)
    s2.tick()
    check("running out of time ends the run", s2.phase == "wrong")
    check("timeout is reported", "faded" in s2.snapshot()["message"],
          s2.snapshot()["message"])

    print("\nno hand in frame")
    clock3 = FakeClock()
    s3 = GameState(GameConfig(ready_ms=100, answer_ms=5000, hold_ms=300),
                   LetterPool(["C"]), now=clock3)
    s3.start()
    clock3.advance_ms(150)
    s3.tick()
    for _ in range(5):
        s3.observe(None, 0.0)
    check("blanks do not end the run early", s3.phase == "answer", s3.phase)
    check("blanks are not counted as misses", s3.misses == [], str(s3.misses))
    clock3.advance_ms(5000)
    s3.tick()
    check("blanks let the clock run out instead", s3.phase == "wrong", s3.phase)
    check("reported as a timeout", "faded" in s3.snapshot()["message"],
          s3.snapshot()["message"])


def test_progression():
    print("\nprogressing through a run")
    clock = FakeClock()
    letters = ["C", "D", "F"]
    cfg = GameConfig(ready_ms=100, answer_ms=5000, hold_ms=300)
    state = GameState(cfg, LetterPool(letters, rng=random.Random(7)), now=clock)
    state.start()

    cleared = 0
    while state.cleared < 3 and state.phase not in ("wrong", "finished"):
        clock.advance_ms(200)
        state.tick()
        if state.phase == "answer":
            run_scored(state, clock, state.target, hold_ms=300)
            cleared += 1
        if state.phase in ("wrong", "finished"):
            break
    check("three letters cleared in one run", cleared == 3, f"cleared={cleared}")
    check("state agrees on the count", state.cleared == 3, str(state.cleared))
    # Order is shuffled by design, so check the set and that nothing repeated
    # before the cycle was exhausted.
    check("plays cover each letter once", sorted(state.plays) == sorted(letters),
          str(state.plays))
    check("score grew", state.score >= 30, str(state.score))
    check("streak tracked", state.streak == 3, str(state.streak))

    print("\nround cap")
    clock2 = FakeClock()
    s2 = GameState(GameConfig(ready_ms=100, answer_ms=5000, hold_ms=300, max_cacti=2),
                   LetterPool(["C", "D", "F"]), now=clock2)
    s2.start()
    for _ in range(60):
        clock2.advance_ms(200)
        s2.tick()
        if s2.phase == "answer":
            run_scored(s2, clock2, s2.target, hold_ms=300)
        if s2.phase == "finished":
            break
    check("cactus cap ends the run cleanly", s2.phase == "finished", s2.phase)
    check("cap respected", s2.cleared == 2, str(s2.cleared))

    print("\nrestart")
    state.start()
    check("start resets the score", state.score == 0)
    check("start resets progress", state.plays == [] and state.misses == [])
    check("start re-enters ready", state.phase == "ready")


def test_snapshot():
    print("\nsnapshot shape")
    clock = FakeClock()
    state = GameState(GameConfig(), LetterPool(["C"]), now=clock)
    state.start()
    snap = state.snapshot(detected="D", confidence=0.71)
    for key in ("phase", "target", "score", "cleared", "remaining_ms", "progress",
                "detected", "confidence", "config"):
        check(f"snapshot has {key}", key in snap)
    check("confidence rounded for json", isinstance(snap["confidence"], float))
    check("detected passes through", snap["detected"] == "D")


def main():
    print("signlang game - logic tests (no camera needed)")
    test_pool()
    test_countdown_timing()
    test_hold_requirement()
    test_timeout_ends_run()
    test_progression()
    test_snapshot()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())