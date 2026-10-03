"""Which letters the shadow-play game asks for.

Read straight off the trained model's label file, so retraining with new
letters needs no code change here. The exclusion list is the only thing to
edit when a pair turns out to be too confusable to grade a player on.
"""

import json
import random

from ..config import LABELS_PATH

# Pairs the classifier reliably mixes up. Asking a player for one of these
# letters turns a learning tool into a coin flip, because the model, not the
# player, decides whether the shape was right. Extend this as better data
# arrives.
#
# Updated after running eval_burst.py (5-fold burst-held-out cross-validation):
# all 25 letters scored ≥99.6% accuracy on unseen bursts, so no exclusions are
# needed.  Only 2 single-frame confusions (W→R, R→X) in 10,234 frames.
# NOTE: this is on one person's hands.  Re-run eval_burst.py after collecting
# data from a new person and re-exclude any letters that drop below 90%.
DEFAULT_EXCLUDE = frozenset()

# Letters with too little recorded data to trust on an unseen hand.
# T has only 56 samples (4 bursts) but scored 100% on burst eval.  Keep an eye
# on it after collecting from other people.
THIN_LABELS = frozenset()


def model_labels(path=None):
    """Labels the trained model knows, in its own order. Empty if untrained."""
    p = path or LABELS_PATH
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [str(v) for v in data] if isinstance(data, list) else []


def build_pool(include=None, exclude=None, thin=THIN_LABELS, labels=None):
    """Letters safe to ask for.

    include  explicit override; if given, only these are considered
    exclude  extra letters to drop, unioned with the defaults
    thin     letters dropped for want of training data
    """
    known = model_labels() if labels is None else list(labels)
    letters = [c for c in known if len(c) == 1 and c.isalpha()]

    if include:
        wanted = {c.upper() for c in include}
        letters = [c for c in letters if c in wanted]

    drop = set(DEFAULT_EXCLUDE if exclude is None else exclude) | set(thin or ())
    return [c for c in letters if c not in drop]


class LetterPool:
    """Draws targets for a run.

    Bag mode takes each letter once before repeating any, so a short run still
    shows variety. Random mode is uniform. Either way the never-repeat-until-
    exhausted rule holds, because a game that asks for the same letter three
    times running teaches nothing.
    """

    def __init__(self, letters, rng=None, mode="bag"):
        self.letters = list(dict.fromkeys(letters))
        # rng may be a random.Random, an int seed, or None
        self.rng = rng if isinstance(rng, random.Random) else random.Random(rng)
        self.mode = mode
        self._bag = []

    def __len__(self):
        return len(self.letters)

    @property
    def empty(self):
        return not self.letters

    def _refill(self):
        bag = list(self.letters)
        if self.mode == "random":
            self._bag = [self.rng.choice(bag)]
        else:
            self.rng.shuffle(bag)
            self._bag = bag

    def next(self):
        """Next target letter, or None when the pool is empty."""
        if not self.letters:
            return None
        if not self._bag:
            self._refill()
        return self._bag.pop()

    def remaining_in_cycle(self):
        """How many distinct targets are still unseen this cycle."""
        return len(self._bag)