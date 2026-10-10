"""Zero-regression tests for ASL mode and HandPipeline defaults."""

from signlang.config import MAX_HANDS, MIRROR, GPU_DELEGATE, DETECT_EVERY
from signlang.hands import HandPipeline


def test_hand_pipeline_defaults_unchanged():
    """Verify HandPipeline default arguments and attributes remain identical for ASL."""
    pipe = HandPipeline(source="nonexistent_source_for_test")
    try:
        assert pipe.num_hands == 1
        assert pipe.num_hands == MAX_HANDS
        assert pipe.mirror == MIRROR
        assert pipe.detect_every == DETECT_EVERY
    finally:
        pipe.close()


def test_hand_pipeline_num_hands_parameterized():
    """Verify HandPipeline cleanly accepts num_hands=2 for ISL mode."""
    pipe = HandPipeline(source="nonexistent_source_for_test", num_hands=2)
    try:
        assert pipe.num_hands == 2
    finally:
        pipe.close()
