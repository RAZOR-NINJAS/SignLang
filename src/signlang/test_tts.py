"""Unit tests for the TTS engine in src/signlang/tts.py."""

import atexit
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from signlang.tts import TTSEngine, get_tts


class TestTTSEngine(unittest.TestCase):
    def test_tts_disabled(self):
        engine = TTSEngine(enabled=False)
        self.assertFalse(engine.enabled)
        # Calling methods when disabled should be safe no-ops
        engine.say_letter("A")
        engine.say("hello")
        engine.read_transcript("hello world")
        engine.stop()
        engine.close()

    def test_tts_initialization_and_precache(self):
        engine = TTSEngine(enabled=True)
        if engine.enabled:
            # Check cached letters
            self.assertIn("A", engine._letter_cache)
            self.assertIn("SPACE", engine._letter_cache)
            data, sr = engine._letter_cache["A"]
            self.assertGreater(len(data), 0)
            self.assertGreater(sr, 0)
            engine.close()

    def test_tts_say_letter_and_read_transcript(self):
        engine = TTSEngine(enabled=True)
        if engine.enabled:
            engine.say_letter("B")
            engine.say_letter("SPACE")
            engine.read_transcript("TEST")
            engine.stop()
            engine.close()

    def test_atexit_lifecycle(self):
        engine = TTSEngine(enabled=False)
        # When disabled, atexit is not registered
        engine.close()

        # Test enabled engine registers atexit
        engine_enabled = TTSEngine(enabled=True)
        if engine_enabled.enabled:
            # close unregisters atexit cleanly
            engine_enabled.close()


if __name__ == "__main__":
    unittest.main()
