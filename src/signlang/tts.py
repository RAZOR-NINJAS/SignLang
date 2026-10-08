"""Text-to-speech engine using Piper ONNX voice and SoundDevice.

Synthesizes speech in background threads so the video frame loop never drops below 60 fps.
Caches audio clips for standard alphabets in memory for zero-latency speech playback.
"""

import io
import os
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .config import PROJECT_ROOT, TTS_VOICE

try:
    import sounddevice as sd
    from piper import PiperVoice

    _HAS_TTS_LIBS = True
except ImportError:
    _HAS_TTS_LIBS = False


class TTSEngine:
    """Asynchronous TTS engine with in-memory caching for single characters."""

    def __init__(self, voice_model_path: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled and _HAS_TTS_LIBS
        self._voice: Optional[PiperVoice] = None
        self._queue: queue.Queue = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._letter_cache: dict[str, tuple[np.ndarray, int]] = {}
        self.is_speaking = False

        if not self.enabled:
            return

        if voice_model_path is None:
            voice_model_path = PROJECT_ROOT / f"{TTS_VOICE}.onnx"

        self.model_path = Path(voice_model_path)
        self.config_path = self.model_path.with_suffix(".onnx.json")

        if not (self.model_path.exists() and self.config_path.exists()):
            self.enabled = False
            return

        try:
            self._voice = PiperVoice.load(str(self.model_path), str(self.config_path))
        except Exception:
            self.enabled = False
            return

        # Pre-cache single-letter pronunciations for instant latency-free feedback
        self._precache_letters()

        # Start non-blocking worker thread for playback and dynamic synthesis
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _precache_letters(self):
        """Pre-synthesize audio for all alphabet letters to guarantee 0ms latency."""
        if not self._voice:
            return
        # Letters A through Z plus space and punctuation
        targets = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        targets.extend(["space", "delete", "clear"])

        for item in targets:
            spoken_text = item
            if item == "space":
                spoken_text = "space"
            elif item == "delete":
                spoken_text = "backspace"
            try:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    self._voice.synthesize_wav(spoken_text, wf)
                buf.seek(0)
                with wave.open(buf, "rb") as wf:
                    audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    sr = wf.getframerate()
                    self._letter_cache[item.upper()] = (audio_data, sr)
            except Exception:
                continue

    def _worker_loop(self):
        """Background thread consuming speech requests."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            task_type, payload = task
            self.is_speaking = True
            try:
                if task_type == "audio":
                    data, sr = payload
                    sd.play(data, sr)
                    sd.wait()
                elif task_type == "text":
                    text = str(payload).strip()
                    if text and self._voice:
                        buf = io.BytesIO()
                        with wave.open(buf, "wb") as wf:
                            self._voice.synthesize_wav(text, wf)
                        buf.seek(0)
                        with wave.open(buf, "rb") as wf:
                            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                            sr = wf.getframerate()
                        sd.play(data, sr)
                        sd.wait()
            except Exception:
                pass
            finally:
                self.is_speaking = False
                self._queue.task_done()

    def say_letter(self, letter: str):
        """Speak an individual alphabet character immediately from cache or synthesis."""
        if not self.enabled:
            return
        key = str(letter).strip().upper()
        if key in (" ", "_", "SPACE"):
            key = "SPACE"
        if key in self._letter_cache:
            data, sr = self._letter_cache[key]
            # Replace queued letter if queue is backed up
            try:
                self._queue.put_nowait(("audio", (data, sr)))
            except queue.Full:
                pass
        else:
            self.say(key)

    def say(self, text: str, interrupt: bool = False):
        """Queue dynamic speech for words or phrases."""
        if not self.enabled or not text:
            return
        if interrupt:
            self.stop()
        try:
            self._queue.put_nowait(("text", text))
        except queue.Full:
            pass

    def read_transcript(self, transcript: str):
        """Read the entire accumulated transcript buffer aloud."""
        clean = str(transcript).strip()
        if not clean:
            self.say("Transcript is empty", interrupt=True)
            return
        self.say(clean, interrupt=True)

    def stop(self):
        """Stop current audio playback and clear pending speech queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break
        if _HAS_TTS_LIBS:
            try:
                sd.stop()
            except Exception:
                pass
        self.is_speaking = False

    def close(self):
        """Shutdown TTS worker thread and release audio resources."""
        self._stop_event.set()
        self.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)


# Module-level singleton instance for convenient import
_instance: Optional[TTSEngine] = None


def get_tts() -> TTSEngine:
    """Retrieve or initialize the global TTS engine instance."""
    global _instance
    if _instance is None:
        enabled = os.environ.get("SIGNLANG_TTS", "1") not in ("0", "false", "False")
        _instance = TTSEngine(enabled=enabled)
    return _instance
