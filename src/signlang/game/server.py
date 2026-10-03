"""Web server for the shadow-play game.

Deliberately stdlib-only: http.server plus threads, no FastAPI, no uvicorn.
That keeps the dependency list at what the project already needed for
collection and training, so installing the game on an exhibition machine
cannot fail on a web framework.

No video streams through here. The browser gets the page, the player clicks
Start, and a small JSON blob carries the target letter, the latest detection
and the run state. The camera belongs to this process (only one process can
open a webcam), so the corner picture is a single JPEG fetched from
/api/preview.jpg a few times a second - not a long-lived stream that could
wedge the page.
"""

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .pool import LetterPool, build_pool
from .scoring import GameConfig, GameState

STATIC_DIR = Path(__file__).resolve().parent / "static"


class GameHub:
    """Owns the run state, the detector thread and the animation clock.

    tick() runs at roughly 60Hz and advances phase timing. The detector runs
    at whatever the camera manages, about 10fps, and publishes into its own
    slot. The two never wait on each other: tick() reads the latest letter and
    carries on regardless.
    """

    def __init__(self, source=None, config=None, include=None, exclude=None,
                 seed=None):
        self.reader = None
        self.state = GameState(
            config or GameConfig(),
            LetterPool(build_pool(include=include, exclude=exclude), rng=seed),
        )
        self.source = source
        self._lock = threading.Lock()
        self._clients_at_start = 0

    # -- camera ----------------------------------------------------------

    def start_detector(self):
        if self.reader is not None:
            return True, "already running"
        from .detector import LetterReader

        reader = LetterReader(source=self.source,
                              hold_ms=self.state.config.hold_ms)
        try:
            ok = reader.start()
        except Exception as exc:
            return False, str(exc)
        if not ok:
            err = reader.error()
            return False, str(err) if err else "camera did not open"
        self.reader = reader
        return True, "camera running"

    def stop_detector(self):
        if self.reader is not None:
            self.reader.stop()
            self.reader = None

    # -- run control -----------------------------------------------------

    def start_run(self):
        if self.state.pool.empty:
            return {"ok": False, "error": "no usable letters in the pool"}
        with self._lock:
            self.state.start()
        return {"ok": True, "pool": self.state.pool.letters}

    def snapshot(self):
        letter, conf, fps, stale, frames, hands = (
            self.reader.reading() if self.reader else (None, 0.0, 0.0, 0.0, 0, 0)
        )
        with self._lock:
            snap = self.state.snapshot(detected=letter, confidence=conf)
        snap["camera"] = {
            "on": self.reader is not None,
            "fps": round(fps, 1),
            "stale_s": round(stale, 2),
            "frames": frames,
            "hands_seen": hands,
        }
        snap["pool_size"] = len(self.state.pool)
        snap["pool"] = self.state.pool.letters
        return snap

    # -- loop ------------------------------------------------------------

    def tick_loop(self, stop_event, hz=60):
        """Drive phase timing and feed the detector's reading into the game.

        The detector's latest reading is applied here, once per animation
        frame. GameState holds its own timer for the hold window, so calling
        observe() 60 times a second with the same letter behaves exactly like
        calling it once: the letter must persist for hold_ms regardless.
        """
        period = 1.0 / hz
        while not stop_event.is_set():
            letter, conf = (None, 0.0)
            if self.reader is not None:
                letter, conf = self.reader.reading()[:2]
            with self._lock:
                if letter is not None and conf:
                    self.state.observe(letter, conf)
                elif letter is None and self.state.phase in ("ready", "answer"):
                    # A blank reading resets the hold but does not end the
                    # attempt; the answer window is what ends it.
                    self.state.observe(None, 0.0)
                self.state.tick()
            stop_event.wait(period)


def make_handler(hub, stop_event):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass  # keep the booth console readable

        # -- helpers --

        def _send(self, code, body=b"", ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode("utf-8"))

        # -- routes --

        def do_GET(self):
            url = urlparse(self.path)
            route = url.path

            if route == "/":
                return self._serve_static("index.html")
            if route.startswith("/static/"):
                return self._serve_static(route[len("/static/"):])
            if route == "/api/state":
                return self._json(hub.snapshot())
            if route == "/api/preview.jpg":
                return self._serve_preview()
            if route == "/api/pool":
                return self._json({"pool": hub.state.pool.letters})
            if route == "/api/config":
                return self._json(hub.state.config.as_dict())
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            payload = {}
            if length:
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    return self._json({"error": "bad json"}, 400)

            if url.path == "/api/start":
                return self._json(hub.start_run())
            if url.path == "/api/camera":
                if payload.get("on"):
                    ok, msg = hub.start_detector()
                else:
                    hub.stop_detector()
                    ok, msg = True, "camera off"
                return self._json({"ok": ok, "message": msg})
            if url.path == "/api/settings":
                cfg = hub.state.config
                for key in ("ready_ms", "answer_ms", "hold_ms", "max_cacti"):
                    if key in payload:
                        try:
                            setattr(cfg, key, int(payload[key]))
                        except (TypeError, ValueError):
                            return self._json({"error": f"bad {key}"}, 400)
                if "min_conf" in payload:
                    try:
                        cfg.min_conf = float(payload["min_conf"])
                    except (TypeError, ValueError):
                        return self._json({"error": "bad min_conf"}, 400)
                return self._json(cfg.as_dict())
            return self._json({"error": "not found"}, 404)

        def _serve_preview(self):
            """One JPEG frame from the detector's camera.

            The page polls this a few times a second. It is deliberately not a
            streaming connection: a long-lived socket is one more thing to
            break at a booth, and a plain image request cannot wedge the page.
            """
            if hub.reader is None:
                return self._json({"error": "camera off"}, 503)
            jpeg, _seq = hub.reader.preview()
            if not jpeg:
                return self._json({"error": "no frame yet"}, 503)
            self._send(200, jpeg, "image/jpeg")

        def _serve_static(self, name):
            safe = Path(name)
            if safe.is_absolute() or ".." in safe.parts:
                return self._json({"error": "bad path"}, 400)
            path = STATIC_DIR / safe
            if not path.is_file():
                return self._json({"error": "not found"}, 404)
            ctype, _ = mimetypes.guess_type(str(path))
            if ctype is None:
                ctype = "application/octet-stream"
            if ctype.startswith("text/") or "javascript" in ctype:
                ctype += "; charset=utf-8"
            self._send(200, path.read_bytes(), ctype)

    return Handler


def serve(host="127.0.0.1", port=8000, source=None, config=None,
          include=None, exclude=None, camera=True):
    """Start the game. Blocks until Ctrl+C."""
    hub = GameHub(source=source, config=config, include=include, exclude=exclude)
    stop_event = threading.Event()

    print("signlang game - shadow play")
    print(f"  pool: {len(hub.state.pool)} letters  {' '.join(hub.state.pool.letters)}")
    print(f"  ready beat {hub.state.config.ready_ms}ms, "
          f"answer window {hub.state.config.answer_ms}ms")

    if camera:
        ok, msg = hub.start_detector()
        print(f"  camera: {msg}")
        if not ok:
            print("  the page will show no picture; start the camera from there")

    server = ThreadingHTTPServer((host, port), make_handler(hub, stop_event))
    clock = threading.Thread(target=hub.tick_loop, args=(stop_event,), daemon=True)
    clock.start()

    shown = f"http://{host}:{port}"
    print(f"  open: {shown}")
    print("  press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        stop_event.set()
        hub.stop_detector()
        server.shutdown()
        server.server_close()