"""The shadow-play game: an ASL fingerspelling game for the browser.

Modules:
    pool       which letters are safe to ask for
    scoring    run state and phase timing, no camera involved
    detector   background hand classification feeding the game
    server     FastAPI app serving the page and small JSON state
"""

__all__ = ["pool", "scoring", "detector", "server"]