"""Console + per-video/per-stage file logging.

Each stage writes to ``logs/<stage>.log`` and mirrors the same lines into
``logs/pipeline.log`` so a single file tells the whole story for a video.
"""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path
from typing import Callable

LOGGER_NAME = "multimodal_pipeline"

_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


def _file_handler(path: Path) -> logging.FileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    return handler


def configure(level: str = "INFO", *, console: bool = True) -> logging.Logger:
    """Configure the root pipeline logger (console output)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()
    if console:
        from rich.console import Console
        from rich.logging import RichHandler

        # rich has shuffled these keyword arguments across releases (``rich_text``
        # existed in 13.x and is gone in 15.x), so pass only what this rich
        # actually accepts instead of pinning the console to one version.
        wanted = {"rich_text": True, "markup": False, "show_path": False, "omit_repeated_times": False}
        accepted = set(inspect.signature(RichHandler.__init__).parameters)
        handler: logging.Handler = RichHandler(console=Console(stderr=True),
                                               **{key: value for key, value in wanted.items() if key in accepted})
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


class StageLogger:
    """Callable logger bound to one stage's file plus the video-level log."""

    def __init__(self, log_dir: Path, stage: str, mirror: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.stage = stage
        self.path = self.log_dir / f"{stage}.log"
        self.logger = logging.getLogger(f"{LOGGER_NAME}.{stage}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()
        self._handlers = [_file_handler(self.path)]
        if mirror:
            self._handlers.append(_file_handler(self.log_dir / "pipeline.log"))
        for handler in self._handlers:
            self.logger.addHandler(handler)

    def __call__(self, message: str, level: int = logging.INFO) -> None:
        self.logger.log(level, str(message))

    def info(self, message: str) -> None:
        self(message, logging.INFO)

    def warning(self, message: str) -> None:
        self(message, logging.WARNING)

    def error(self, message: str) -> None:
        self(message, logging.ERROR)

    def close(self) -> None:
        for handler in self._handlers:
            try:
                self.logger.removeHandler(handler)
                handler.close()
            except Exception:  # pragma: no cover - teardown nicety
                pass
        self._handlers = []

    def __enter__(self) -> "StageLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_stage_logger_factory(log_dir: Path) -> Callable[[str], StageLogger]:
    """Return ``stage_name -> StageLogger`` for one video dataset."""

    def factory(stage: str) -> StageLogger:
        return StageLogger(log_dir, stage)

    return factory
