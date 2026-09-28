"""Structured logging system with secret redaction for RAPHAEL."""

import logging
import re
import sys

# Known token/secret patterns to redact automatically
SENSITIVE_PATTERNS = [
    re.compile(r"(nvapi-[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(sk-[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(gsk_[A-Za-z0-9_-]{10,})", re.IGNORECASE),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/-]+", re.IGNORECASE),
]


class SecretMaskingFilter(logging.Filter):
    """Logging filter that masks sensitive API keys and tokens from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact(v) if isinstance(v, str) else v for k, v in record.args.items()
                }
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(
                    self._redact(v) if isinstance(v, str) else v for v in record.args
                )
        return True

    @staticmethod
    def _redact(text: str) -> str:
        for pattern in SENSITIVE_PATTERNS:
            text = pattern.sub("[REDACTED_SECRET]", text)
        return text


class RaphaelFormatter(logging.Formatter):
    """Clean, structured formatter for RAPHAEL console output."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


def setup_logging(log_level: str | None = None) -> logging.Logger:
    """Initialize structured logging for RAPHAEL with secret masking."""
    if log_level is None:
        from raphael.config import get_settings

        log_level = get_settings().app.log_level

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(RaphaelFormatter())
    console_handler.addFilter(SecretMaskingFilter())

    root_logger.addHandler(console_handler)

    # Silence overly verbose external libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger("raphael")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Obtain a namespaced logger for RAPHAEL components."""
    if not name.startswith("raphael"):
        name = f"raphael.{name}"
    return logging.getLogger(name)
