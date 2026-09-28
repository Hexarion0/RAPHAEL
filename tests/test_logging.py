"""Tests for structured logging and secret masking."""

import logging

from raphael.logging import SecretMaskingFilter, get_logger, setup_logging


def test_secret_masking_filter():
    """Verify that known sensitive API key patterns are masked."""
    redactor = SecretMaskingFilter()

    # Test raw message masking
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Using key nvapi-abcdef1234567890 for NIM API",
        args=(),
        exc_info=None,
    )
    assert redactor.filter(record) is True
    assert "nvapi-abcdef1234567890" not in record.msg
    assert "[REDACTED_SECRET]" in record.msg

    # Test args masking (tuple/list)
    record_args = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Calling Groq with key: %s",
        args=("gsk_1234567890abcdefgh",),
        exc_info=None,
    )
    assert redactor.filter(record_args) is True
    assert "gsk_1234567890abcdefgh" not in record_args.args[0]
    assert "[REDACTED_SECRET]" in record_args.args[0]


def test_setup_logging():
    """Verify logger creation and namespacing."""
    logger = setup_logging("DEBUG")
    assert logger.name == "raphael"

    sub_logger = get_logger("providers.nim")
    assert sub_logger.name == "raphael.providers.nim"
