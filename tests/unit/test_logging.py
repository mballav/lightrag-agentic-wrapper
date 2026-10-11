import json
import logging
import sys

from lightrag_wrapper.log import configure_logging, get_logger, uvicorn_log_config


def test_configure_logging_renders_json_when_not_a_tty(capsys, monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    configure_logging("DEBUG")
    get_logger("lightrag_wrapper.test").info("adapter_ready", mode="embedded")
    rendered = capsys.readouterr().err
    line = next(line for line in rendered.splitlines() if "adapter_ready" in line)
    payload = json.loads(line)
    assert payload["event"] == "adapter_ready"
    assert payload["mode"] == "embedded"
    assert payload["level"] == "info"


def test_configure_logging_renders_console_on_a_tty(capsys, monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    configure_logging("INFO")
    get_logger("lightrag_wrapper.test").info("tty_ready", mode="server")
    rendered = capsys.readouterr().err
    assert "tty_ready" in rendered
    assert "server" in rendered
    assert not rendered.lstrip().startswith("{")


def test_exception_log_includes_traceback(capsys):
    configure_logging("INFO")
    try:
        raise RuntimeError("catalog boom")
    except RuntimeError:
        get_logger("lightrag_wrapper.test").exception(
            "status_failed",
            track_id="track-1",
        )
    rendered = capsys.readouterr().err
    assert "status_failed" in rendered
    assert "catalog boom" in rendered
    assert "track-1" in rendered


def test_unknown_log_level_falls_back_to_info():
    config = uvicorn_log_config("not-a-level")
    assert config["root"]["level"] == "INFO"
    configure_logging("not-a-level")
    assert logging.getLogger().level == logging.INFO
