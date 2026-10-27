import json
import logging
import sys

import lightrag_wrapper.log as log_module
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


def test_log_config_rolls_the_file_at_ten_megabytes():
    file_handler = uvicorn_log_config("INFO")["handlers"]["file"]
    assert file_handler["class"] == "logging.handlers.RotatingFileHandler"
    assert file_handler["maxBytes"] == 10 * 1024 * 1024
    assert file_handler["backupCount"] == 4
    assert uvicorn_log_config("INFO")["root"]["handlers"] == ["structured", "file"]


def test_configure_logging_writes_json_to_the_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(log_module, "_LOG_FILE", tmp_path / "wrapper.log")
    configure_logging("INFO")
    get_logger("lightrag_wrapper.test").info("file_ready", mode="server")
    for handler in logging.getLogger().handlers:
        handler.flush()
    line = next(
        line
        for line in (tmp_path / "wrapper.log").read_text(encoding="utf-8").splitlines()
        if "file_ready" in line
    )
    payload = json.loads(line)
    assert payload["event"] == "file_ready"
    assert payload["mode"] == "server"
    for handler in list(logging.getLogger().handlers):
        handler.close()
