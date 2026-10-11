import uvicorn

from lightrag_wrapper.__main__ import main
from tests.conftest import make_config


def test_main_runs_uvicorn(monkeypatch):
    config = make_config(app_host="127.0.0.1", app_port=8123, log_level="DEBUG")

    async def fake_get_config():
        return config

    called = {}

    def fake_run(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr("lightrag_wrapper.__main__.get_config", fake_get_config)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    main()
    assert called["args"] == ("lightrag_wrapper.app:app",)
    assert called["kwargs"]["host"] == "127.0.0.1"
    assert called["kwargs"]["port"] == 8123
    assert called["kwargs"]["log_level"] == "debug"
    assert called["kwargs"]["log_config"]["root"]["handlers"] == ["structured"]
