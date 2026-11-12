import pickle
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import psycopg2
from streamlit.testing.v1 import AppTest

import api.app as api


def test_active_model_loads_when_default_version_matches(tmp_path, monkeypatch):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(pickle.dumps({"model": "first"}))
    cursor = Mock()
    cursor.fetchone.return_value = ("v1.0.0", str(artifact))
    connection = Mock()
    connection.cursor.return_value = cursor
    monkeypatch.setattr(api.psycopg2, "connect", lambda *_: connection)
    monkeypatch.setattr(api, "current_model", None)
    monkeypatch.setattr(api, "current_version", "v1.0.0")

    api.load_active_model()

    assert api.current_model == {"model": "first"}
    assert api.current_version == "v1.0.0"
    connection.close.assert_called_once()


def test_missing_new_artifact_keeps_last_good_model(tmp_path, monkeypatch):
    cursor = Mock()
    cursor.fetchone.return_value = ("v2.0.0", str(tmp_path / "missing.pkl"))
    connection = Mock()
    connection.cursor.return_value = cursor
    monkeypatch.setattr(api.psycopg2, "connect", lambda *_: connection)
    model = object()
    monkeypatch.setattr(api, "current_model", model)
    monkeypatch.setattr(api, "current_version", "v1.0.0")

    api.load_active_model()

    assert api.current_model is model
    assert api.current_version == "v1.0.0"
    connection.close.assert_called_once()


def test_dashboard_renders_toggles_and_recovers_from_query_error(monkeypatch):
    connections = []
    failing = [False]

    def connect(*_):
        connection = Mock()
        connections.append(connection)
        return connection

    def read_sql(query, connection):
        if failing[0]:
            raise RuntimeError("query failed")
        if "FROM predictions" in query:
            return pd.DataFrame({"timestamp": ["2026-01-01"], "prediction_prob": [0.3], "prediction_class": [0]})
        return pd.DataFrame({"window_end": ["2026-01-01"], "metric_name": ["accuracy"], "metric_value": [0.8]})

    monkeypatch.setattr(psycopg2, "connect", connect)
    monkeypatch.setattr(pd, "read_sql", read_sql)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=15)
    app.run()

    assert not app.exception
    assert app.toggle[0].value is True
    assert len(app.get("plotly_chart")) == 3
    assert connections[-1].close.call_count == 1

    failing[0] = True
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert app.toggle[0].value is False
    assert "query failed" in app.error[0].value
    assert connections[-1].close.call_count == 1

    failing[0] = False
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) == 3
    assert connections[-1].close.call_count == 1
