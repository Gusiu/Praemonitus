from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    import database

    monkeypatch.setattr(database, "DEFAULT_DB_PATH", tmp_path / "app.db")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_form_renders(app):
    assert app.title[0].value == "Praemonitus"
    assert len(app.number_input) == 5
    assert app.info[0].value == "No predictions yet."


def test_power_failure_prediction_and_history(app):
    app.number_input(key="rotational_speed_rpm").set_value(1400.0)
    app.number_input(key="torque_nm").set_value(15.0)
    app.button[0].click().run()
    assert not app.exception
    assert any("FAILURE expected" in e.value for e in app.error)
    assert len(app.dataframe) == 1
    assert len(app.dataframe[0].value) == 1


def test_out_of_range_warning(app):
    app.number_input(key="air_temp_k").set_value(310.0)
    app.button[0].click().run()
    assert any("Outside the training data range" in w.value for w in app.warning)


def test_rejects_negative_tool_wear(app):
    app.number_input(key="tool_wear_min").set_value(-5.0)
    app.button[0].click().run()
    assert any("Tool wear" in e.value for e in app.error)
