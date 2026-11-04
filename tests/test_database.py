import csv
import json

import pytest

import database


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


def save(db, prediction=0, source="cli", inputs=None):
    return database.save_prediction(
        model_id="abc123",
        source=source,
        inputs=inputs or {"type": "M", "torque_nm": 40.0},
        probability=0.9 if prediction else 0.01,
        threshold=0.3,
        prediction=prediction,
        out_of_range=False,
        latency_ms=1.5,
        db_path=db,
    )


def test_save_and_history(db):
    first = save(db, 0)
    second = save(db, 1)
    history = database.get_history(10, db_path=db)
    assert [row["id"] for row in history] == [second, first]
    assert history[0]["created_at"].endswith("+00:00")
    assert json.loads(history[1]["input_json"]) == {"torque_nm": 40.0, "type": "M"}
    assert [r["id"] for r in database.get_history(10, prediction=1, db_path=db)] == [second]
    assert len(database.get_history(1, db_path=db)) == 1


def test_export_csv(db, tmp_path):
    save(db, 0)
    save(db, 1, source="batch")
    out = tmp_path / "history.csv"
    assert database.export_csv(out, db_path=db) == 2
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["source"] for r in rows] == ["cli", "batch"]
    assert list(rows[0]) == database.COLUMNS


def test_sql_injection_is_stored_as_text(db):
    payload = "M'); DROP TABLE predictions; --"
    save(db, 0, inputs={"type": payload})
    save(db, 0)
    history = database.get_history(10, db_path=db)
    assert len(history) == 2
    assert json.loads(history[1]["input_json"])["type"] == payload


def test_unknown_source_rejected(db):
    with pytest.raises(ValueError):
        save(db, 0, source="web")


def test_no_user_columns():
    assert not {"user", "username", "ip", "email"} & set(database.COLUMNS)
