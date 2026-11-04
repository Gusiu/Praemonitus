import pytest

from ml.predictor import Predictor
from ml.validation import InputError

NORMAL = {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0,
          "rotational_speed_rpm": 1500, "torque_nm": 40, "tool_wear_min": 100}


@pytest.fixture
def predictor(tmp_path):
    return Predictor(db_path=tmp_path / "p.db")


def test_obvious_power_failure(predictor):
    # 15 Nm at 1400 rpm is about 2199 W, below the 3500 W power failure limit.
    result = predictor.predict({**NORMAL, "rotational_speed_rpm": 1400, "torque_nm": 15})
    assert result.prediction == 1
    assert result.derived["power_w"] == pytest.approx(2199.1, abs=0.1)


def test_normal_run_and_logging(predictor):
    result = predictor.predict(NORMAL, source="cli")
    assert result.prediction == 0
    assert result.record_id is not None
    assert not result.out_of_range


def test_out_of_range_warning(predictor):
    result = predictor.predict({**NORMAL, "air_temp_k": 310.0})
    assert result.out_of_range and result.out_of_range_fields == ["air_temp_k"]


def test_invalid_input_not_logged(predictor):
    import database

    with pytest.raises(InputError):
        predictor.predict({**NORMAL, "type": "X"})
    assert database.get_history(10, db_path=predictor.db_path) == []
