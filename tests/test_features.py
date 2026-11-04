import math

import pandas as pd
import pytest

from ml.features import add_physics_features, compute_derived, feature_step


def sample():
    return pd.DataFrame(
        {
            "type": ["M", "L"],
            "air_temp_k": [300.0, 298.1],
            "process_temp_k": [310.0, 308.6],
            "rotational_speed_rpm": [1500, 1551],
            "torque_nm": [40.0, 42.8],
            "tool_wear_min": [100, 0],
        }
    )


def test_temperature_difference():
    derived = compute_derived(sample())
    assert derived["temp_diff_k"].tolist() == pytest.approx([10.0, 10.5])


def test_power_by_hand():
    # 40 Nm * 1500 rpm * 2*pi/60 = 2000*pi W = 6283.19 W
    derived = compute_derived(sample())
    assert derived["power_w"].iloc[0] == pytest.approx(2000 * math.pi)
    assert derived["power_w"].iloc[0] == pytest.approx(6283.185, abs=1e-3)


def test_strain_by_hand():
    derived = compute_derived(sample())
    assert derived["strain_min_nm"].tolist() == pytest.approx([4000.0, 0.0])


def test_physics_step_adds_columns_and_drops_extras():
    X = sample().assign(udi=[1, 2], twf=[0, 1])
    out = feature_step("physics").fit_transform(X)
    assert "udi" not in out.columns and "twf" not in out.columns
    assert list(out.columns)[-3:] == ["temp_diff_k", "power_w", "strain_min_nm"]
    assert add_physics_features(X).shape == (2, 9)


def test_unknown_feature_set():
    with pytest.raises(ValueError):
        feature_step("magic")
