import math

import pytest

from ml.validation import InputError, out_of_range_fields, validate_input

VALID = {
    "type": "M",
    "air_temp_k": 300.0,
    "process_temp_k": 310.0,
    "rotational_speed_rpm": 1500,
    "torque_nm": 40.0,
    "tool_wear_min": 100,
}
RANGES = {
    "air_temp_k": {"min": 295.3, "max": 304.5},
    "process_temp_k": {"min": 305.7, "max": 313.8},
    "rotational_speed_rpm": {"min": 1168, "max": 2886},
    "torque_nm": {"min": 3.8, "max": 76.6},
    "tool_wear_min": {"min": 0, "max": 253},
}


def test_valid_input():
    values = validate_input(VALID)
    assert values.type == "M" and values.torque_nm == 40.0
    assert out_of_range_fields(values, RANGES) == []


def test_strings_and_lowercase_type_are_accepted():
    values = validate_input({**{k: str(v) for k, v in VALID.items()}, "type": " h "})
    assert values.type == "H" and values.air_temp_k == 300.0


@pytest.mark.parametrize(
    "field, value",
    [
        ("type", "X"),
        ("air_temp_k", 0),
        ("process_temp_k", -1),
        ("rotational_speed_rpm", 0),
        ("torque_nm", -0.1),
        ("tool_wear_min", -5),
        ("air_temp_k", math.nan),
        ("torque_nm", math.inf),
        ("torque_nm", "abc"),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(InputError) as exc:
        validate_input({**VALID, field: value})
    assert exc.value.messages


def test_missing_and_extra_fields_are_rejected():
    with pytest.raises(InputError):
        validate_input({k: v for k, v in VALID.items() if k != "torque_nm"})
    with pytest.raises(InputError):
        validate_input({**VALID, "operator": "Jan"})


def test_boundaries():
    values = validate_input({**VALID, "torque_nm": 0, "tool_wear_min": 0})
    assert values.torque_nm == 0
    at_edges = validate_input({**VALID, "air_temp_k": 295.3, "tool_wear_min": 253})
    assert out_of_range_fields(at_edges, RANGES) == []


def test_out_of_range_is_soft():
    values = validate_input({**VALID, "air_temp_k": 310.0, "tool_wear_min": 300})
    assert out_of_range_fields(values, RANGES) == ["air_temp_k", "tool_wear_min"]
