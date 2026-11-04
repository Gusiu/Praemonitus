"""Input validation for the application.

Hard validation rejects physically impossible or malformed input with a
readable message. Soft validation flags values outside the training ranges:
the prediction is still made, but with a warning.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

FIELD_LABELS = {
    "type": "Product type",
    "air_temp_k": "Air temperature [K]",
    "process_temp_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}


class ProcessInput(BaseModel):
    """Raw process parameters of one run, exactly as the operator enters them."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)

    type: Literal["L", "M", "H"]
    air_temp_k: float = Field(gt=0)
    process_temp_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: float = Field(ge=0)


class InputError(ValueError):
    """Raised when input fails hard validation; ``messages`` are user-readable."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("; ".join(messages))


def _format_errors(exc: ValidationError) -> list[str]:
    messages = []
    for err in exc.errors():
        field = str(err["loc"][0]) if err["loc"] else "input"
        label = FIELD_LABELS.get(field, field)
        messages.append(f"{label}: {err['msg']}")
    return messages


def validate_input(raw: dict) -> ProcessInput:
    """Hard validation. Accepts strings for numbers (e.g. from a CSV or console)."""
    cleaned = dict(raw)
    if isinstance(cleaned.get("type"), str):
        cleaned["type"] = cleaned["type"].strip().upper()
    try:
        return ProcessInput.model_validate(cleaned)
    except ValidationError as exc:
        raise InputError(_format_errors(exc)) from None


def out_of_range_fields(values: ProcessInput, ranges: dict) -> list[str]:
    """Soft validation: names of fields outside the [min, max] seen in training."""
    flagged = []
    for name, bounds in ranges.items():
        value = getattr(values, name)
        if value < bounds["min"] or value > bounds["max"]:
            flagged.append(name)
    return flagged
