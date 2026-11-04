"""Console interface: interactive prediction, history, CSV export and batch mode.

Examples:
    python cli.py
    python cli.py --history 10
    python cli.py --export history.csv
    python cli.py --batch runs.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

import database
from ml.predictor import Predictor
from ml.validation import FIELD_LABELS, InputError, validate_input

PROMPT_FIELDS = list(FIELD_LABELS)


def ask_value(name: str, ranges: dict, read=input, write=print) -> str:
    """Ask for one field until it passes hard validation on its own."""
    label = FIELD_LABELS[name]
    hint = "L/M/H" if name == "type" else f"training range {ranges[name]['min']:g}-{ranges[name]['max']:g}"
    probe = {
        "type": "M",
        "air_temp_k": 300,
        "process_temp_k": 310,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40,
        "tool_wear_min": 100,
    }
    while True:
        value = read(f"{label} ({hint}): ").strip()
        probe[name] = value
        try:
            validate_input(probe)
            return value
        except InputError as exc:
            relevant = [m for m in exc.messages if m.startswith(label)] or exc.messages
            write("  Invalid value - " + "; ".join(relevant) + ". Try again.")


def format_result(result) -> str:
    lines = [
        f"Failure probability: {result.probability:.3f} (threshold {result.threshold:.3f})",
        f"Decision: {'FAILURE - stop and inspect the machine' if result.prediction else 'no failure'}",
        "Derived: temperature difference {temp_diff_k:.1f} K, power {power_w:.0f} W, "
        "strain {strain_min_nm:.0f} min*Nm".format(**result.derived),
    ]
    if result.out_of_range:
        lines.append(
            "WARNING: outside the training data range: "
            + ", ".join(FIELD_LABELS[f] for f in result.out_of_range_fields)
        )
    lines.append(f"Model: {result.model_id}, saved as record {result.record_id}")
    return "\n".join(lines)


def interactive(predictor: Predictor, read=input, write=print) -> None:
    raw = {name: ask_value(name, predictor.ranges, read, write) for name in PROMPT_FIELDS}
    write(format_result(predictor.predict(raw, source="cli")))


def run_batch(predictor: Predictor, path: str, write=print) -> tuple[int, int]:
    """Predict every row of a CSV with the six input columns; invalid rows are reported."""
    ok = failed = 0
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            raw = {name: row.get(name) for name in PROMPT_FIELDS}
            try:
                result = predictor.predict(raw, source="batch")
            except InputError as exc:
                failed += 1
                write(f"line {line_no}: rejected - {exc}")
                continue
            ok += 1
            flag = " (out of range)" if result.out_of_range else ""
            write(f"line {line_no}: p={result.probability:.3f} -> {result.label}{flag}")
    write(f"Batch done: {ok} predicted, {failed} rejected.")
    return ok, failed


def print_history(limit: int, write=print, db_path=None) -> None:
    rows = database.get_history(limit, db_path=db_path or database.DEFAULT_DB_PATH)
    if not rows:
        write("No predictions yet.")
        return
    for row in rows:
        write(
            f"{row['id']:>5}  {row['created_at']}  {row['source']:<9}  p={row['probability']:.3f}"
            f"  {'FAILURE' if row['prediction'] else 'ok':<7}"
            f"  {'out-of-range' if row['out_of_range'] else ''}  {row['input_json']}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Praemonitus - machine failure prediction")
    parser.add_argument("--history", type=int, metavar="N", help="show the last N predictions")
    parser.add_argument("--export", metavar="FILE", help="export the prediction history to CSV")
    parser.add_argument("--batch", metavar="FILE", help="predict every row of a CSV file")
    parser.add_argument("--db", metavar="FILE", help="database path (default predictions.db)")
    args = parser.parse_args(argv)
    db_path = args.db or database.DEFAULT_DB_PATH

    if args.history is not None:
        print_history(args.history, db_path=db_path)
        return 0
    if args.export:
        count = database.export_csv(args.export, db_path=db_path)
        print(f"Exported {count} rows to {args.export}")
        return 0
    predictor = Predictor(db_path=db_path)
    if args.batch:
        _, failed = run_batch(predictor, args.batch)
        return 1 if failed else 0
    try:
        interactive(predictor)
    except (KeyboardInterrupt, EOFError):
        print()
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
