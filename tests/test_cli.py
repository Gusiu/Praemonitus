from pathlib import Path

import cli
import database
from ml.predictor import Predictor

# Sample runs used in the README example: three valid runs, one outside the
# training range and two invalid rows.
SAMPLE_RUNS = Path(__file__).parent / "data" / "demo_runs.csv"


def test_interactive_retries_invalid_values(tmp_path):
    answers = iter(["X", "m", "300", "abc", "310", "1400", "-1", "15", "100"])
    output = []
    predictor = Predictor(db_path=tmp_path / "c.db")
    cli.interactive(predictor, read=lambda _: next(answers), write=output.append)
    retries = [line for line in output if "Invalid value" in line]
    assert len(retries) == 3
    assert "FAILURE" in output[-1]
    assert len(database.get_history(10, db_path=tmp_path / "c.db")) == 1


def test_batch_and_export(tmp_path):
    db = tmp_path / "b.db"
    runs = tmp_path / "runs.csv"
    runs.write_text(
        "type,air_temp_k,process_temp_k,rotational_speed_rpm,torque_nm,tool_wear_min\n"
        "M,300,310,1500,40,100\n"
        "M,300,310,1400,15,100\n"
        "X,300,310,1500,40,100\n",
        encoding="utf-8",
    )
    output = []
    ok, failed = cli.run_batch(Predictor(db_path=db), str(runs), write=output.append)
    assert (ok, failed) == (2, 1)
    assert "rejected" in output[2]
    history = database.get_history(10, db_path=db)
    assert {row["source"] for row in history} == {"batch"}
    assert cli.main(["--db", str(db), "--export", str(tmp_path / "out.csv")]) == 0
    assert (tmp_path / "out.csv").read_text(encoding="utf-8").count("\n") == 3


def test_batch_on_sample_runs(tmp_path, capsys):
    db = tmp_path / "s.db"
    assert cli.main(["--db", str(db), "--batch", str(SAMPLE_RUNS)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("-> no failure")
    assert lines[1].endswith("-> FAILURE")
    assert lines[2].endswith("-> FAILURE")
    assert lines[3].endswith("(out of range)")
    assert lines[4].startswith("line 6: rejected - Product type")
    assert lines[5].startswith("line 7: rejected - Tool wear")
    assert lines[6] == "Batch done: 4 predicted, 2 rejected."
    assert len(database.get_history(10, db_path=db)) == 4
