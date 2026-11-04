import cli
import database
from ml.predictor import Predictor


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
