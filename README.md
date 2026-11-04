# Praemonitus

*Praemonitus, praemunitus* - forewarned is forearmed. Praemonitus predicts,
from the parameters of a machining run, whether the run will end with a
machine failure, so that maintenance can stop the process and inspect the
machine first. Course project for *Artificial Intelligence Systems*.

The model is trained on the public AI4I 2020 Predictive Maintenance Dataset.
The application validates the input, returns the failure probability and a
decision, stores every call in SQLite and shows the history.

## Result

Random forest on raw and physics-derived features (temperature difference,
power, tool strain), decision threshold chosen for a cost ratio of 10:1
(missed failure : false alarm). On the held-out test set (2000 runs, 68
failures), used once after the choice was frozen:

| Metric | Model | Always "no failure" |
|---|---|---|
| PR-AUC | 0.871 | 0.034 |
| Recall (failures found) | 0.824 (56 of 68) | 0.000 |
| Precision | 0.918 | - |
| False alarms per 1000 runs | 2.5 | 0.0 |
| Cost per 1000 runs (10*FN + FP) | 62.5 | 340.0 |
| Accuracy | 0.992 | 0.966 |

Failures produced by the heat dissipation, power and overstrain rules are
found in 56 of 57 cases; tool wear and unexplained failures are random with
respect to the inputs. Details: `reports/report.md` and the notebook.

## Requirements

- Python 3.12
- the libraries pinned in `requirements.txt`: pandas, NumPy, scikit-learn
  1.9.1, SciPy, matplotlib, seaborn, joblib, Pydantic, Streamlit, pytest,
  Jupyter (nbconvert, nbformat, ipykernel); `sqlite3` from the standard library
- optional, only to rebuild the report: pandoc and LibreOffice

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Web interface (form, result, history, CSV export, model information):

```bash
streamlit run app.py
```

Console:

```bash
python cli.py                       # interactive, asks again on invalid values
python cli.py --history 10          # last 10 predictions
python cli.py --export history.csv  # export the history to CSV
python cli.py --batch docs/demo_runs.csv
```

Predictions are stored in `predictions.db` (ignored by git); set the
`PRAEMONITUS_DB` environment variable or `--db` to use another file.

## Reproducing the model

```bash
python -m ml.train
```

It verifies the SHA-256 of `data/raw/ai4i2020.csv`, splits the data, compares
the models with cross-validation, tunes the threshold, evaluates the frozen
choice on the test set and writes `models/model.joblib` and
`models/model_card.json`. Training is deterministic (`random_state=42`); the
card also stores the file hash, which the application checks before loading.

The full analysis is in `notebooks/analysis_and_model.ipynb`, committed with
its outputs. To execute it again:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis_and_model.ipynb
```

## Tests

```bash
pytest
```

Data checksum, absence of leakage, derived feature formulas, input
validation, model save and reload (including rejection of a modified file),
predictor, database (including an SQL injection attempt), console interface
and the Streamlit app (`streamlit.testing.v1.AppTest`).

## Report

`reports/report.md` is the single source; `report.docx` and `report.pdf` are
generated from it:

```bash
cd reports
pandoc report.md -o ../report.docx
pandoc report.md -o ../report.pdf --pdf-engine=xelatex -V mainfont="DejaVu Serif"
```

Without XeLaTeX: `soffice --headless --convert-to pdf report.docx`.

## Structure

```
app.py                  Streamlit interface
cli.py                  console interface
database.py             SQLite log of predictions
ml/data.py              loading, checksum, column names, split, quality checks
ml/features.py          feature sets and preprocessing (inside the Pipeline)
ml/train.py             model comparison, threshold, final model, model card
ml/evaluate.py          metrics, cost scorer, plots, error analysis
ml/validation.py        input validation (Pydantic)
ml/predictor.py         shared prediction core
models/                 model.joblib and model_card.json
notebooks/              analysis_and_model.ipynb
reports/                report.md, figures/, results.json
docs/defense.md         questions and answers for the defence
docs/demo.md            demo script; docs/demo_runs.csv sample batch
data/raw/ai4i2020.csv   original data; data/SOURCE.md attribution
tests/                  pytest suite
report.docx, report.pdf generated report
```

## Data and licence

AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine
Learning Repository. https://doi.org/10.24432/C5HS5C - licence CC BY 4.0.
Source paper: S. Matzka, *Explainable Artificial Intelligence for Predictive
Maintenance Applications*, AI4I 2020. The data are synthetic and contain no
personal data; the file is included unchanged (see `data/SOURCE.md`).

The code is released under the licence in `LICENSE`.
