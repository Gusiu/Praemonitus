# Demo script (3 minutes)

Start: `streamlit run app.py` (or `python cli.py` in a terminal). The values
below were checked against the saved model (`model_id` in
`models/model_card.json`); the probabilities come from `ml.predictor`.

| # | Case | Type | Air [K] | Process [K] | Speed [rpm] | Torque [Nm] | Wear [min] | Model result |
|---|---|---|---|---|---|---|---|---|
| 1 | normal run | M | 300.0 | 310.0 | 1500 | 40 | 100 | no failure, p = 0.00 (power 6283 W) |
| 2 | low power | M | 300.0 | 310.0 | 1400 | 15 | 100 | FAILURE, p = 0.55 (power 2199 W < 3500 W) |
| 3 | overheating | M | 302.0 | 310.4 | 1300 | 45 | 50 | FAILURE, p = 0.76 (difference 8.4 K < 8.6 K at 1300 rpm) |

Decision threshold: 0.292 (cost of a missed failure : false alarm = 10:1).

## Steps

1. **(20 s) Context.** Maintenance wants to know before a run whether it will
   end with a machine failure. The model was trained on AI4I 2020 (10 000
   synthetic runs, 3.4% failures).
2. **(30 s) Case 1.** Enter the normal run. Show the probability, the decision
   and the derived values a mechanic understands: temperature difference,
   power, tool strain.
3. **(30 s) Case 2.** Change speed to 1400 and torque to 15. Power falls to
   2199 W, below the 3500 W limit - the model raises an alarm.
4. **(30 s) Case 3.** Overheating: the temperature difference is 8.4 K at
   1300 rpm - the heat dissipation rule. Alarm.
5. **(20 s) Out of range.** Set air temperature to 310 K. The prediction is
   made, with the warning "outside the training data range".
6. **(20 s) Rejection.** Console: `python cli.py`, enter type `X` or tool wear
   `-5` - the value is rejected with a readable message and asked again.
7. **(30 s) History.** Show the history table (filter "failures"), the CSV
   export button, and `python cli.py --history 5`. Open "Model information":
   model id, training date, test metrics. Mention that the database stores no
   user data.

## Backup commands

```bash
python cli.py --batch docs/demo_runs.csv
python cli.py --history 5
python cli.py --export history.csv
```
