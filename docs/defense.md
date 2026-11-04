# Defence preparation - questions and short answers

Every technical decision of the project, as a question with a short answer.
Numbers come from the executed notebook (`notebooks/analysis_and_model.ipynb`)
and `models/model_card.json`.

## Problem and metrics

**1. Why not accuracy?**
Failures are 3.39% of runs. A model that always answers "no failure" has
96.61% accuracy and finds nothing. Our model has 99.2% on the test set - the
difference of 2.6 points hides the whole value of the model. Accuracy is
reported only to show this trap.

**2. Why PR-AUC and not ROC-AUC?**
With a rare positive class there are very many true negatives, so the false
positive rate stays small and ROC-AUC looks good for almost any model (0.85-0.98
in our comparison, even for weak logistic regression 0.90-0.92). PR-AUC uses
precision, which drops with every false alarm, and its baseline is the failure
rate (0.034), so it separates models clearly (0.45-0.90).

**3. Where does the threshold come from and what does 10:1 mean?**
We assume a missed failure (FN) costs ten times a false alarm (FP) and choose
the threshold that minimises `10*FN + FP` with `TunedThresholdClassifierCV` on
training folds. The result is 0.292. 10:1 is an assumption, not a fact: for
5:1 the threshold is 0.316, for 20:1 it drops to 0.054 and the model raises
many more alarms (on the test set 96 false alarms instead of 5). The ratio is a
business decision to confirm with maintenance.

**4. Why is the cost normalised per 1000 runs?**
Folds have slightly different sizes; per 1000 runs the costs of folds and of
the test set are directly comparable and do not change the chosen threshold.

**5. Why classification and not regression?**
The course card and project specification require classification; the topic
list also allows regression. Classification satisfies both, and the decision
"stop or not" is binary.

## Data and leakage

**6. Why were TWF, HDF, PWF, OSF and RNF removed?**
They define the label: `Machine failure = 1` when at least one of them is 1.
Used as features they would give an almost perfect but meaningless result -
in practice the mode of a failure is known only after it happened. They are
used only in the error analysis, after the test.

**7. Why were UDI and Product ID removed?**
They are identifiers. Product ID contains the variant letter (already in
`Type`) and a serial number with no physical meaning; a model could memorise
it.

**8. Are the derived features not leakage?**
No. Temperature difference, power and strain are computed only from the six
inputs that are known before the run. They encode domain knowledge (heat
dissipation margin, mechanical power, tool load). They are computed inside the
Pipeline, so the app passes raw values and the same code runs in training and
prediction.

**9. How was leakage during scaling avoided?**
The stratified 80/20 split is the first operation after the quality checks.
Scaling, one-hot encoding and derived features are Pipeline steps, so in
cross-validation they are fitted on the training part of each fold only. A
test (`test_no_leakage.py`) checks that the scaler mean equals the mean of the
training data and that shuffling identifiers and flags does not change any
prediction.

**10. Why is the EDA done on the training set?**
Looking at the test set during EDA would let its patterns influence feature
and model decisions. Only the quality checks (missing values, duplicates,
label consistency) use the full file, because they are about the file itself.

**11. Why stratified split and stratified folds?**
With 339 failures a random split could put noticeably more or fewer failures
in a fold. Stratification keeps the 3.4% rate in every part (train 271 of
8000, test 68 of 2000).

**12. Why does the label stay unchanged despite the discrepancies with UCI?**
The file has 46 TWF flags instead of 120, 19 RNF instead of 5, 9 failures with
no flag and 18 RNF flags without failure. The label is what we are asked to
predict; correcting it would mean inventing data. The discrepancies are
described in the report and the 9 unflagged failures are tracked in the error
analysis.

**13. Why is the data file checked with SHA-256?**
To guarantee that every result was computed on the original UCI file. The
file is stored with `-text` in `.gitattributes`, because a line-ending
conversion would change the checksum.

## Models and validation

**14. Why these models?**
A baseline (shows the accuracy trap), logistic regression (simple, linear,
interpretable), a decision tree (interpretable rules), a random forest and
gradient boosting (strong for tabular data). Each on two feature sets, so the
effect of the derived features can be measured.

**15. Why the same folds for all models?**
So that differences between models are not caused by different splits.
`StratifiedKFold(5, shuffle=True, random_state=42)` is created by one function.

**16. Why nested cross-validation for the threshold?**
The threshold is itself learned from data. To estimate recall and false alarms
at the tuned threshold honestly, each outer fold tunes the threshold on its own
inner folds and is scored on data it has not seen.

**17. Why small grids?**
The dataset is small (8000 training rows) and the goal is a fair comparison,
not squeezing the last decimal. Large grids would also increase the risk of
overfitting the validation folds.

**18. Why the random forest and not gradient boosting, which has a higher
PR-AUC?**
The PR-AUC difference (0.892 vs 0.898) is a quarter of one standard deviation.
At the cost-based threshold the forest has a lower cost (56.9 vs 59.0 per 1000
runs), the same recall (0.85 vs 0.86), half the false alarms (5.6 vs 10.2 per
1000) and more stable results between folds. It is slower (about 60 ms per
prediction instead of 7 ms), which does not matter for interactive use.

**19. Was the test set really used once?**
For the decision, yes: the model and threshold were chosen only from the
cross-validated tables. During development the training script was once run
end to end with gradient boosting as a placeholder choice and printed its test
metrics (PR-AUC 0.903, recall 0.853). The final choice (random forest) was made
from CV and was not changed after the test, even though gradient boosting
happened to look slightly better on the test set. This is described in the
report.

**20. Why did the test PR-AUC (0.871) come out lower than CV (0.892)?**
The test set has only 68 failures, so one or two cases change PR-AUC
noticeably; 0.871 is within one standard deviation of the CV estimate
(0.025). The cost on the test set (62.5 per 1000) is also within one standard
deviation of the CV estimate (56.9 +/- 11.6).

## Errors and explainability

**21. What will the model not detect and why?**
Tool wear failures (TWF, 0 of 9 in the test set) and failures without any flag
(0 of 2). TWF happens at a random moment in the wear window, so the inputs only
say "the tool is old". The unflagged failures are not explained by anything in
the data. Failures from the deterministic rules (HDF, PWF, OSF) are found in
56 of 57 cases. The ceiling for recall from these inputs is about 84% (the
share of rule-explained failures); the model reaches 82.4%.

**22. What are the false alarms?**
All 5 are near misses at rule boundaries: temperature difference of 8.6 K at
low speed, strain just below 11 000 minNm, power close to 9000 W. Exactly the
cases an engineer would want to look at.

**23. What did the decision tree learn?**
Without seeing the rules it split at 1380 rpm and about 8.6 K (HDF), 3496 W
and 8995 W (PWF: 3500 and 9000 W) and 11 014 minNm (OSF limit for type L). The
model rediscovered the mechanism that generated the data.

**24. How is permutation importance computed and why on the test set?**
Each input column of the test set is shuffled 10 times and the drop of PR-AUC
is measured. On the test set it describes the model's behaviour on unseen data.
The pipeline takes raw inputs, so derived features are recomputed from the
shuffled columns. Torque and speed matter most, product type least.

**25. Why not SHAP?**
Outside the project scope; permutation importance and tree rules answer the
question "what does the model use" with standard scikit-learn tools.

## Application and database

**26. Why do we check the checksum before loading the model?**
`joblib` files are pickles: loading one can execute arbitrary code. The app
loads only our own file and first compares its SHA-256 with the model card; a
modified file is rejected (`test_model_io.py`).

**27. What is the difference between hard and soft validation?**
Hard validation (Pydantic) rejects impossible input: type outside L/M/H,
non-finite numbers, temperatures and speed <= 0, negative torque or wear.
Soft validation flags values outside the training ranges from the model card:
the prediction is made with a warning, because a tree ensemble cannot
extrapolate.

**28. Why one shared core for Streamlit and the console?**
`ml/predictor.py` does validation, prediction and logging once; both
interfaces only collect input and display the result, so they cannot diverge.

**29. How is SQL injection prevented?**
Only parameterised queries with `?`. A test stores the text
`M'); DROP TABLE predictions; --` and checks that it is saved as plain text
and the table still exists.

**30. Why is a connection opened per operation?**
Streamlit reruns the script on every interaction and may use several threads;
SQLite connections should not be shared between threads. The model, in
contrast, is loaded once with `st.cache_resource`.

**31. What does the database store and why no user data?**
Time (UTC, ISO 8601), model id, source, inputs, probability, threshold,
decision, out-of-range flag and latency. No user name or IP: they are not
needed for the purpose (auditing predictions), in line with data minimisation.

**32. Why store the model id?**
After retraining, old predictions can still be attributed to the model that
produced them.

## Data in general

**33. How do synthetic data differ from real data?**
Here failures follow a few simple, known rules with sharp boundaries; real
machines degrade gradually, sensors are noisy and drift, failure causes
interact and are not known in advance, and labels are often incomplete. A
model that is excellent here would need real data, time series and
monitoring before use in a plant.

**34. What would be the next steps?**
Real data with sensor time series, classification of the failure type, storing
the actual outcome of each run in the database to monitor model quality, and
agreeing the cost ratio with maintenance.
