# Task 2.1: supervised ECG heartbeat classification

Train a classifier on the public labelled ECG beats and predict the rhythm class for each held-out heartbeat.

## Output contract

- Input file: `dataset/task2_1/test_features.csv`
- Required output file: `outputs/task2_1_predictions.csv`
- Required columns: `heartbeat_id`, `prediction`
- `prediction` uses the integer class IDs `0` through `4` shown in `train.csv`.
- The output must contain exactly one row for every row in `test_features.csv`, with matching heartbeat IDs and no missing values.
- The submitted solution must be `your-sollution/task2_1_solution.py` or `your-sollution/task2_1_solution.ipynb`.
