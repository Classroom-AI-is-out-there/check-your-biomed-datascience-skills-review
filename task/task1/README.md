# Task 1: tabular supervised classification

Train a classification model using the public training data and generate predictions for the held-out feature rows.

## Output contract

- Input file: `dataset/task1/test_features.csv`
- Required output file: `outputs/task1_predictions.csv`
- Required columns: `patient_id`, `prediction`
- The output must contain exactly one row for every row in `test_features.csv`, with matching patient IDs and no missing values.
- The submitted solution must be `your-sollution/task1_solution.py` or `your-sollution/task1_solution.ipynb`.
