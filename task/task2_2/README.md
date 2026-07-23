# Task 2.2: unsupervised PPG activity clustering

Discover activity clusters from the public PPG training windows without using labels during feature extraction or clustering, then assign clusters to the held-out windows.

## Output contract

- Input file: `dataset/task2_2/test_features.npz`
- Required output file: `outputs/task2_2_predictions.csv`
- Required columns: `window_id`, `prediction`
- `prediction` is the integer cluster assignment; cluster numbers do not need to match the activity class numbers.
- The output must contain exactly one row for every window in `test_features.npz`, with matching window IDs and no missing values.
- The submitted solution must be `your-sollution/task2_2_solution.py` or `your-sollution/task2_2_solution.ipynb`.
