# Task 3.2: unsupervised histopathology image clustering

Cluster the public PathMNIST-derived images without using their tissue labels during fitting, then assign clusters to every held-out image.

## Output contract

- Input file: `dataset/task3_2/test_features.npz`
- Required output file: `outputs/task3_2_predictions.csv`
- Required columns: `image_id`, `prediction`
- The `images` array contains 28×28 RGB `uint8` images; `prediction` is an integer cluster assignment whose numbering need not match class IDs.
- The output must contain exactly one row for every image in `test_features.npz`, with matching image IDs and no missing values.
- The submitted solution must be `your-sollution/task3_2_solution.py` or `your-sollution/task3_2_solution.ipynb`.
