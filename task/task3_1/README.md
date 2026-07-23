# Task 3.1: supervised histopathology image classification

Train a nine-class CNN on the public PathMNIST-derived images and predict the tissue class for every held-out image.

## Output contract

- Input file: `dataset/task3_1/test_features.npz`
- Required output file: `outputs/task3_1_predictions.csv`
- Required columns: `image_id`, `prediction`
- The `images` array contains 28×28 RGB `uint8` images; `prediction` uses integer class IDs `0` through `8`.
- The output must contain exactly one row for every image in `test_features.npz`, with matching image IDs and no missing values.
- The submitted solution must be `your-sollution/task3_1_solution.py` or `your-sollution/task3_1_solution.ipynb`.
