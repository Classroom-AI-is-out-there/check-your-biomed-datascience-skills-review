# Task 4.1: supervised genetic-population classification

Train a graph classifier from the labelled public genetic sequences and predict the population for every held-out individual.

## Output contract

- Input file: `dataset/task4_1/test_features.txt`
- Required output file: `outputs/task4_1_predictions.csv`
- Required columns: `sample_id`, `prediction`
- Each input row contains `sample_id`, gender, then 10,101 nucleotide values; the population field is intentionally absent.
- The output must contain exactly one row for every row in `test_features.txt`, with matching sample IDs and no missing values.
- The submitted solution must be `your-sollution/task4_1_solution.py` or `your-sollution/task4_1_solution.ipynb`.
