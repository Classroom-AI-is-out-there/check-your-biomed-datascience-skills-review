# Task 4.2: clustering biomedical modeling

Build a reproducible **clustering** model using the public training set, then predict/cluster every held-out test example. Do not develop against `grading/` labels if you want meaningful self-assessment.

## Output contract

| Item | Required value |
|---|---|
| Training data | `dataset/task4_2/train.fasta` |
| Test-feature data | `dataset/task4_2/test_features.fasta` |
| Local output | `outputs/task4_2_predictions.csv` |
| Exact columns | `sequence_id`, `cluster` (no additional columns) |
| Rows | exactly 7, equal to the test examples |

- `sequence_id` identifies the test example. Preserve its source data type (values are `str`) and include every test ID exactly once.
- `cluster` is the model result: a finite scalar cluster label; use 2–20 clusters.
- Missing or extra IDs, duplicate IDs, missing values, and numeric positive/negative infinity are forbidden.
- Row order does not matter because grading joins by ID. Additional columns are not allowed.

```csv
sequence_id,cluster
seq_005,0
seq_007,1
seq_010,2
```

> **The example above demonstrates the required CSV format only. The prediction values are not supplied answers.**

Minimal construction and save pattern:

```python
import pandas as pd
from biomed_submission import get_output_path

submission = pd.DataFrame(
    {
        "sequence_id": test_ids,  # extract these from the test-feature data
        "cluster": predicted_labels,
    }
)
output_path = get_output_path("task4_2")
submission.to_csv(output_path, index=False)
```

`index=False` prevents Pandas from creating an unwanted CSV index column. The helper writes locally to `outputs/`, but honors the grader's temporary `BIOMED_OUTPUT_PATH`; never hard-code a different output path.

## Check your work

```bash
python grader/run_grading.py --task task4_2 --sanity-only
python grader/run_grading.py --task task4_2
```

The sanity check executes the solution once and checks the artifact; complete assessment additionally reports quality. Common messages are actionable: `missing prediction file` means the helper was not used, `missing required columns`/`extra columns` means the header is wrong (often a saved Pandas index), `row count` means predictions were dropped or added, `IDs ... do not exactly match` means IDs were changed, `invalid classification labels` means output is outside the allowed classes, and `only one cluster` means clustering collapsed. Inspect `your_solution/task4_2_solution.py` or `.ipynb`, fix it, and rerun the same command.

## Metric and scoring

The canonical configuration is [`config.yaml`](config.yaml). The main metric is **ARI**; its target is **0.35**. Clustering reports both ARI and NMI. Execution earns 10 points, valid format earns 10, and metric quality earns 0/20/50/80 points at below half / half / 75% / 100% of the target. A low metric is feedback—not an output-format failure.
