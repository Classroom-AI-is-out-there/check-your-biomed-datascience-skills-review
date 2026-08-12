# Task 2.1: classification biomedical modeling

Build a reproducible **classification** model using the public training set, then predict/cluster every held-out test example. Do not develop against `grading/` labels if you want meaningful self-assessment.

## Output contract

| Item | Required value |
|---|---|
| Training data | `dataset/task2_1/train.csv` |
| Test-feature data | `dataset/task2_1/test_features.csv` |
| Local output | `outputs/task2_1_predictions.csv` |
| Exact columns | `heartbeat_id`, `prediction` (no additional columns) |
| Rows | exactly 1000, equal to the test examples |

- `heartbeat_id` identifies the test example. Preserve its source data type (values are `int64`) and include every test ID exactly once.
- `prediction` is the model result: one of `[0, 1, 2, 3, 4]`.
- Missing or extra IDs, duplicate IDs, missing values, and numeric positive/negative infinity are forbidden.
- Row order does not matter because grading joins by ID. Additional columns are not allowed.

```csv
heartbeat_id,prediction
2,0
5,1
12,2
```

> **The example above demonstrates the required CSV format only. The prediction values are not supplied answers.**

Minimal construction and save pattern:

```python
import pandas as pd
from biomed_submission import get_output_path

submission = pd.DataFrame(
    {
        "heartbeat_id": test_ids,  # extract these from the test-feature data
        "prediction": predicted_labels,
    }
)
output_path = get_output_path("task2_1")
submission.to_csv(output_path, index=False)
```

`index=False` prevents Pandas from creating an unwanted CSV index column. The helper writes locally to `outputs/`, but honors the grader's temporary `BIOMED_OUTPUT_PATH`; never hard-code a different output path.

## Check your work

```bash
python grader/run_grading.py --task task2_1 --sanity-only
python grader/run_grading.py --task task2_1
```

The sanity check executes the solution once and checks the artifact; complete assessment additionally reports quality. Common messages are actionable: `missing prediction file` means the helper was not used, `missing required columns`/`extra columns` means the header is wrong (often a saved Pandas index), `row count` means predictions were dropped or added, `IDs ... do not exactly match` means IDs were changed, `invalid classification labels` means output is outside the allowed classes, and `only one cluster` means clustering collapsed. Inspect `your_solution/task2_1_solution.py` or `.ipynb`, fix it, and rerun the same command.

## Metric and scoring

The canonical configuration is [`config.yaml`](config.yaml). The main metric is **F1** with `macro` averaging; its target is **0.4**. Clustering reports both ARI and NMI. Execution earns 10 points, valid format earns 10, and metric quality earns 0/20/50/80 points at below half / half / 75% / 100% of the target. A low metric is feedback—not an output-format failure.
