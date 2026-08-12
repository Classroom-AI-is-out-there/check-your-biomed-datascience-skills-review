"""Prediction artifact paths for local and automated execution."""
import os
from pathlib import Path

def get_output_path(task_id: str) -> Path:
    """Return the grader override or the task's local output path, creating its parent."""
    default = Path("outputs") / f"{task_id}_predictions.csv"
    output_path = Path(os.environ.get("BIOMED_OUTPUT_PATH", default))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
