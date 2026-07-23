#!/usr/bin/env python3
"""Execute and grade a task solution from the repository root."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def load_config(task_id: str) -> dict:
    config_path = REPO_ROOT / "task" / task_id / "config.yaml"
    if not config_path.is_file():
        fail(f"CONFIG_ERROR: missing configuration: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("task_id") != task_id:
        fail(f"CONFIG_ERROR: invalid configuration for {task_id}")
    return config


def detect_task() -> str:
    """Return the one configured task for which a solution was submitted."""
    configured_tasks = []
    for config_path in sorted((REPO_ROOT / "task").glob("*/config.yaml")):
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if isinstance(config, dict) and config.get("task_id"):
            configured_tasks.append(str(config["task_id"]))

    submitted = []
    for task_id in configured_tasks:
        base = REPO_ROOT / "your-sollution" / f"{task_id}_solution"
        for suffix in (".py", ".ipynb"):
            solution = base.with_suffix(suffix)
            if solution.is_file():
                submitted.append((task_id, solution))

    if not submitted:
        expected = ", ".join(f"{task_id}_solution.py/.ipynb" for task_id in configured_tasks)
        fail(
            "DETECTION_ERROR: no configured task solution found in your-sollution/. "
            f"Expected one of: {expected or 'no tasks are configured'}"
        )
    if len(submitted) > 1:
        names = ", ".join(str(path.relative_to(REPO_ROOT)) for _, path in submitted)
        fail(f"DETECTION_ERROR: submit exactly one solution; found: {names}")
    return submitted[0][0]


def find_solution(task_id: str) -> Path:
    base = REPO_ROOT / "your-sollution" / f"{task_id}_solution"
    candidates = [base.with_suffix(".py"), base.with_suffix(".ipynb")]
    found = [path for path in candidates if path.is_file()]
    if not found:
        fail(
            "EXECUTION_ERROR: solution file not found; expected "
            f"{candidates[0].relative_to(REPO_ROOT)} or "
            f"{candidates[1].relative_to(REPO_ROOT)}",
            2,
        )
    if len(found) > 1:
        fail("EXECUTION_ERROR: submit only one Task solution (.py or .ipynb)", 2)
    return found[0]


def install_task_requirements(task_id: str, timeout: int) -> None:
    requirements = REPO_ROOT / "task" / task_id / "requirements.txt"
    if not requirements.is_file():
        return
    print(f"Installing task dependencies from {requirements.relative_to(REPO_ROOT)}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        fail(f"EXECUTION_ERROR: task dependency installation failed: {error}", 2)
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        fail(f"EXECUTION_ERROR: task dependency installation failed: {details}", 2)


def execute_solution(solution: Path, timeout: int) -> None:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    if solution.suffix == ".py":
        command = [sys.executable, str(solution)]
        temporary_directory = None
    else:
        temporary_directory = tempfile.TemporaryDirectory(prefix="grading-notebook-")
        environment["JUPYTER_CONFIG_DIR"] = temporary_directory.name
        command = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(solution),
            "--output",
            "executed.ipynb",
            "--output-dir",
            temporary_directory.name,
            f"--ExecutePreprocessor.timeout={timeout}",
        ]

    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        if temporary_directory is not None:
            temporary_directory.cleanup()
        fail(f"EXECUTION_ERROR: {error}", 2)

    if temporary_directory is not None:
        temporary_directory.cleanup()
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        details = result.stderr.strip() or f"process exited with code {result.returncode}"
        fail(f"EXECUTION_ERROR: {details}", 2)


def validate_predictions(task_id: str, config: dict) -> pd.DataFrame:
    id_column = config["id_column"]
    target_column = config["target_column"]
    feature_candidates = sorted(
        (REPO_ROOT / "dataset" / task_id).glob("test_features.*")
    )
    if len(feature_candidates) != 1:
        fail(
            "CONFIG_ERROR: expected exactly one test_features file, found "
            f"{len(feature_candidates)}"
        )
    features_path = feature_candidates[0]
    predictions_path = REPO_ROOT / "outputs" / f"{task_id}_predictions.csv"

    if not predictions_path.is_file():
        fail(f"VALIDATION_ERROR: missing output file: {predictions_path}")

    if features_path.suffix == ".csv":
        expected_ids = pd.read_csv(features_path, usecols=[id_column])[id_column]
    elif features_path.suffix == ".npz":
        with np.load(features_path, allow_pickle=False) as features:
            if id_column not in features.files:
                fail(f"CONFIG_ERROR: {features_path} has no {id_column} array")
            expected_ids = pd.Series(features[id_column])
    elif features_path.suffix == ".txt":
        with features_path.open(encoding="utf-8") as stream:
            expected_ids = pd.Series(
                line.split(maxsplit=1)[0] for line in stream if line.strip()
            )
    else:
        fail(f"CONFIG_ERROR: unsupported test feature format: {features_path.suffix}")

    predictions = pd.read_csv(predictions_path)
    required = [id_column, target_column]
    missing_columns = [column for column in required if column not in predictions.columns]
    if missing_columns:
        fail(f"VALIDATION_ERROR: missing required columns: {missing_columns}")
    if len(predictions) != len(expected_ids):
        fail(
            "VALIDATION_ERROR: row count mismatch: "
            f"expected {len(expected_ids)}, got {len(predictions)}"
        )
    if predictions[required].isna().any().any():
        fail("VALIDATION_ERROR: predictions contain missing values")
    if predictions[id_column].duplicated().any():
        fail("VALIDATION_ERROR: prediction IDs must be unique")
    expected_ids = set(expected_ids)
    predicted_ids = set(predictions[id_column])
    if predicted_ids != expected_ids:
        fail("VALIDATION_ERROR: prediction IDs do not match test feature IDs")
    return predictions[required]


def grade(task_id: str, config: dict, predictions: pd.DataFrame) -> None:
    id_column = config["id_column"]
    target_column = config["target_column"]
    labels_path = REPO_ROOT / "grading" / task_id / "test_labels.csv"
    if not labels_path.is_file():
        fail(f"GRADING_ERROR: missing held-out labels: {labels_path}")
    labels = pd.read_csv(labels_path)

    if labels[id_column].duplicated().any():
        fail("GRADING_ERROR: label IDs must be unique")
    evaluated = labels.merge(
        predictions,
        on=id_column,
        how="left",
        validate="one_to_one",
        suffixes=("_true", "_pred"),
    )
    predicted_column = f"{target_column}_pred"
    true_column = f"{target_column}_true"
    if evaluated[predicted_column].isna().any():
        count = int(evaluated[predicted_column].isna().sum())
        fail(f"VALIDATION_ERROR: {count} held-out rows have no prediction")

    metric = str(config.get("metric", "")).lower()
    task_type = str(config.get("type", "")).lower()
    true_values = evaluated[true_column]
    predicted_values = evaluated[predicted_column]
    if metric == "f1" and task_type == "classification":
        average = config.get("average")
        score = f1_score(true_values, predicted_values, average=average)
        metric_label = f"F1 score ({average})"
    elif metric == "ari" and task_type == "clustering":
        score = adjusted_rand_score(true_values, predicted_values)
        metric_label = "Adjusted Rand index"
        nmi = normalized_mutual_info_score(true_values, predicted_values)
        print(f"Normalized mutual information (visibility only): {nmi:.6f}")
    elif metric == "nmi" and task_type == "clustering":
        score = normalized_mutual_info_score(true_values, predicted_values)
        metric_label = "Normalized mutual information"
    else:
        fail(f"CONFIG_ERROR: unsupported type/metric combination: {task_type}/{metric}")

    threshold = float(config["threshold"])
    print(f"{metric_label}: {score:.6f}")
    print(f"Threshold: {threshold:.6f}")
    if score < threshold:
        fail(f"METRIC_FAILED: {score:.6f} < {threshold:.6f}")
    print(f"METRIC_PASSED: {score:.6f} >= {threshold:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", help="Task ID from task/<id>/config.yaml")
    mode.add_argument("--detect-task", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()

    if args.detect_task:
        print(detect_task())
        return

    config = load_config(args.task)
    install_task_requirements(args.task, int(config["timeout_seconds"]))
    execute_solution(find_solution(args.task), int(config["timeout_seconds"]))
    predictions = validate_predictions(args.task, config)
    if args.sanity_only:
        print(f"SANITY_PASSED: validated {len(predictions)} predictions")
        return
    grade(args.task, config, predictions)


if __name__ == "__main__":
    main()
