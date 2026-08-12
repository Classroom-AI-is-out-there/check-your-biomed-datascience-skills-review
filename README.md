# Biomedical data-science coding self-assessment

> **One template repository · one GitHub Classroom assignment · one invitation link · choose exactly one of seven tasks.**

## What this repository teaches

This beginner-friendly course exercise teaches you to inspect biomedical data, write Python/Jupyter code, train classification or clustering models, produce a strict prediction table, validate it locally, submit with Git, and interpret automated model-quality feedback. It is a transparent, reproducible self-assessment—not a secure exam.

## Task catalog

The values below come from each task's canonical `config.yaml`.

| ID | Modality | Learning | Difficulty | Metric | Target | Output columns |
|---|---|---|---|---|---:|---|
| `task1` | clinical table | classification | beginner | weighted F1 | 0.75 | `patient_id,prediction` |
| `task2_1` | ECG | classification | intermediate | macro F1 | 0.40 | `heartbeat_id,prediction` |
| `task2_2` | PPG | clustering | intermediate | ARI | 0.05 | `window_id,prediction` |
| `task3_1` | histopathology image | classification | intermediate | macro F1 | 0.35 | `image_id,prediction` |
| `task3_2` | histopathology image | clustering | advanced | ARI | 0.12 | `image_id,prediction` |
| `task4_1` | population genetics | classification | advanced | macro F1 | 0.10 | `sample_id,prediction` |
| `task4_2` | protein sequence | clustering | advanced | ARI | 0.35 | `sequence_id,cluster` |

All public datasets and task descriptions remain in this repository. Follow the selected [`task/<task_id>/README.md`](task/) for its exact contract.

## Start here

### 1. Accept and clone

Open the single invitation link supplied by your instructor, accept the assignment, then copy the generated repository URL:

```bash
git clone <your-classroom-repository-url>
cd check-your-biomed-datascience-skills-review
```

The instructor configures the assignment name, deadline, visibility, starter repository, Feedback Pull Request, and invitation link in Classroom. The tracked workflow is the only autograding definition; no duplicate Classroom-interface tests should be configured.

### 2. Prerequisites and environment

Install Git and Python 3.10+. Create an isolated environment (Windows activation: `.venv\Scripts\activate`):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The grader installs the selected task requirements as needed. To work interactively, run `jupyter notebook` from the repository root.

### 3. Choose and initialize exactly one task

Read the catalog and selected task README, then copy its genuinely incomplete scaffold:

```bash
python tools/start_task.py task2_1
```

This creates only `your_solution/task2_1_solution.ipynb`. Edit that file only. You may instead submit a correctly named `.py` script. Do not copy multiple templates.

### 4. Build and run

Load the documented training and test-feature paths, set a random seed, preprocess without test leakage, validate your approach, train, and predict every test row. Run a script from the root with `python your_solution/task2_1_solution.py`; run a notebook top-to-bottom with Jupyter. Your final cell writes via `get_output_path`.

A prediction CSV is a plain table connecting every test example's stable ID to your model's prediction. IDs let grading compare rows safely even when order changes. Exact universal rules: the two documented columns only; every test ID exactly once; no missing/extra/duplicate IDs; the expected row count; allowed class or valid cluster labels; no NaN/infinity; save with `index=False`. Task-specific examples and types are in each task README.

### 5. Check locally

```bash
python grader/run_grading.py --task task2_1 --sanity-only
python grader/run_grading.py --task task2_1
python grader/run_grading.py --task task2_1 --report-json grading-report.json
```

Sanity mode installs/confirms dependencies, executes once, and validates output. Complete mode performs that same single execution, then calculates metrics and points. Set a seed (commonly 42), record important choices, and ensure a clean restart gives comparable results.

## What do I submit?

Submit **exactly one solution source file**:

```
your_solution/<task_id>_solution.py
```

or:

```
your_solution/<task_id>_solution.ipynb
```

For Task 2.1, `your_solution/task2_1_solution.ipynb` is accepted. It must run from the repository root and generate the CSV described by that task. Do **not** commit generated predictions, reports, figures, executed notebooks, environments, or model checkpoints; `.gitignore` excludes them.

```bash
python grader/run_grading.py --task task2_1 --sanity-only
git status
git add your_solution/task2_1_solution.ipynb
git commit -m "Complete Task 2.1"
git push
```

After pushing, open **Actions** or **Checks** and the optional Classroom Feedback PR.

## What happens after `git push`?

```
Accept the Classroom assignment
              ↓
       Clone repository
              ↓
       Choose one task
              ↓
      Initialize scaffold
              ↓
    Complete one solution
              ↓
   Run the solution locally
              ↓
   Generate prediction CSV
              ↓
    Run local sanity check
              ↓
   Commit the solution source
              ↓
          git push
              ↓
  GitHub Actions detects task
              ↓
  Solution executes exactly once
              ↓
   Output format is validated
              ↓
   Metric and score are calculated
              ↓
   Student reads the feedback
```

Actions caches dependencies, detects the one filename, installs selected dependencies, gives the solution a fresh temporary `BIOMED_OUTPUT_PATH`, executes once, validates that exact artifact, scores it, writes a Markdown summary, and uploads JSON. A stale local output therefore cannot pass. Errors distinguish detection, installation, execution, format, and quality stages.

## Metrics, points, and feedback

Execution is 10 points and valid output is 10. Quality earns 0 points below half the configured target, 20 at half, 50 at 75%, and 80 at the target. Classification uses configured F1 averaging. Clustering shows ARI and NMI even when ARI controls bands. Low quality invites iteration; malformed output is a separate issue. Find details in the Actions log, job summary, `grading-report` artifact, Classroom check, and Feedback PR.

## Troubleshooting (meaning → cause → inspect → rerun)

In every row, inspect your active `your_solution/<task_id>_solution.*` unless another file is named.

| Problem | Meaning and likely cause | Inspect / corrective action | Rerun |
|---|---|---|---|
| No solution detected | No accepted active filename exists. | `your_solution/`; run `python tools/start_task.py <task_id>`. | `python grader/run_grading.py --detect-task` |
| Multiple solutions | Two tasks or both extensions exist. | `your_solution/`; remove all but one. | detection command above |
| Incorrect filename | Name is outside the configured convention. | Rename it to `<task_id>_solution.py` or `.ipynb`. | detection command |
| Notebook not top-to-bottom | Hidden state or an earlier cell fails. | Restart kernel, Run All, fix first failing cell. | task sanity command |
| Wrong working directory | Script relies on its file directory. | Use repository-relative documented paths and launch at root. | task sanity command |
| Missing dependency | Import is not installed. | Add it to the selected task `requirements.txt` (or avoid it), then install that file. | `pip install -r task/<task_id>/requirements.txt` |
| Missing CSV | Code never saved or stopped first. | Final save cell; use `get_output_path`. | task sanity command |
| CSV in wrong directory | A path was hard-coded. | Replace it with `get_output_path("<task_id>")`. | task sanity command |
| Wrong columns | Headers differ in spelling/case. | Task README contract; construct exact names. | task sanity command |
| Extra Pandas index | `to_csv` wrote `Unnamed: 0`. | Use `to_csv(output_path, index=False)`. | task sanity command |
| Wrong row count | Rows were dropped/duplicated. | Prediction construction; predict every test example. | task sanity command |
| Missing IDs | ID extraction/alignment failed. | Test-feature loader and ID column. | task sanity command |
| Duplicate IDs | Merge or batching repeated examples. | Check `submission[id].duplicated()`. | task sanity command |
| IDs mismatch | IDs were renumbered or sourced from train. | Copy test IDs unchanged. | task sanity command |
| NaN/infinite prediction | Preprocessing/model created non-finite output. | Check `np.isfinite` before saving. | task sanity command |
| Invalid class label | Labels fall outside configured allowed values. | Task README/config and label encoding. | task sanity command |
| One cluster | Clustering collapsed. | Features, scaling, hyperparameters, cluster count. | full task command |
| Execution timeout | Code exceeded the configured limit. | Config timeout and slow model/loops; simplify or vectorize. | task sanity command |
| Metric below target | Artifact is valid but quality is low. | Validation design/features/model; avoid grading labels. | full task command |
| Workflow did not start | Push/Actions/default workflow issue. | Confirm commit is remote and Actions enabled; manually dispatch `classroom.yml`. | `git push` |

Replace `<task_id>` with your choice, for example `task2_1`.

## Transparency, reproducibility, and academic integrity

Grader code and reference labels are visible deliberately: this teaches how validation and metrics work. For an honest, meaningful self-assessment, do not read or train against `grading/` labels while developing. Explain your method and seed, and commit code—not generated artifacts.

## Краткая инструкция на русском

Это один общий шаблон и одно задание GitHub Classroom: примите одну ссылку, клонируйте репозиторий, выберите **одно** из семи заданий, выполните `python tools/start_task.py task2_1`, редактируйте только созданный файл в `your_solution/`, запустите sanity/full-команды выше и отправьте только исходник через `git add`, `git commit`, `git push`. CSV должен содержать ровно указанные в README задания столбцы, каждый тестовый ID один раз и конечное предсказание; используйте `get_output_path` и `index=False`. После push смотрите Actions/Checks, JSON-артефакт и Feedback PR. Проверяющий код открыт; не используйте метки из `grading/` при разработке, если хотите честную самооценку.
