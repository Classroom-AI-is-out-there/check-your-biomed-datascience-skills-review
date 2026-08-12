#!/usr/bin/env python3
"""Beginner-friendly, single-execution grader for the biomedical tasks."""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = REPO_ROOT / "your_solution"

class GradingError(Exception):
    """An actionable error which is safe to show to a student."""

def task_ids() -> list[str]:
    return [p.parent.name for p in sorted((REPO_ROOT / "task").glob("*/config.yaml"))]

def load_config(task_id: str) -> dict:
    path = REPO_ROOT / "task" / task_id / "config.yaml"
    if not path.is_file():
        raise GradingError(f"CONFIG_ERROR [{task_id}]: configuration does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("task_id") != task_id:
        raise GradingError(f"CONFIG_ERROR [{task_id}]: task_id is missing or inconsistent in {path}")
    if "timeout_seconds" in config:
        warnings.warn("timeout_seconds is deprecated; use install_timeout_seconds and execution_timeout_seconds", DeprecationWarning)
        config.setdefault("install_timeout_seconds", config["timeout_seconds"])
        config.setdefault("execution_timeout_seconds", config["timeout_seconds"])
    required = {"type", "metric", "id_column", "target_column", "threshold", "install_timeout_seconds", "execution_timeout_seconds"}
    missing = sorted(required - config.keys())
    if missing: raise GradingError(f"CONFIG_ERROR [{task_id}]: missing keys in {path}: {missing}")
    return config

def submitted_solutions() -> list[tuple[str, Path]]:
    found=[]
    for task_id in task_ids():
        for suffix in (".py", ".ipynb"):
            path=SUBMISSION_DIR / f"{task_id}_solution{suffix}"
            if path.is_file(): found.append((task_id,path))
    return found

def detect_task() -> tuple[str, Path]:
    found=submitted_solutions()
    if not found:
        raise GradingError("DETECTION_ERROR: No active solution was found.\n\nChoose one task and run:\n  python tools/start_task.py task2_1\n\nExpected: your_solution/<task_id>_solution.py or .ipynb")
    if len(found)!=1:
        names="\n  ".join(str(p.relative_to(REPO_ROOT)) for _,p in found)
        raise GradingError(f"DETECTION_ERROR: Submit exactly one solution; found {len(found)} files:\n  {names}\nRemove the extra file(s) from your_solution/ and retry.")
    return found[0]

def find_solution(task_id: str) -> Path:
    found=[p for tid,p in submitted_solutions() if tid==task_id]
    if len(found)!=1:
        listing=", ".join(str(p.relative_to(REPO_ROOT)) for p in found) or "none"
        raise GradingError(f"EXECUTION_ERROR [{task_id}]: expected exactly one .py or .ipynb solution; found: {listing}")
    return found[0]

def install_requirements(task_id: str, timeout: int, skip: bool=False) -> None:
    if skip: return
    paths=[REPO_ROOT/"requirements.txt", REPO_ROOT/"task"/task_id/"requirements.txt"]
    command=[sys.executable,"-m","pip","install"] + [x for p in paths if p.is_file() for x in ("-r",str(p))]
    try: result=subprocess.run(command,cwd=REPO_ROOT,text=True,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired: raise GradingError(f"INSTALLATION_ERROR [{task_id}]: dependency installation exceeded {timeout} seconds")
    except OSError as e: raise GradingError(f"INSTALLATION_ERROR [{task_id}]: could not start pip: {e}")
    if result.returncode: raise GradingError(f"INSTALLATION_ERROR [{task_id}]: pip failed. Check task/{task_id}/requirements.txt.\n{result.stderr[-2000:]}")

def execute(solution: Path, task_id: str, output: Path, timeout: int) -> float:
    env={**os.environ,"MPLBACKEND":"Agg","BIOMED_OUTPUT_PATH":str(output)}
    # A script launched by filename otherwise places only your_solution/ on
    # sys.path.  Keep repository helpers importable exactly as they are in a
    # notebook launched from the repository root.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    started=time.monotonic()
    if solution.suffix==".py": command=[sys.executable,str(solution)]; notebook_tmp=None
    else:
        notebook_tmp=tempfile.TemporaryDirectory(prefix="biomed-notebook-")
        env["JUPYTER_CONFIG_DIR"]=notebook_tmp.name
        command=["jupyter","nbconvert","--to","notebook","--execute",str(solution),"--output","executed.ipynb","--output-dir",notebook_tmp.name,f"--ExecutePreprocessor.timeout={timeout}"]
    try: result=subprocess.run(command,cwd=REPO_ROOT,text=True,capture_output=True,timeout=timeout,env=env)
    except subprocess.TimeoutExpired: raise GradingError(f"EXECUTION_ERROR [{task_id}]: solution exceeded {timeout} seconds")
    except OSError as e: raise GradingError(f"EXECUTION_ERROR [{task_id}]: could not start solution: {e}")
    finally:
        if notebook_tmp: notebook_tmp.cleanup()
    if result.returncode: raise GradingError(f"EXECUTION_ERROR [{task_id}] in {solution.relative_to(REPO_ROOT)}:\n{(result.stderr or result.stdout)[-3000:]}")
    return round(time.monotonic()-started,3)

def expected_ids(task_id: str, config: dict) -> pd.Series:
    path=REPO_ROOT/config["test_features"] ; key=config["id_column"]
    if path.suffix==".csv": return pd.read_csv(path,usecols=[key])[key]
    if path.suffix==".npz":
        with np.load(path,allow_pickle=False) as data: return pd.Series(data[key])
    if path.suffix==".txt": return pd.Series([x.split(maxsplit=1)[0] for x in path.read_text().splitlines() if x.strip()])
    if path.suffix in {".fasta",".fa",".faa"}: return pd.Series([x[1:].split(maxsplit=1)[0] for x in path.read_text().splitlines() if x.startswith(">")])
    raise GradingError(f"CONFIG_ERROR [{task_id}]: unsupported test-feature format: {path}")

def validate_predictions(task_id: str, config: dict, path: Path) -> tuple[pd.DataFrame,dict]:
    ids=expected_ids(task_id,config); required=[config["id_column"],config["target_column"]]
    if not path.is_file(): raise GradingError(f"VALIDATION_ERROR [{task_id}]: missing prediction file: {path}. Use get_output_path('{task_id}').")
    try: frame=pd.read_csv(path)
    except Exception as e: raise GradingError(f"VALIDATION_ERROR [{task_id}]: cannot parse CSV {path}: {e}")
    duplicates=frame.columns[frame.columns.duplicated()].tolist()
    if duplicates: raise GradingError(f"VALIDATION_ERROR [{task_id}]: duplicate CSV columns in {path}: {duplicates}")
    missing=[c for c in required if c not in frame]
    if missing: raise GradingError(f"VALIDATION_ERROR [{task_id}]: missing required columns in {path}: {missing}; expected {required}")
    extras=[c for c in frame if c not in required]
    if extras: raise GradingError(f"VALIDATION_ERROR [{task_id}]: extra columns are not allowed in {path}: {extras}")
    if len(frame)!=len(ids): raise GradingError(f"VALIDATION_ERROR [{task_id}]: row count for {path}; expected {len(ids)}, actual {len(frame)}")
    idc,target=required
    if frame[idc].isna().any(): raise GradingError(f"VALIDATION_ERROR [{task_id}]: {idc} contains missing values in {path}")
    if frame[target].isna().any(): raise GradingError(f"VALIDATION_ERROR [{task_id}]: {target} contains missing values in {path}")
    numeric=pd.to_numeric(frame[target],errors="coerce")
    if numeric.notna().any() and np.isinf(numeric.dropna()).any(): raise GradingError(f"VALIDATION_ERROR [{task_id}]: {target} contains infinite values in {path}")
    if frame[idc].duplicated().any(): raise GradingError(f"VALIDATION_ERROR [{task_id}]: {idc} must be unique in {path}; duplicates found")
    if set(map(str,frame[idc])) != set(map(str,ids)):
        raise GradingError(f"VALIDATION_ERROR [{task_id}]: IDs in {path} do not exactly match {config['test_features']} (no missing or extra IDs allowed)")
    if config["type"]=="classification" and "allowed_labels" in config:
        invalid=sorted(set(frame[target])-set(config["allowed_labels"]),key=str)
        if invalid: raise GradingError(f"VALIDATION_ERROR [{task_id}]: invalid classification labels in {path}: {invalid}; allowed {config['allowed_labels']}")
    if config["type"]=="clustering":
        if frame[target].map(lambda x: np.isscalar(x)).eq(False).any(): raise GradingError(f"VALIDATION_ERROR [{task_id}]: cluster labels must be scalar")
        count=frame[target].nunique()
        if count==1 and not config.get("allow_single_cluster",False): raise GradingError(f"VALIDATION_ERROR [{task_id}]: only one cluster was generated in {path}; create at least two")
        if count < config.get("min_clusters",2) or count > config.get("max_clusters",len(frame)):
            raise GradingError(f"VALIDATION_ERROR [{task_id}]: cluster count in {path}; expected {config.get('min_clusters',2)}..{config.get('max_clusters',len(frame))}, actual {count}")
    return frame[required],{"passed":True,"rows_expected":len(ids),"rows_received":len(frame)}

def metrics(task_id: str, config: dict, predictions: pd.DataFrame) -> dict:
    labels=pd.read_csv(REPO_ROOT/"grading"/task_id/"test_labels.csv")
    idc,target=config["id_column"],config["target_column"]
    joined=labels.merge(predictions,on=idc,validate="one_to_one",suffixes=("_true","_pred"))
    y,yp=joined[f"{target}_true"],joined[f"{target}_pred"]
    if config["type"]=="classification": return {f"f1_{config['average']}":float(f1_score(y,yp,average=config["average"]))}
    return {"ari":float(adjusted_rand_score(y,yp)),"nmi":float(normalized_mutual_info_score(y,yp))}

def assess(task_id: str, sanity: bool, report_path: Path|None, skip_install: bool) -> int:
    config=load_config(task_id); solution=find_solution(task_id)
    report={"task_id":task_id,"solution_path":str(solution.relative_to(REPO_ROOT)),"execution":{"passed":False},"validation":{"passed":False},"metrics":{},"thresholds":{},"score":{"earned":0,"maximum":100},"feedback":[]}
    try:
        install_requirements(task_id,int(config["install_timeout_seconds"]),skip_install)
        with tempfile.TemporaryDirectory(prefix="biomed-grading-") as directory:
            output=Path(directory)/f"{task_id}_predictions.csv"
            report["execution"]={"passed":True,"duration_seconds":execute(solution,task_id,output,int(config["execution_timeout_seconds"]))}; report["score"]["earned"]=10; report["feedback"].append("Solution execution passed.")
            predictions,validation=validate_predictions(task_id,config,output); report["validation"]=validation; report["score"]["earned"]=20; report["feedback"].append("Prediction format passed.")
            if not sanity:
                values=metrics(task_id,config,predictions); report["metrics"]=values
                key=f"f1_{config.get('average')}" if config["metric"]=="f1" else config["metric"]
                threshold=float(config["threshold"]); value=values[key]; report["thresholds"]={key:threshold}
                # Three quality bands, worth 20/50/80 metric points.
                earned=80 if value>=threshold else 50 if value>=threshold*.75 else 20 if value>=threshold*.5 else 0
                report["score"]["earned"]+=earned
                report["feedback"].append(f"{key}: {value:.4f}; target: {threshold:.4f}; quality points: {earned}/80.")
        print(f"TASK_ID={task_id}\nSOLUTION={report['solution_path']}\nSCORE={report['score']['earned']}/100")
        for item in report["feedback"]: print(f"- {item}")
    except GradingError as e:
        report["feedback"].append(str(e)); print(str(e),file=sys.stderr)
    finally:
        if report_path:
            report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return 0 if report["execution"]["passed"] and report["validation"]["passed"] and (sanity or report["metrics"]) else 1

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--task",choices=task_ids()); modes.add_argument("--detect-task",action="store_true")
    parser.add_argument("--sanity-only",action="store_true"); parser.add_argument("--report-json",type=Path); parser.add_argument("--skip-install",action="store_true",help="Use already-installed dependencies")
    args=parser.parse_args()
    try:
        if args.detect_task:
            task,path=detect_task(); print(f"TASK_ID={task}\nSOLUTION_PATH={path.relative_to(REPO_ROOT)}"); return
        raise SystemExit(assess(args.task,args.sanity_only,args.report_json,args.skip_install))
    except GradingError as e: print(str(e),file=sys.stderr); raise SystemExit(1)
if __name__=="__main__": main()
