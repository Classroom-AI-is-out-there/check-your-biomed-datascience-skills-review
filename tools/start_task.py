#!/usr/bin/env python3
"""Copy one empty task scaffold into the active submission directory."""
import argparse, shutil, sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def active(root=ROOT):
 return sorted(p for p in (root/'your_solution').glob('*_solution.*') if p.suffix in {'.py','.ipynb'})
def initialize(task_id: str, force: bool=False, root: Path=ROOT) -> tuple[Path,dict]:
 config_path=root/'task'/task_id/'config.yaml'
 if not config_path.is_file(): raise ValueError(f"unknown task ID {task_id!r}")
 config=yaml.safe_load(config_path.read_text()); source=root/'solution_templates'/f"{task_id}_solution.ipynb"; destination=root/'your_solution'/source.name
 others=[p for p in active(root) if p!=destination]
 if others: raise FileExistsError('another active solution exists: '+', '.join(str(p.relative_to(root)) for p in others))
 if destination.exists() and not force: raise FileExistsError(f'{destination.relative_to(root)} already exists; use --force to replace it')
 shutil.copyfile(source,destination); return destination,config
def main():
 parser=argparse.ArgumentParser(); parser.add_argument('task_id'); parser.add_argument('--force',action='store_true'); args=parser.parse_args()
 try: destination,config=initialize(args.task_id,args.force)
 except ValueError: parser.error(f"unknown task ID {args.task_id!r}; choose: {', '.join(p.parent.name for p in sorted((ROOT/'task').glob('*/config.yaml')))}")
 except FileExistsError as error: sys.exit(f'START_ERROR: {error}')
 source=ROOT/'solution_templates'/f"{args.task_id}_solution.ipynb"
 title=args.task_id.removeprefix('task').replace('_','.')
 print(f'''Selected Task {title}\n\nRead:\n  task/{args.task_id}/README.md\n\nEdit:\n  your_solution/{source.name}\n\nTraining data:\n  {config['train_data']}\nTest features:\n  {config['test_features']}\n\nYour program must generate:\n  outputs/{args.task_id}_predictions.csv\n\nRequired columns:\n  {config['id_column']},{config['target_column']}\n\nCheck locally:\n  python grader/run_grading.py --task {args.task_id} --sanity-only\n\nRun complete assessment:\n  python grader/run_grading.py --task {args.task_id}\n\nSubmit:\n  git add your_solution/{source.name}\n  git commit -m "Complete Task {title}"\n  git push''')
if __name__=='__main__': main()
