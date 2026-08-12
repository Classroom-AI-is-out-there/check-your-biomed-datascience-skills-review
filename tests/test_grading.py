import json, os, shutil
from pathlib import Path
import numpy as np, pandas as pd, pytest, yaml
import grader.run_grading as grading
from biomed_submission import get_output_path
from tools.start_task import initialize

@pytest.fixture
def isolated(monkeypatch,tmp_path):
    (tmp_path/'task/task1').mkdir(parents=True); (tmp_path/'your_solution').mkdir(); (tmp_path/'solution_templates').mkdir()
    config={'task_id':'task1','type':'classification','metric':'f1','average':'weighted','id_column':'patient_id','target_column':'prediction','threshold':.75,'install_timeout_seconds':600,'execution_timeout_seconds':300,'train_data':'dataset/task1/train.csv','test_features':'dataset/task1/test_features.csv','allowed_labels':[0,1]}
    (tmp_path/'task/task1/config.yaml').write_text(yaml.safe_dump(config)); monkeypatch.setattr(grading,'REPO_ROOT',tmp_path); monkeypatch.setattr(grading,'SUBMISSION_DIR',tmp_path/'your_solution')
    return tmp_path,config

def test_detection_none_and_templates_ignored(isolated):
 root,_=isolated; (root/'solution_templates/task1_solution.py').write_text('')
 with pytest.raises(grading.GradingError,match='No active'): grading.detect_task()
@pytest.mark.parametrize('suffix',['.py','.ipynb'])
def test_detection_one(isolated,suffix):
 root,_=isolated; p=root/'your_solution'/f'task1_solution{suffix}'; p.write_text('{}'); assert grading.detect_task()==('task1',p)
def test_detection_both_extensions(isolated):
 root,_=isolated
 for s in ['.py','.ipynb']: (root/'your_solution'/f'task1_solution{s}').write_text('{}')
 with pytest.raises(grading.GradingError,match='exactly one'): grading.detect_task()
def make_features(root):
 (root/'dataset/task1').mkdir(parents=True); pd.DataFrame({'patient_id':[1,2,3]}).to_csv(root/'dataset/task1/test_features.csv',index=False)
def test_validation_matrix(isolated):
 root,c=isolated; make_features(root); p=root/'pred.csv'
 cases=[(pd.DataFrame({'bad':[1,2,3]}),'missing required'),(pd.DataFrame({'patient_id':[1,2],'prediction':[0,1]}),'row count'),(pd.DataFrame({'patient_id':[1,1,3],'prediction':[0,1,0]}),'must be unique'),(pd.DataFrame({'patient_id':[1,2,4],'prediction':[0,1,0]}),'do not exactly match'),(pd.DataFrame({'patient_id':[1,2,3],'prediction':[0,np.nan,0]}),'missing values'),(pd.DataFrame({'patient_id':[1,2,3],'prediction':[0,np.inf,0]}),'infinite'),(pd.DataFrame({'patient_id':[1,2,3],'prediction':[0,9,0]}),'invalid classification')]
 for frame,message in cases:
  frame.to_csv(p,index=False)
  with pytest.raises(grading.GradingError,match=message): grading.validate_predictions('task1',c,p)
 pd.DataFrame({'patient_id':[1,2,3],'prediction':[0,1,0]}).to_csv(p,index=False); assert grading.validate_predictions('task1',c,p)[1]['passed']
def test_missing_output(isolated):
 root,c=isolated; make_features(root)
 with pytest.raises(grading.GradingError,match='missing prediction'): grading.validate_predictions('task1',c,root/'no.csv')
def test_clustering_single_and_success(isolated):
 root,c=isolated; make_features(root); c.update(type='clustering',min_clusters=2,max_clusters=3); p=root/'p.csv'
 pd.DataFrame({'patient_id':[1,2,3],'prediction':[0,0,0]}).to_csv(p,index=False)
 with pytest.raises(grading.GradingError,match='only one'): grading.validate_predictions('task1',c,p)
 pd.DataFrame({'patient_id':[1,2,3],'prediction':[0,1,0]}).to_csv(p,index=False); grading.validate_predictions('task1',c,p)
def test_output_paths(monkeypatch,tmp_path):
 monkeypatch.chdir(tmp_path); monkeypatch.delenv('BIOMED_OUTPUT_PATH',raising=False); assert get_output_path('task1')==Path('outputs/task1_predictions.csv')
 override=tmp_path/'fresh/p.csv'; monkeypatch.setenv('BIOMED_OUTPUT_PATH',str(override)); assert get_output_path('task1')==override and override.parent.is_dir()
def test_separate_timeouts(isolated):
 _,c=isolated; loaded=grading.load_config('task1'); assert loaded['install_timeout_seconds']==600 and loaded['execution_timeout_seconds']==300

def test_start_task_and_refuse_overwrite(tmp_path):
 (tmp_path/'task/task1').mkdir(parents=True); (tmp_path/'solution_templates').mkdir(); (tmp_path/'your_solution').mkdir()
 (tmp_path/'task/task1/config.yaml').write_text('task_id: task1\n'); (tmp_path/'solution_templates/task1_solution.ipynb').write_text('{}')
 destination,_=initialize('task1',root=tmp_path); assert destination.read_text()=='{}'
 with pytest.raises(FileExistsError,match='already exists'): initialize('task1',root=tmp_path)
 initialize('task1',force=True,root=tmp_path)

def test_all_reference_labels_satisfy_contract():
 for tid in grading.task_ids():
  c=grading.load_config(tid); labels=pd.read_csv(grading.REPO_ROOT/'grading'/tid/'test_labels.csv')
  frame=labels.rename(columns={labels.columns[-1]:c['target_column']}) if c['target_column'] not in labels else labels
  with pytest.MonkeyPatch.context() as mp:
   with __import__('tempfile').TemporaryDirectory() as d:
    p=Path(d)/'pred.csv'; frame[[c['id_column'],c['target_column']]].to_csv(p,index=False)
    validated,_=grading.validate_predictions(tid,c,p); assert len(validated)==len(frame)

def test_readme_catalog_matches_configs():
 readme=(grading.REPO_ROOT/'README.md').read_text()
 for tid in grading.task_ids():
  c=grading.load_config(tid); assert f'`{tid}`' in readme and f"{c['threshold']:.2f}" in readme

def test_structured_report_shape(isolated,monkeypatch,tmp_path):
 root,c=isolated; make_features(root); (root/'grading/task1').mkdir(parents=True); pd.DataFrame({'patient_id':[1,2,3],'prediction':[0,1,0]}).to_csv(root/'grading/task1/test_labels.csv',index=False)
 solution=root/'your_solution/task1_solution.py'; solution.write_text("import os,pandas as pd\npd.DataFrame({'patient_id':[1,2,3],'prediction':[0,1,0]}).to_csv(os.environ['BIOMED_OUTPUT_PATH'],index=False)\n")
 report=root/'report.json'; assert grading.assess('task1',False,report,True)==0; data=json.loads(report.read_text()); assert set(['task_id','solution_path','execution','validation','metrics','thresholds','score','feedback'])<=data.keys()
