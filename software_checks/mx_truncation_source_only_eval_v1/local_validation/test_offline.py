"""CPU fixtures only. No SVHN data, GPU run or scientific accuracies produced."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import shutil
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
torch.set_num_threads(1)
import svhn_support as s
import resume_core as core
import test_source_eval as tests
import run_eval as runner


def main(out):
    out.mkdir(exist_ok=False)
    rt=s.runtime(20261009,'cpu',strict=False)
    log=s.legacy.Log(out/'offline.log')
    result=tests.run_tests(out,rt,log)
    # Exercise parent staging, two independent workers, CSV comparison and notebook
    # using small fixtures. Only this test harness replaces production inspection
    # and data jobs; run_eval evaluate exposes no fixture/CPU bypass.
    sample=out/'synthetic_validation/staged'
    p=torch.load(sample/'source_E2M3.pt',weights_only=True)
    ref=out/'synthetic_reference';ref.mkdir()
    config=json.loads((ROOT/'REFERENCE_PROTOCOL.json').read_text())
    r={'config':config,'environment':rt.environment,'execution_settings':rt.settings,
       'checkpoint_selection':{},'mx_exports':{},'dataset_files':{'files':{'test_32x32.mat':{}}},
       'dataset_selection':{'test':{}},'transforms':{'test':'synthetic fixture only'}}
    metrics=json.loads((out/'synthetic_validation/fresh_process/worker_report.json').read_text())['metrics']
    history=[]
    for role,suffix,epoch in [('best_E2M3_test','best',52),('last_epoch','last',100)]:
        q=copy.deepcopy(p);q['provenance']['source_epoch']=epoch
        name='mx_source_E2M3_'+suffix+'.pt';saved=core.atomic_save(ref/name,q)
        r['mx_exports'][role]={'6':{'file':name,'sha256':saved['sha256']}}
        r['checkpoint_selection'][suffix]={'epoch':epoch,'sha256':'0'*64}
        for row in metrics:
            history.append({'role':role,'epoch':epoch,'weight_bits':row['weight_bits'],'n':row['n'],
                            'correct1':row['correct1'],'correct5':row['correct5'],
                            'objective_loss':row['hard_cross_entropy']})
    s.write_json(ref/'run_report.json',r)
    s.write_json(ref/'checkpoint_selection.json',r['checkpoint_selection'])
    s.write_csv(ref/'selected_metrics.csv',history)
    before={p.name:core.file_sha256(p) for p in ref.iterdir() if p.is_file()}
    original_launch=runner.launch_worker
    def fixture_launch(job_file,output,logger):
        j=json.loads(job_file.read_text());j['synthetic_fixture_only']=True;j['device']='cpu'
        shutil.copy2(sample/'synthetic_input.npz',job_file.parent/'synthetic_input.npz')
        job_file.write_text(json.dumps(j),encoding='utf-8')
        return original_launch(job_file,output,logger)
    dest=out/'synthetic_evaluation';dest.mkdir()
    report={'schema':'mx-truncation-source-only-eval-v1','status':'PASS','command':'evaluate',
            'warnings':[],'session_wall_seconds':0.0,'synthetic_fixture_only':True,'optimizer_steps':0}
    args=SimpleNamespace(reference_run=ref,data_root=out/'no_real_data')
    with patch.object(runner,'inspect_reference',return_value=(r,history,before)), \
         patch.object(runner,'launch_worker',side_effect=fixture_launch):
        runner.evaluate(args,dest,report,rt,log)
    report['synthetic_fixture_only']=True
    report['full_test_export_evaluation_completed']=False
    report['validation_note']='Small synthetic examples only, no SVHN evaluation.'
    s.write_json(dest/'run_report.json',report)
    runner.feedback(dest)
    assert report['reproduction_status']=='MATCH_COUNTS_CE_CLOSE'
    assert len(report['metrics'])==4
    assert all(w['synthetic_fixture_only'] for w in report['workers'].values())
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    nbpath=dest/'MX_bron_only_evaluatie.ipynb'
    nb=nbformat.read(nbpath,as_version=4)
    NotebookClient(nb,timeout=90,kernel_name='python3',resources={'metadata':{'path':str(dest)}}).execute()
    nbformat.write(nb,dest/'synthetic_notebook_executed.ipynb')
    html,_=HTMLExporter().from_notebook_node(nb)
    (dest/'synthetic_notebook.html').write_text(html,encoding='utf-8')
    import base64
    for i,c in enumerate(nb.cells):
        for j,o in enumerate(c.get('outputs',[])):
            png=o.get('data',{}).get('image/png')
            if png:(dest/f'notebook_{i}_{j}.png').write_bytes(base64.b64decode(png))
    result['two_role_orchestration_fixture']='PASS'
    result['notebook_execution']='PASS'
    s.write_json(out/'validation_report.json',result)
    log('OFFLINE VALIDATION PASS: synthetic only');log.handle.close()

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
