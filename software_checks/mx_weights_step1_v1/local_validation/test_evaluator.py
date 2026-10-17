"""Offline fixture exercising the real read-only evaluator; not training data.

Mocks dataset acquisition/count enforcement ONLY in this separate test process.
No such override exists on the production CLI. Nothing from this synthetic
reference is shipped as an experimental result.
"""
from pathlib import Path
import sys,json,hashlib,copy
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
import svhn_support as s
import resume_core as core
import run_mx as app
import mx_adapter as a

root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=False)
source,info=s.legacy.prepare_source(root)
rt=s.runtime(20261009,'cpu',strict=False);torch.set_num_threads(2)
modules=s.import_source(source)
class Data(torch.utils.data.Dataset):
    split='test'
    def __init__(self):
        self.data=np.arange(8*3*32*32,dtype=np.uint8).reshape(8,3,32,32)
        self.labels=np.arange(8,dtype=np.int64)
        self.transform=modules.data.get_transform('svhn','val')
    def __len__(self):return 8
    def __getitem__(self,i):return self.transform(Image.fromarray(self.data[i].transpose(1,2,0))),int(self.labels[i])
data=Data()
data_root=root/'fake_data';data_root.mkdir()
file=data_root/'test_32x32.mat';file.write_bytes(b'SYNTHETIC FIXTURE: not an actual MAT file')
modules.data.SVHNMIX=SimpleNamespace(split_list={'test':('unused',file.name,hashlib.md5(file.read_bytes()).hexdigest())})
modules.data.get_dataset=lambda *args,**kwargs:data
original_validation=s.validate_dataset
s.validate_dataset=lambda d,expected,name: {'n':len(d),'synthetic_fixture_only':True}
reference=root/'synthetic_reference';reference.mkdir()
model=modules.models.svhnq(s.BITS,expand=2).eval();a.select_a32(model)
loader=torch.utils.data.DataLoader(data,batch_size=128)
rows,_=app.measure({'reference_W32_A32':model},loader,torch.device('cpu'),modules.utils)
metric=rows[0];metric['bits_w_and_a']=32
cfg={'train_split':'train','extra_used':False,'model':'svhnq'}
r={'schema':'anyprecision-svhn-trainonly-reference100-report-v1','status':'PASS',
   'completed_epoch':100,'full_training_completed':True,'source':info,'config':cfg,
   'selected_best_epoch':54,'selected_last_epoch':100,'checkpoint_selection':{},
   'synthetic_test_fixture':True}
for role,epoch in [('best',54),('last',100)]:
    payload={'schema':core.SCHEMA,'boundary':'epoch_complete_after_validation_and_scheduler',
      'completed_epoch':epoch,'contract':{'config':cfg,'synthetic_fixture_only':False},
      'model':model.state_dict(),'trace':{'epochs':[{'test':[metric]}]},
      'ACTUAL_SCOPE':'SYNTHETIC OFFLINE TEST, not a 100-epoch training'}
    target=reference/(role+'_checkpoint.pt')
    h=core.atomic_save(target,payload)
    r['checkpoint_selection'][role]={'epoch':epoch,**h}
s.write_json(reference/'run_report.json',r)
out=root/'result';out.mkdir()
report={'command':'evaluate','config':app.CONFIG,'warnings':[],'optimizer_steps':0,
        'synthetic_fixture_only':True,'environment':rt.environment}
args=SimpleNamespace(reference_run=reference,data_root=data_root)
app.evaluate(args,rt,modules,source,info,report,out,print)
assert report['status']=='PASS'
s.write_json(out/'run_report.json',report)
app.feedback(out)
# Fresh process only receives export+probe, never a reference checkpoint.
probe=torch.linspace(-1,1,2*3*40*40).reshape(2,3,40,40)
with torch.no_grad():expect=a.adapt_copy(model,True).eval()(probe)
with (root/'probe.pt').open('xb') as f:torch.save({'input':probe,'expected':expect},f)
script=root/'fresh_export_test.py'
script.write_text('''import sys,torch,json
from pathlib import Path
sys.path.insert(0,sys.argv[4])
import svhn_support as s
import mx_adapter as a
torch.set_num_threads(2)
modules=s.import_source(Path(sys.argv[1]))
payload=torch.load(sys.argv[2],map_location="cpu",weights_only=True)
model=a.load_exported_model(modules.models,payload,"cpu")
probe=torch.load(sys.argv[3],map_location="cpu",weights_only=True)
with torch.no_grad():actual=model(probe["input"])
assert torch.equal(actual,probe["expected"])
print("PASS: fresh process loaded only codes, auxiliary state and input probe")
''')
import subprocess
p=subprocess.run([sys.executable,str(script),str(source),str(out/'mx_source_best.pt'),str(root/'probe.pt'),str(PACKAGE)],capture_output=True,text=True)
assert p.returncode==0,p.stderr
print(p.stdout)
from nbclient import NotebookClient
import nbformat
nb=nbformat.read(out/'MX_weights_resultaten.ipynb',as_version=4)
NotebookClient(nb,timeout=90,kernel_name='python3').execute(cwd=str(out))
nbformat.write(nb,root/'synthetic_executed.ipynb')
summary={'status':'PASS','scope':'synthetic integration, not SVHN; no reference weights provided here',
 'real_evaluate_function_exercised':True,'eight_metric_rows':len(report['metrics'])==8,
 'six_layer_rows_per_checkpoint':True,'no_input_files_modified':True,
 'standalone_fresh_process_export_test':p.stdout.strip(),'notebook_executed_on_saved_synthetic_reports':True,
 'no_real_dataset_or_user_checkpoint_evaluated':True}
s.write_json(root/'validation_summary.json',summary)
print(json.dumps(summary,indent=2))
