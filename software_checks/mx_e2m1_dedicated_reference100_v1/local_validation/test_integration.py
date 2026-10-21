from pathlib import Path
import sys,json,copy,shutil,hashlib,base64
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from dedicated_data import load_dedicated
from voeg_toe_aan_overzicht import integrate
B=Path(sys.argv[1]).resolve(); V=Path(sys.argv[2]).resolve()
# Synthetic test fixture may be read ONLY using internal test flag, not CLI.
try: load_dedicated(V)
except ValueError: pass
else: raise AssertionError('Production integration allowed synthetic results')
data=load_dedicated(V,allow_synthetic=True)
assert data['code_scale_bytes']==69360
O=B/'meeting_integration_test'
base=Path(sys.argv[3]).resolve()
original={str(x.relative_to(base)):hashlib.sha256(x.read_bytes()).hexdigest() for x in base.rglob('*') if x.is_file()}
result=integrate(V,base,O,_synthetic_test=True)
assert original=={str(x.relative_to(base)):hashlib.sha256(x.read_bytes()).hexdigest() for x in base.rglob('*') if x.is_file()}
try: integrate(V,base,O,_synthetic_test=True)
except ValueError: pass
else: raise AssertionError('Existing output not rejected')
import nbformat
from nbclient import NotebookClient
n=nbformat.read(O/'MX_onderzoeksoverzicht.ipynb',as_version=4)
NotebookClient(n,timeout=90,kernel_name='python3').execute(cwd=str(O))
nbformat.write(n,O/'MX_onderzoeksoverzicht_uitgevoerd_TEST.ipynb')
(O/'figuren').mkdir(exist_ok=True)
for i,c in enumerate(n.cells):
 for j,out in enumerate(c.get('outputs',[])):
  if 'image/png' in out.get('data',{}):(O/'figuren'/f'cell_{i}_{j}.png').write_bytes(base64.b64decode(out.data['image/png']))
# Wrong schema / missing metrics must be rejected.
F=B/'invalid_integration_fixture';F.mkdir()
for x in V.iterdir():
 if x.is_file() and x.suffix in ['.json','.csv']:shutil.copy2(x,F/x.name)
r=json.loads((F/'run_report.json').read_text());r['schema']='wrong';(F/'run_report.json').write_text(json.dumps(r))
try:load_dedicated(F,allow_synthetic=True)
except ValueError:pass
else:raise AssertionError('Wrong schema accepted')
(B/'integration_validation.json').write_text(json.dumps({'status':'PASS','scope':'SYNTHETIC_DEDICATED_DATA_ONLY',
 'completed_overview_notebook_executed':True,'original_overview_byte_identical':True,
 'production_rejects_synthetic':True,'overwrite_rejected':True,'wrong_schema_rejected':True,
 'no_real_E2M1_accuracy_available':True,'dedicated_payload_code_scale_bytes':69360},indent=2))
print('PASS integration and executed meeting notebook')
