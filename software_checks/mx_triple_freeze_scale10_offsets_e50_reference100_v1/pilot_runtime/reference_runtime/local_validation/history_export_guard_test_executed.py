from pathlib import Path
import copy,hashlib,json,sys,time
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

base=Path('/workspace/scratch/4619498905be')
pkg=base/'work/mx_triple_learned_scale10_reference100_v1'
sys.path.insert(0,str(pkg/'local_validation'))
import test_training_integration as h
h.torch.set_num_threads(2)
old=base/'work/scale10_full100_validation/integration_v1/resume50_to100/20261104_191435_287243_joint_resume'
out=base/'work/scale10_full100_validation/export_readback_v2'
out.mkdir(parents=True,exist_ok=False)
old_report=json.loads((old/'run_report.json').read_text())
checkpoint=old/'checkpoints/epoch_00100.pt'
state=h.core.load_verified(checkpoint)
trace=copy.deepcopy(state['trace'])
rt=h.s.runtime(state['contract']['config']['seed'],'cpu',strict=False)
h.r.restore_timing_sidecar(rt,trace)
assert len(trace['epochs'])==100 and trace['epochs'][-1]['epoch']==100
expected=(json.dumps(trace,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
corrupted=(old/'history.json').read_bytes()
reconstructed=out/'reconstructed_history_from_checkpoint.json'
with reconstructed.open('xb') as f:f.write(expected)
result={'synthetic_fixture_only':True,'training_performed':False,
 'source_checkpoint_sha256':h.core.file_sha256(checkpoint),
 'original_corrupt_history_sha256':hashlib.sha256(corrupted).hexdigest(),
 'corrupt_bytes':len(corrupted),'reconstructed_bytes':len(expected),
 'corrupt_is_exact_prefix':corrupted==expected[:len(corrupted)],
 'corrupt_is_exact_first_half':corrupted==expected[:len(expected)//2],
 'reconstructed_sha256':hashlib.sha256(expected).hexdigest(),
 'original_corrupt_history_preserved':True,
 'restored_final_timing_from_immutable_sidecar':True}
print(json.dumps(result,indent=2),flush=True)
source=base/'work/scale10_full100_validation/integration_v1/source/source_workcopy'
modules=h.s.import_source(source)
model=h.t.build_model(modules,'joint',rt.device)
model.load_state_dict(state['model'],strict=True)
model.eval()
report=copy.deepcopy(old_report)
report.update(output_directory=str(out),artifact_reexport_only=True,
              artifact_reexport_source_directory=str(old),synthetic_fixture_only=True)
ctx=SimpleNamespace(c=state['contract']['config'],rt=rt,model=model,
                    modules=modules,contract=state['contract'],report=report,
                    source=source,source_info=report['source'])
events=[]
original_write=h.s.write_json

def observed_write(path,value):
 t0=time.perf_counter();original_write(path,value)
 event={'filename':path.name,'seconds':time.perf_counter()-t0,'bytes':path.stat().st_size}
 if path.name=='history.json':
  b=path.read_bytes();event.update(sha256=hashlib.sha256(b).hexdigest(),equals_expected=b==expected,
                                epochs=len(json.loads(b)['epochs']))
 events.append(event)

log=h.s.legacy.Log(out/'export_only.log')
with patch.object(h.s,'write_json',observed_write):
 h.r.export_results(ctx,out,trace,log)
 original_write(out/'run_report.json',ctx.report)
 h.r.feedback_bundle(out)
log.handle.close()
actual=(out/'history.json').read_bytes()
with ZipFile(out/'feedback_bundle.zip') as z:zbytes=z.read('history.json')
result.update(export_only_completed=True,export_history_bytes=len(actual),
              export_history_sha256=hashlib.sha256(actual).hexdigest(),
              export_history_equals_reconstructed=actual==expected,
              feedback_history_equals_reconstructed=zbytes==expected,
              export_json_events=events,
              original_corrupt_history_unchanged=(old/'history.json').read_bytes()==corrupted,
              checkpoint_unchanged=h.core.file_sha256(checkpoint)==result['source_checkpoint_sha256'])
assert result['original_corrupt_history_unchanged'] and result['checkpoint_unchanged']
assert actual==expected and zbytes==expected
assert len(json.loads(actual)['epochs'])==100
# The final production exporter must record the bytes it actually checked.
validation=ctx.report['history_export_validation']
assert validation == {'file':'history.json','bytes':len(expected),
                     'sha256':hashlib.sha256(expected).hexdigest(),'epochs':100,
                     'exact_trace_match':True}
with ZipFile(out/'feedback_bundle.zip') as z:
 archived_report=json.loads(z.read('run_report.json'))
 assert archived_report['history_export_validation']==validation
# Deliberate corruption belongs only to a separate new test copy.
negative=out/'deliberately_truncated_history_copy.json'
with negative.open('xb') as handle:handle.write(expected[:len(expected)//2])
try:
 h.r.validate_history_export(negative,trace,h.core)
 raise AssertionError('Truncated history was accepted.')
except RuntimeError as exc:
 assert 'onleesbaar of onvolledig' in str(exc)
 truncated_error=str(exc)
try:
 h.r.validate_history_export(out/'missing_history_copy.json',trace,h.core)
 raise AssertionError('Missing history was accepted.')
except RuntimeError as exc:
 assert 'onleesbaar of onvolledig' in str(exc)
 missing_error=str(exc)
# Valid JSON with the same top-level keys but missing epochs also cannot pass.
incomplete=out/'deliberately_incomplete_history_copy.json'
small_trace=copy.deepcopy(trace)
small_trace['epochs']=[]
with incomplete.open('x') as handle:json.dump(small_trace,handle,indent=2)
try:
 h.r.validate_history_export(incomplete,trace,h.core)
 raise AssertionError('Incomplete valid JSON history was accepted.')
except RuntimeError as exc:
 assert 'wijkt af van de volledige epochgeschiedenis' in str(exc)
 incomplete_error=str(exc)
# Bind all previous numerical execution to the final guard-only code change.
import ast
continuous=base/'work/scale10_full100_validation/integration_v1/continuous100/20261104_191338_916146_joint_run'
with ZipFile(continuous/'feedback_bundle.zip') as z:original_runner=z.read('runner_source/run_train.py')
original_report=json.loads((continuous/'run_report.json').read_text())
assert hashlib.sha256(original_runner).hexdigest()==original_report['package_sha256']['run_train.py']
current_runner=(pkg/'run_train.py').read_bytes()
old_ast=ast.parse(original_runner)
new_ast=ast.parse(current_runner)
new_ast.body=[node for node in new_ast.body if not (
 isinstance(node,ast.FunctionDef) and node.name=='validate_history_export') and not (
 isinstance(node,ast.Import) and len(node.names)==1 and node.names[0].name=='hashlib')]
export_fn=next(node for node in new_ast.body if isinstance(node,ast.FunctionDef) and node.name=='export_results')
removed=[]
for node in list(export_fn.body):
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant)
       and t.slice.value=='history_export_validation' for t in node.targets):
  removed.append(node);export_fn.body.remove(node)
assert len(removed)==1
assert ast.dump(new_ast,include_attributes=False)==ast.dump(old_ast,include_attributes=False)
final_runtime=h.runtime_hashes()
assert set(final_runtime)==set(original_report['package_sha256'])
changed_runtime=[name for name in final_runtime if final_runtime[name]!=original_report['package_sha256'][name]]
assert changed_runtime==['run_train.py'], changed_runtime
assert (old/'history.json').read_bytes()==corrupted
assert (out/'history.json').read_bytes()==expected
result.update(history_export_validation=validation,
              guard_rejects_deliberately_truncated_copy=True,guard_rejects_missing_file=True,
              guard_rejects_incomplete_valid_JSON=True,
              negative_messages={'truncated':truncated_error,'missing':missing_error,'incomplete':incomplete_error},
              original_corruption_cause='Unknown; not reproduced by two export-only runs; original evidence retained.',
              AST_binding={'only_export_guard_import_helper_and_final_call_changed':True,
                           'previously_executed_run_train_sha256':hashlib.sha256(original_runner).hexdigest(),
                           'final_run_train_sha256':hashlib.sha256(current_runner).hexdigest(),
                           'all_other_training_runtime_files_unchanged':True},
              final_training_runtime_sha256=final_runtime,
              test_script_sha256=h.core.file_sha256(Path(__file__)))
result['status']='PASS' 
(out/'history_export_guard_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','output':str(out),'history_bytes':len(actual),'training_performed':False},indent=2),flush=True)
