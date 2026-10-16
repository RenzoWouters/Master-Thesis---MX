"""Integration of CLI/report export on an existing SYNTHETIC fixture.

No training or dataset access is performed here. Run test_svhn.py first.
Production entrypoint has no synthetic bypass; mocks live only in this test.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile
import json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_svhn as runner
import svhn_support as s
import resume_core as core
assert len(sys.argv)==3, 'Usage: test_cli_export.py EXISTING_SYNTHETIC_VALIDATION_DIR NEW_OUTPUT_ROOT'
fixture=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve()
assert not out.exists(), 'Output must be new'
state=core.load_verified(fixture/'instrumented/checkpoints/epoch_00100.pt')
assert state['contract']['synthetic_fixture_only'] is True
s.verify_package()

def context(args, directory, report, log, check_only=False):
    assert args.command=='run' and args.checkpoint is None and args.stop_after_epoch==100
    rt=SimpleNamespace(core=core)
    c=state['contract']['config']
    report.update(config=c,environment=state['contract']['environment'],
                  execution_settings=state['contract']['settings'],synthetic_fixture_only=True)
    return SimpleNamespace(rt=rt,c=c,contract=state['contract'],report=report)

def run_epochs(ctx, directory, log, stop_after, resume):
    assert stop_after==100 and resume is None
    trace=json.loads((fixture/'instrumented/history.json').read_text())
    assert len(trace['epochs'])==100
    return trace

with patch.object(sys,'argv',['run_svhn.py','run','--output-root',str(out)]), \
     patch.object(runner,'create_context',context), \
     patch.object(runner,'run_epochs',run_epochs), \
     patch.object(runner,'verify_unchanged',lambda *a: None):
    assert runner.main()==0
run_dir=next(out.glob('*_run'))
report=json.loads((run_dir/'run_report.json').read_text())
assert report['status']=='PASS' and report['completed_epoch']==100
assert report['full_training_completed'] is True and report['optimizer_steps']==200
with ZipFile(run_dir/'feedback_bundle.zip') as z:
    expected={'run_report.json','completion_summary.json','learning_rate_schedule.csv','epoch_times.csv',
              'checkpoint_selection.json','selected_metrics.csv','history.json','timing_estimate.json'}
    assert expected.issubset(set(z.namelist()))
    assert not any(n.endswith('.pt') for n in z.namelist())
assert (run_dir/'SVHN_trainonly_reference100_resultaten.ipynb').is_file()
checks={'status':'PASS','scope':'SYNTHETIC fixture, mocked context/epoch execution only; no real training',
        'default_run_endpoint_100':True,'fresh_run_has_no_checkpoint_argument':True,
        'CLI_final_report_and_feedback_bundle':True,'real_GPU_test':False}
(out/'cli_export_report.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
