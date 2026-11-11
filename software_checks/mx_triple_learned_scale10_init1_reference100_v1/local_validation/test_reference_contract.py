"""Read-only reference guards using duplicated historical traces as schema fixtures."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,copy,csv,tempfile
P=Path(sys.argv[1]).resolve();OUT=Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(P));sys.dont_write_bytecode=True
import run_train as r
full=r.load_initialization_reference();assert full['kind']=='full100'
pilot=r.load_initialization_reference(P/'init64_reference');assert pilot['kind']=='pilot5'
h=copy.deepcopy(full['history'])
reports={}
for label,reference,expected_count in [('full100',full,100),('pilot_fallback',pilot,5)]:
    destination=OUT/label;destination.mkdir()
    ctx=SimpleNamespace(contract={'synthetic_fixture_only':True},initialization_reference=reference,report={})
    result=r.write_initialization_comparison(ctx,destination,h)
    with (destination/'initialization_comparison.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==expected_count*3
    assert result['matched_epochs']==list(range(1,expected_count+1))
    assert result['unmatched_epochs']==list(range(expected_count+1,101))
    assert result['synthetic_fixture_only'] and result['all_sample_orders_match']
    # The full trace compared to itself must have zero differences; scores
    # between the separate full run and pilot are allowed to differ.
    if label=='full100':assert all(float(x['difference_percentage_points'])==0 for x in rows)
    for kind in ('base','order'):
        bad=copy.deepcopy(h)
        if kind=='base':bad['initial_base_model_sha256']='different'
        else:bad['epochs'][expected_count-1]['coverage']['train']['order_sha256']='different'
        try:r.write_initialization_comparison(ctx,destination,bad)
        except RuntimeError:pass
        else:raise AssertionError((label,kind))
    reports[label]={'matched_epochs':expected_count,'rows':len(rows),'remaining_epochs_explicitly_unmatched':100-expected_count,'mismatched_base_and_last_matched_order_rejected':True}
result={'status':'PASS','software_schema_fixture_only':True,'no_new_accuracy_data':True,'historical_files_untouched':True,'cases':reports,'run_train_sha256':r.s.legacy.sha256(P/'run_train.py')}
(OUT/'reference_contract_report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
