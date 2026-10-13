"""Small source/config/reporting checks; no actual CIFAR training or target accuracy."""
from pathlib import Path
import argparse, hashlib, json, re, sys, tempfile, zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_reference as ref
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
checks={}
hashes=ref.verify_package();c=ref.load_config()
with zipfile.ZipFile(ROOT/'upstream/Any-Precision-DNNs-master.zip') as z:
    sh=z.read('Any-Precision-DNNs-master/train_cifar10.sh').decode()
    train=z.read('Any-Precision-DNNs-master/train.py').decode()
checks['epochs_match_source_script']=int(re.search(r'--epochs (\d+)',sh)[1])==c['planned_epochs']==200
checks['milestones_match_source_script']=[int(x) for x in re.search(r'--lr_decay "([\d,]+)"',sh)[1].split(',')]==c['scheduler_milestones']==[100,150,180]
checks['lr_matches_source_script']=float(re.search(r'--lr ([\d.]+)',sh)[1])==c['lr']==0.001
checks['batch_size_matches_source_default']="'--batch-size', default=128" in train and c['batch_size']==128
checks['wd_and_adam_match_script']='--optimizer adam' in sh and '--weight-decay 0.0' in sh and c['weight_decay']==0 and c['optimizer']=='adam'
reference=json.loads((ROOT/'github_targets.json').read_text())
line=next(s for s in (ROOT/'source_notes/upstream_README.md').read_text().splitlines() if s.startswith('| [Resnet-20-Any (recursive'))
checks['github_reference_extracted_from_readme']=[float(s.strip()) for s in line.split('|')[2:-1]]==reference['top1_percent']
rows=[{'bits_w_and_a':b,'phase':'official_test','epoch':200,'n':10000,'top1_percent':v-0.5} for b,v in zip(reference['bits'],reference['top1_percent'])]
r=ref.github_comparison(rows,200,False)
checks['comparison_pp_and_same_checkpoint']=all(x['difference_percentage_points']==-0.5 and x['epoch']==200 for x in r['rows'])
checks['comparison_has_no_accuracy_threshold']=r['comparison_has_no_accuracy_pass_threshold']
try:
    ref.github_comparison(rows[:-1],200,False)
    checks['reject_missing_precision']=False
except RuntimeError:checks['reject_missing_precision']=True
bad=[dict(x) for x in rows];bad[0]['phase']='train'
try:
    ref.github_comparison(bad,200,False)
    checks['reject_training_accuracy_comparison']=False
except RuntimeError:checks['reject_training_accuracy_comparison']=True
bad=[dict(x) for x in rows];bad[0]['epoch']=199
try:
    ref.github_comparison(bad,200,False)
    checks['reject_mixed_checkpoint_epochs']=False
except RuntimeError:checks['reject_mixed_checkpoint_epochs']=True
with tempfile.TemporaryDirectory() as temp:
    folder=Path(temp);bad=dict(c);bad['planned_epochs']=400;bad['scheduler_milestones']=[150,250,350]
    (folder/'reference_config.json').write_text(json.dumps(bad))
    old=ref.ROOT;ref.ROOT=folder
    try:
        ref.load_config();checks['reject_400_config']=False
    except RuntimeError:checks['reject_400_config']=True
    finally:ref.ROOT=old
est=ref.estimate_time([{'epoch':1,'total_epoch_seconds':50.0},{'epoch':2,'total_epoch_seconds':36.0}],2,200)
checks['timing_projection_uses_200']=est['planned_epochs']==200 and est['projected_full_training_hours_at_this_rate']==2.0 and est['estimated_remaining_training_hours']==1.98 and est['epochs_used']==[2]
checks['legacy_400_fields_removed']='projected_400_epoch_training_hours_at_this_rate' not in (ROOT/'run_reference.py').read_text() and 'storage_projection_401' not in (ROOT/'run_reference.py').read_text()
result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'scope':'source/config/reporting regression only; invented scores for arithmetic unit test, not model results',
        'checked_code_sha256':hashes}
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
assert all(checks.values())
