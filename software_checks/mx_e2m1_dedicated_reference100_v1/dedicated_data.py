"""Read-only checks for adding a measured dedicated E2M1 run to the overview."""
from pathlib import Path
import csv, json, math, hashlib

SCHEMA = 'mx-e2m1-dedicated-reference100-report-v1'
FILES = ('run_report.json','epoch_metrics.csv','selected_metrics.csv','epoch_times.csv',
         'history.json','checkpoint_selection.json','training_summary.json',
         'mx_export_layers_best.csv','mx_export_layers_last.csv','training_change_manifest.json')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def load_dedicated(root, *, allow_synthetic=False):
    root=Path(root)
    r=json.loads((root/'run_report.json').read_text(encoding='utf-8'))
    c=r['config']; synthetic=bool(r.get('synthetic_fixture_only',False))
    require(not synthetic or allow_synthetic, 'Synthetische data mogen niet als onderzoeksresultaat worden toegevoegd.')
    require(r['schema']==SCHEMA and r['status']=='PASS' and r['full_training_completed'], 'Geen volledig geslaagde dedicated E2M1-run.')
    require(r['completed_epoch']==100 and c['planned_epochs']==100, 'Geen honderd epochs.')
    for key,value in {'mode':'mx','weight_format':'MXFP4_E2M1','activation_bits':32,
                      'master_dtype':'float32','distillation':False,'truncation':False,
                      'train_split':'train','extra_used':False,'passes_per_training_batch':1,
                      'optimizer_steps_per_training_batch':1,'mx_block_size':32,'mx_scale_format':'E8M0',
                      'model':'svhnq','expand':2,'lr':.001,'weight_decay':0.,
                      'scheduler_milestones':[50,75,90],'scheduler_gamma':.1}.items():
        require(c.get(key)==value, f'Verkeerd protocolveld: {key}.')
    if not synthetic:
        require(c['train_examples']==73257 and c['test_examples']==26032 and c['batch_size']==128, 'Afwijkende data of batch.')
        require(r['optimizer_steps']==57300,'Afwijkend aantal updates.')
    h=json.loads((root/'history.json').read_text(encoding='utf-8'))
    require([z['epoch'] for z in h['epochs']]==list(range(1,101)),'Onvolledige historie.')
    def rows(name):
        with (root/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    e=rows('epoch_metrics.csv');require(len(e)==200,'Verwacht 200 metrics.')
    expected={(i,ph) for i in range(1,101) for ph in ['train','test']}
    require({(int(z['epoch']),z['phase']) for z in e}==expected,'Ontbrekende/dubbele metrics.')
    for z in e:
        ep=int(z['epoch']);ph=z['phase'];v=h['epochs'][ep-1][ph][0]
        require(z['weight_format']=='MXFP4_E2M1','Verkeerde metricprecisie.')
        require(z['objective_type']=='hard_cross_entropy','Onverwacht leerdoel.')
        n=int(z['n']);co=int(z['correct1']);acc=float(z['count_top1_percent'])
        require(n==c['train_examples' if ph=='train' else 'test_examples'] and 0<=co<=n,'Onjuist aantal voorbeelden.')
        require(abs(acc-100*co/n)<1e-8 and math.isfinite(float(z['objective_loss'])),'Accuracy/loss inconsistent.')
        require(co==v['correct1'] and n==v['n'] and abs(float(z['objective_loss'])-v['objective_loss'])<1e-10,'CSV/historie inconsistent.')
        lr=c['lr']
        for milestone in c['scheduler_milestones']:
            if ep>milestone:lr*=c['scheduler_gamma']
        require(abs(float(z['lr'])-lr)<1e-15,'Verkeerd LR-schema.')
    sel=rows('selected_metrics.csv');require(len(sel)==2,'Verwacht best en last.')
    best=h['epochs'][0]
    for v in h['epochs'][1:]:
        if v['test'][0]['top1_percent']>best['test'][0]['top1_percent']:best=v
    for z in sel:
        role=z['role'];require(role in ('best_active_test','last_epoch'),'Onbekende checkpointrol.')
        ep=best['epoch'] if role=='best_active_test' else 100
        require(int(z['epoch'])==ep and int(z['correct1'])==h['epochs'][ep-1]['test'][0]['correct1'],'Verkeerde checkpointselectie.')
    times=rows('epoch_times.csv'); require(len(times)==100,'Onvolledige tijden.')
    elapsed=sum(float(z['total_epoch_seconds']) for z in times)
    layer=rows('mx_export_layers_last.csv')
    require(len(layer)==6,'Onvolledige laagdiagnostiek.')
    payload=sum(int(z['packed_weight_and_scale_bytes']) for z in layer)
    require(payload==69360,'Onverwachte directe E2M1-code/schaalomvang.')
    require(h['initial_model_sha256']==r['initial_model_sha256'],'Initialisatiehash mismatch.')
    return {'report':r,'history':h,'metrics':e,'selected':sel,'times':times,
            'epoch_seconds':elapsed,'code_scale_bytes':payload,'synthetic':synthetic,
            'files_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in FILES}}
