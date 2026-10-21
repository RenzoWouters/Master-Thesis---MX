"""Read-only integration of supplied reports. No model calls, downloads or training."""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
import pandas as pd

FULL = ['single_fp100', 'single_mx100', 'dual100', 'trunc100']

def require(ok, message):
    if not bool(ok):
        raise ValueError(message)

def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)

def read_csv(path):
    return pd.read_csv(path)

def initial_hash(report, history):
    return report.get('initial_model_sha256', history.get('initial_model_sha256'))

def load_all(base):
    base = Path(base)
    manifest = read_json(base/'bronmanifest.json')
    integrity = []
    for f in manifest['files']:
        p = base/f['path']
        ok = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256']
        require(ok, 'Source changed or missing: '+str(p))
        integrity.append({'path': f['path'], 'sha256_match': ok})
    reports, histories, metrics, selected, times = {}, {}, {}, {}, {}
    audits = []
    for spec in manifest['runs']:
        rid = spec['id']; root = base/'bronnen'/rid
        r = read_json(root/'run_report.json'); reports[rid] = r
        require(r['status'] == 'PASS', rid+': run not PASS')
        if not spec['epochs']:
            continue
        df = read_csv(root/'epoch_metrics.csv')
        if 'weight_bits' not in df:
            if 'bits_w_and_a' in df: df['weight_bits'] = df['bits_w_and_a']
            else: df['weight_bits'] = 6 if spec['family']=='single_mx' else 32
        if 'activation_bits' not in df:
            df['activation_bits'] = df['bits_w_and_a']
        df['accuracy_percent'] = df['correct1'].astype(float)/df['n']*100
        df['run_id'] = rid
        require(np.isfinite(df[['objective_loss', 'accuracy_percent','lr']]).all().all(), rid+': non-finite')
        require((df['correct1'].between(0,df['n'])).all(), rid+': invalid counts')
        require(np.allclose(df['top1_percent'],df['accuracy_percent'],atol=2e-5,rtol=0), rid+': accuracy/count mismatch')
        require(not df.duplicated(['epoch','phase','weight_bits']).any(),rid+': duplicated rows')
        epochs = spec['epochs']; modes = sorted(df['weight_bits'].unique())
        require(len(df)==epochs*2*len(modes),rid+': missing mode/epoch/phase')
        require(set(df['epoch'])==set(range(1,epochs+1)),rid+': missing epochs')
        require(r['completed_epoch']==epochs,rid+': incomplete run')
        c=r['config']; ntr=c['train_examples']; nte=c['test_examples']
        require((df.loc[df.phase=='train','n']==ntr).all(),rid+': train count')
        require((df.loc[df.phase=='test','n']==nte).all(),rid+': test count')
        exp_steps=epochs*math.ceil(ntr/c['batch_size'])
        require(r['optimizer_steps']==exp_steps,rid+': optimizer steps')
        expected_lr=df['epoch'].map(lambda e:0.001*(0.1**sum(e>x for x in [50,75,90])))
        require(np.allclose(df.lr,expected_lr,atol=1e-15,rtol=1e-12),rid+': LR mismatch')
        sel=read_csv(root/'selected_metrics.csv')
        if 'weight_bits' not in sel:
            sel['weight_bits']=sel['bits_w_and_a'] if 'bits_w_and_a' in sel else (6 if spec['family']=='single_mx' else 32)
        sel['accuracy_percent']=100*sel.correct1/sel.n
        sel['run_id']=rid
        for _,s in sel.iterrows():
            matched=df[(df.epoch==s.epoch)&(df.phase=='test')&(df.weight_bits==s.weight_bits)]
            require(len(matched)==1 and int(matched.iloc[0].correct1)==int(s.correct1),rid+': selected score mismatch')
        high=df[(df.phase=='test')&(df.weight_bits==max(modes))].sort_values('epoch')
        best_epoch=int(high.loc[high.top1_percent.idxmax(),'epoch'])
        require(best_epoch==r['selected_best_epoch'],rid+': selection mismatch')
        h=read_json(root/'history.json');histories[rid]=h
        for ep in h['epochs']:
            for phase in ['train','test']:
                require(ep['coverage'][phase]['complete_once'],rid+': incomplete sample coverage')
                for row in ep[phase]:
                    bit=row.get('weight_bits',row.get('bits_w_and_a',6 if spec['family']=='single_mx' else 32))
                    check=df[(df.epoch==ep['epoch'])&(df.phase==phase)&(df.weight_bits==bit)].iloc[0]
                    require(int(row['correct1'])==int(check.correct1),rid+': CSV/history mismatch')
            if 'adam_checks' in ep:
                require(ep['adam_checks']['moments_fp32_finite'] and ep['adam_checks']['all_active_adam_steps_correct'],rid+': Adam check')
                require(ep['batchnorm_counters_correct'],rid+': BN check')
                require(ep['gradient_checks']['finite_gradient_checks']==math.ceil(ntr/c['batch_size']),rid+': gradient check')
        t=read_csv(root/'epoch_times.csv'); times[rid]=t
        require(len(t)==epochs and (t.total_epoch_seconds>0).all(),rid+': timing missing')
        require(np.allclose(t.total_epoch_seconds,t.train_seconds+t.test_seconds+t.checkpoint_seconds+t.other_seconds,atol=1e-6),rid+': timing summation')
        metrics[rid]=df;selected[rid]=sel
        audits.append({'run_id':rid,'label':spec['label'],'epochs':epochs,'modes':','.join(map(str,modes)),'optimizer_steps':exp_steps,'metric_rows':len(df),'best_epoch':best_epoch,'warnings':len(r.get('warnings',[])),'status':'PASS (rapportcontrole)'})
    pair_checks=[]
    for a,b in [('single_fp100','single_mx100'),('dual100','trunc100'),('single_mx100','dual100')]:
        ha,hb=histories[a],histories[b]
        ah,bh=initial_hash(reports[a],ha),initial_hash(reports[b],hb)
        rows=[]
        for ea,eb in zip(ha['epochs'],hb['epochs']):
            for ph in ['train','test']:
                rows.append(ea['coverage'][ph]['order_sha256']==eb['coverage'][ph]['order_sha256'])
        pair_checks.append({'run_a':a,'run_b':b,'initialization_hash_equal':ah is not None and ah==bh,'order_hashes_equal':sum(rows),'order_hashes_compared':len(rows),'environment_equal':reports[a]['environment']==reports[b]['environment'],'backend_equal':reports[a]['execution_settings']==reports[b]['execution_settings'],'note':'Reported hashes only; underlying sample arrays and checkpoints not provided. Different mode counts/losses remain possible.'})
    gap=read_csv(base/'bronnen/gap/metrics.csv');gap['accuracy_percent']=100*gap.correct1/gap.n
    require(np.allclose(gap.top1_percent,gap.accuracy_percent,atol=1e-8,rtol=0),'Gap count mismatch')
    cast=read_csv(base/'bronnen/cast/metrics.csv');cast['accuracy_percent']=100*cast.correct1/cast.n
    # Cross-check the H and A paths against the direct training history.
    for _,g in gap.iterrows():
        if g['path'] in ['H_source_E2M3','A_direct_E2M1']:
            bit=6 if g['path']=='H_source_E2M3' else 4
            m=metrics['dual100'].query('epoch == @g.epoch and phase == "test" and weight_bits == @bit')
            require(len(m)==1 and int(m.iloc[0].correct1)==int(g.correct1),'H/A failed training-history reproduction')
    return dict(base=base,manifest=manifest,reports=reports,histories=histories,metrics=metrics,selected=selected,times=times,audits=pd.DataFrame(audits),pair_checks=pd.DataFrame(pair_checks),gap=gap,cast=cast,integrity=integrity)

def final_row(data,rid,bit,epoch=100):
    df=data['metrics'][rid]
    rows=df[(df.epoch==epoch)&(df.phase=='test')&(df.weight_bits==bit)]
    require(len(rows)==1,f'Missing {rid}/{epoch}/{bit}')
    return rows.iloc[0]

def gap_row(data,path,epoch=100):
    df=data['gap'];rows=df[(df.epoch==epoch)&(df.path==path)]
    require(len(rows)==1,'Missing gap path')
    return rows.iloc[0]

def make_tables(data):
    rows=[]
    names={'single_fp100':'W32/A32 afzonderlijk','single_mx100':'E2M3/A32 afzonderlijk','dual100':'Gezamenlijk: directe MX-kwantisatie','trunc100':'Gezamenlijk: truncatiebewust'}
    for rid in FULL:
        r=data['reports'][rid]
        for bit in sorted(data['metrics'][rid].weight_bits.unique(),reverse=True):
            a=final_row(data,rid,bit)
            best=final_row(data,rid,bit,r['selected_best_epoch'])
            rows.append({'run_id':rid,'methode':names[rid],'weight_bits':int(bit),'modus':('W32' if bit==32 else ('E2M3' if bit==6 else 'E2M1'))+'/A32','epoch':100,'accuracy_percent':a.accuracy_percent,'correct1':int(a.correct1),'test_ce':a.objective_loss,'best_epoch':int(best.epoch),'best_accuracy_percent':best.accuracy_percent,'source':f'bronnen/{rid}/selected_metrics.csv'})
    primary=pd.DataFrame(rows)
    gaprows=[]
    for path,label in [('A_direct_E2M1','Directe training / direct E2M1'),('B_round_from_E2M3','Directe training / afronden uit E2M3'),('C_truncate_E2M3','Directe training / trunceren uit E2M3')]:
        g=gap_row(data,path)
        gaprows.append({'methode':label,'path':path,'accuracy_percent':g.accuracy_percent,'correct1':int(g.correct1),'test_ce':g.hard_cross_entropy,'source':'bronnen/gap/metrics.csv'})
    g=final_row(data,'trunc100',4)
    gaprows.append({'methode':'Truncatiebewuste training / trunceren','path':'TA','accuracy_percent':g.accuracy_percent,'correct1':int(g.correct1),'test_ce':g.objective_loss,'source':'bronnen/trunc100/epoch_metrics.csv'})
    gapcompare=pd.DataFrame(gaprows)
    times=[]
    for rid in ['any100']+FULL:
        r=data['reports'][rid];t=data['times'][rid]
        times.append({'run_id':rid,'methode':('Any-Precision vijf W/A-modi' if rid=='any100' else names[rid]),'passes_per_batch':len(data['metrics'][rid].weight_bits.unique()),'train_s':t.train_seconds.sum(),'test_s':t.test_seconds.sum(),'epoch_total_s':t.total_epoch_seconds.sum(),'session_s':r['session_wall_seconds'],'source':f'bronnen/{rid}/epoch_times.csv'})
    return primary,gapcompare,pd.DataFrame(times)

def storage_tables(data):
    b=data['base'];inv=read_json(b/'bronnen/tensorinventaris.json')
    layers=read_csv(b/'bronnen/trunc100/mx_export_layers_6_last.csv')
    low=read_csv(b/'bronnen/trunc100/mx_export_layers_4_last.csv')
    direct_high=read_csv(b/'bronnen/dual100/mx_export_layers_6_last.csv')
    direct_low=read_csv(b/'bronnen/dual100/mx_export_layers_4_last.csv')
    for df in [low,direct_high,direct_low]:
        for c in ['n_weights','blocks','padded_slots']:
            require((df[c].to_numpy()==layers[c].to_numpy()).all(),'Different MX block layouts')
    layers=layers[['layer','n_weights','blocks','padded_slots','fp32_weight_bytes']].copy()
    layers['high_code_bytes']=layers.blocks*32*6//8
    layers['low_code_bytes']=layers.blocks*32*4//8
    layers['scale_bytes']=layers.blocks
    layers['high_bytes']=layers.high_code_bytes+layers.scale_bytes
    layers['low_bytes']=layers.low_code_bytes+layers.scale_bytes
    require((layers.high_bytes.to_numpy()==direct_high.packed_weight_and_scale_bytes.to_numpy()).all(),'E2M3 packing mismatch')
    require((layers.low_bytes.to_numpy()==low.packed_weight_and_scale_bytes.to_numpy()).all(),'E2M1 packing mismatch')
    N=int(layers.n_weights.sum());K=int(layers.blocks.sum());P=int(layers.padded_slots.sum());aux=inv['auxiliary_bytes_all_banks']
    high=int(layers.high_bytes.sum());lo=int(layers.low_bytes.sum());fp=int(layers.fp32_weight_bytes.sum())
    require(fp==inv['target_weight_bytes_fp32'],'Shape/byte mismatch with supplied architecture')
    require(N+P==32*K,'Padding mismatch')
    require(data['reports']['cast']['checks']['last']['auxiliary_state_bytes']==aux,'Auxiliary state evidence mismatch')
    values={'n_weights':N,'blocks':K,'padding':P,'scale_bytes':K,'high_code_bytes':int(layers.high_code_bytes.sum()),'low_code_bytes':int(layers.low_code_bytes.sum()),'high_bytes':high,'low_bytes':lo,'fp32_target_bytes':fp,'auxiliary_bytes':aux,'fp32_all_state_bytes':fp+aux,'source_high_all_bytes':high+aux,'view_low_all_bytes':lo+aux,'source_two_dedup_bytes':high+lo-K+aux,'source_two_independent_bytes':high+lo+2*aux,'six_weights_ratio_fp32_to_high':fp/high,'all_state_ratio_fp32_to_high':(fp+aux)/(high+aux),'target_traffic_reduction_6_to_4_pct':100*(1-lo/high),'source_dedup_reduction_pct':100*(1-(high+aux)/(high+lo-K+aux))}
    representation=pd.DataFrame([
        {'representatie':'FP32-referentie','element_bits':32,'theoretical_bits_with_scale':32.,'weight_code_bytes':fp,'shared_scale_bytes':0,'auxiliary_bytes':aux,'total_bytes':fp+aux,'scope':'FP32 modeltensorinhoud; geen optimizer'},
        {'representatie':'E2M3-bron (ondersteunt 6 en 4 bit)','element_bits':6,'theoretical_bits_with_scale':6.25,'weight_code_bytes':high-K,'shared_scale_bytes':K,'auxiliary_bytes':aux,'total_bytes':high+aux,'scope':'Benodigde behouden bron; vijf BN-banken ongewijzigd'},
        {'representatie':'E2M1-uitvoeringskopie (alleen 4 bit)','element_bits':4,'theoretical_bits_with_scale':4.25,'weight_code_bytes':lo-K,'shared_scale_bytes':K,'auxiliary_bytes':aux,'total_bytes':lo+aux,'scope':'Actieve lage kopie; onvoldoende om later 6 bit te herstellen'},
    ])
    scenarios=pd.DataFrame([
        {'scenario':'FP32-bron, doelprecisie herkwantiseren','code_bytes':fp,'scale_bytes':0,'aux_bytes':aux,'total_bytes':fp+aux,'evidence':'Layoutberekening; bestaande FP32-masterafhankelijke route'},
        {'scenario':'Twee directe MX-codesets, metadata gedeeld','code_bytes':high+lo-2*K,'scale_bytes':K,'aux_bytes':aux,'total_bytes':high+lo-K+aux,'evidence':'Berekenbaar scenario; schalen/aux gelijk bevestigd; gedeeld containerformaat niet gebouwd'},
        {'scenario':'Twee zelfstandige MX-modellen/bestanden','code_bytes':high+lo-2*K,'scale_bytes':2*K,'aux_bytes':2*aux,'total_bytes':high+lo+2*aux,'evidence':'Zelfstandige exports dupliceren metadata; geen gemeten dedicated E2M1-accuracy'},
        {'scenario':'Een E2M3-bron met mantissatruncatie','code_bytes':high-K,'scale_bytes':K,'aux_bytes':aux,'total_bytes':high+aux,'evidence':'Bronexport gerapporteerd; volledige nieuwe exportevaluatie nog open'},
    ])
    # Physical file sizes are reported values, not byte measurements of unavailable exports here.
    physical=[]
    for rid in ['single_mx100','dual100','trunc100']:
        ex=data['reports'][rid]['mx_exports']['last_epoch']
        if rid=='single_mx100':ex={'6':ex}
        for bit,item in ex.items():
            physical.append({'run_id':rid,'bits':int(bit),'file':item['file'],'reported_file_bytes':item['bytes'],'raw_tensor_bytes':(high if int(bit)==6 else lo)+aux,'full_test_rerun':item['full_test_forward_rerun'],'source':f'bronnen/{rid}/run_report.json:mx_exports.last_epoch'})
    # Training state: not a complete RAM / VRAM budget.
    train=[]
    for rid,active in [('single_fp100',['32']),('single_mx100',['32']),('dual100',['32','4']),('trunc100',['32','4']),('any100',['1','2','4','8','32'])]:
        pbytes=sum(t['bytes'] for t in inv['tensors'] if t['is_parameter'] and (t['bn_bank'] is None or t['bn_bank'] in active))
        train.append({'run_id':rid,'master_parameters_bytes_all_banks':sum(t['bytes'] for t in inv['tensors'] if t['is_parameter']),'model_state_bytes_including_BN_buffers':fp+aux,'adam_two_moments_bytes':2*pbytes,'subtotal_state_plus_moments_bytes':fp+aux+2*pbytes,'excluded':'gradients, Adam step counters, activations, temporary caches, workspaces, allocator, Python, saved RNG'})
    return values,layers,representation,scenarios,pd.DataFrame(physical),pd.DataFrame(train)
