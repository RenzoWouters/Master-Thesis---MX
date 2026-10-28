"""Compose the six model layouts into reproducible loading/switching traces."""
from __future__ import annotations
import math
import time
from model_layouts import MODELS, build_layout, variants
from traffic_model import simulate_trace, make_switch_sequence
from activation_model import activation_traffic


def _activation_totals(result):
    # Public activation_model API; keep modelled payload and rounded transfers apart.
    t=result['total']
    return dict(activation_useful_read_bytes=t['useful_read_bytes'],
                activation_useful_write_bytes=t['useful_write_bytes'],
                activation_read_bytes=t['transfer_read_bytes'],
                activation_write_bytes=t['transfer_write_bytes'],
                activation_total_bytes=t['transfer_read_bytes']+t['transfer_write_bytes'])


def validate_config(c):
    required=('n_inferences','dwell','batch_size','main_line_bytes','cache_kib','scenarios',
              'preload_policies','sensitivity_line_bytes','sensitivity_cache_kib')
    if not all(k in c for k in required): raise ValueError('Onvolledige experimentconfiguratie')
    for key in ('n_inferences','dwell','batch_size','main_line_bytes'):
        if type(c[key]) is not int or c[key]<=0: raise ValueError('Ongeldige '+key)
    if not 2<=c['n_inferences']<=4096: raise ValueError('n_inferences moet tussen 2 en 4096 liggen')
    if any(type(x) is not int or x<0 for x in c['cache_kib']): raise ValueError('cache_kib')
    if len(set(c['cache_kib']))!=len(c['cache_kib']): raise ValueError('Dubbele cachegrootte')
    if set(c['preload_policies'])-{'none','start_mode','all_regions'}: raise ValueError('preload_policies')


def run_experiment(config, progress=lambda message: None):
    validate_config(config); started=time.perf_counter()
    storage=[]; initial_loading=[]; mode_payloads=[]; summary=[]; trace=[]; acts=[]
    act_cache={}
    def activation(bits,line):
        key=(bits,line,config['batch_size'])
        if key not in act_cache:
            act_cache[key]=activation_traffic(bits,line_bytes=line,batch_size=config['batch_size'])
        return act_cache[key]

    for model in MODELS:
        progress('MODELLERING: '+model['label'])
        for variant in variants(model['id']):
            layout=build_layout(model['id'],variant)
            base=dict(model_id=model['id'],model_label=model['label'],variant=variant,
                      primary_variant=variant!='full_source',group_size=model['group_size'])
            storage.append(dict(base,**layout['storage'],
                                n_modes=len(model['modes']),modes=model['modes'],
                                padded_slots=sum(x['padded_slots'] for x in layout['layer_rows']),
                                mx_scale_groups=sum(x['groups'] for x in layout['layer_rows']),
                                inactive_checkpoint_state_excluded=True))
            byid={r['id']:r for r in layout['regions']}
            line=config['main_line_bytes']
            for mode in model['modes']+['ALL']:
                regions=layout['regions'] if mode=='ALL' else [byid[r] for r in dict.fromkeys(x['region'] for x in layout['requests_by_mode'][str(mode)])]
                row=dict(base,mode=str(mode),line_bytes=line,load_kind='all_supported_modes' if mode=='ALL' else 'chosen_mode_cold',
                         requested_bytes=sum(r['size_bytes'] for r in regions),
                         read_bytes=sum(math.ceil(r['size_bytes']/line)*line for r in regions))
                for kind in ('codes','scales','aux'):
                    row[kind+'_payload_bytes']=sum(r['size_bytes'] for r in regions if r['kind']==kind)
                    row[kind+'_read_bytes']=sum(math.ceil(r['size_bytes']/line)*line for r in regions if r['kind']==kind)
                initial_loading.append(row)
            for mode in model['modes']:
                abits=mode if model['activation_policy']=='paired' else 32
                payload=layout['mode_payloads'][str(mode)]
                mode_payloads.append(dict(base,mode=mode,activation_bits=abits,**payload))
                if variant!='full_source':
                    for line in sorted(set([config['main_line_bytes']]+config['sensitivity_line_bytes'])):
                        a=activation(abits,line)
                        acts.append(dict(model_id=model['id'],model_label=model['label'],mode=mode,
                                         activation_bits=abits,line_bytes=line,batch_size=config['batch_size'],
                                         **_activation_totals(a)))

            combinations=[]
            for cache_kib in config['cache_kib']:
                for scenario in config['scenarios']:
                    for preload in config['preload_policies']:
                        combinations.append((config['main_line_bytes'],cache_kib,scenario,preload,'main'))
            for line in config['sensitivity_line_bytes']:
                if line==config['main_line_bytes']: continue
                for preload in config['preload_policies']:
                    combinations.append((line,config['sensitivity_cache_kib'],'alternating_extremes',preload,'line_sensitivity'))
            for line,cache_kib,scenario,preload,grid in combinations:
                sequence=make_switch_sequence(model['modes'],scenario,
                                              n_inferences=config['n_inferences'],dwell=config['dwell'])
                sim=simulate_trace(layout['regions'],layout['requests_by_mode'],sequence,
                                   line_bytes=line,cache_bytes=cache_kib*1024,preload_policy=preload)
                per=sim['per_inference']; totals=sim['totals']; pre=sim['initial_load']
                # Demand-first: first inference IS the first parameter loading.
                initial=per[0]['read_bytes'] if preload=='none' else pre['read_bytes']
                initial_useful=per[0]['requested_bytes'] if preload=='none' else pre['requested_bytes']
                rest=totals['read_bytes']-initial
                if initial+rest!=totals['read_bytes'] or rest<0: raise AssertionError('Initial loading double-count')
                activation_sum={k:0 for k in _activation_totals(activation(32,line))}
                for mode in sequence:
                    abits=int(mode) if model['activation_policy']=='paired' else 32
                    for k,v in _activation_totals(activation(abits,line)).items(): activation_sum[k]+=v
                trace_id='|'.join(map(str,[model['id'],variant,line,cache_kib,scenario,preload]))
                common=dict(base,trace_id=trace_id,scenario=scenario,line_bytes=line,cache_kib=cache_kib,
                            preload_policy=preload,grid=grid)
                row=dict(common,n_inferences=len(sequence),n_switches=sum(a!=b for a,b in zip(sequence,sequence[1:])),
                         mode_counts={str(m):sequence.count(m) for m in model['modes']},
                         initial_load_bytes=initial,initial_load_requested_bytes=initial_useful,
                         continuation_bytes=rest,weight_total_bytes=totals['read_bytes'],
                         weight_requested_bytes=totals['requested_bytes'],weight_hit_ratio=totals['hit_ratio'],
                         weight_line_requests=totals['line_requests'],weight_miss_lines=totals['miss_lines'],
                         first_inference_bytes=per[0]['read_bytes'],prefetch_bytes=pre['read_bytes'],
                         inference_only_bytes=sim['inference_totals']['read_bytes'],
                         avg_weight_bytes_per_inference=totals['read_bytes']/len(sequence),
                         cold_image_allocated_bytes=sim['layout']['unique_line_bytes'],
                         all_modes_fit_cache=sim['layout']['unique_line_bytes']<=cache_kib*1024,
                         **activation_sum)
                for kind in ('codes','scales','aux'):
                    row[kind+'_read_bytes']=totals['by_kind'].get(kind,{}).get('read_bytes',0)
                summary.append(row)
                # Full temporal traces for central scenarios, all cache sizes.
                if grid=='main' and scenario=='alternating_extremes':
                    cumulative=pre['read_bytes']
                    for i,entry in enumerate(per):
                        cumulative+=entry['read_bytes']
                        mode=int(entry['mode']); abits=mode if model['activation_policy']=='paired' else 32
                        trace.append(dict(common,index=i,mode=mode,activation_bits=abits,
                                          is_switch=i>0 and sequence[i]!=sequence[i-1],
                                          weight_read_bytes=entry['read_bytes'],
                                          weight_requested_bytes=entry['requested_bytes'],
                                          weight_hit_ratio=entry['hit_ratio'],
                                          cumulative_weight_bytes=cumulative,
                                          initial_prefetch_bytes=pre['read_bytes'],
                                          **_activation_totals(activation(abits,line))))

    result=dict(schema='mx_precision_switch_transfer_results_v1',config=config,models=MODELS,
                storage=storage,initial_loading=initial_loading,mode_payloads=mode_payloads,
                summary=summary,trace=trace,activations=acts,
                activation_geometry=list(act_cache.values()),
                simulation_seconds=time.perf_counter()-started,
                interpretation=dict(weight_interface='external parameter image to persistent read-only parameter cache',
                                    activation_interface='separate fused-stage materialization buffers; one input read and output write per stage',
                                    inference_only=True,training=False,hardware_measured=False,
                                    actual_tensor_decode=False,layout='prospective packed inference image',
                                    first_last_fp32=True,active_bn_included=True,
                                    code_scale_aux_separate=True,initial_plus_continuation_equals_total=True))
    return result
