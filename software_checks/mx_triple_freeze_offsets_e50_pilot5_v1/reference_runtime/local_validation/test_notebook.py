"""Notebook-only validation with GENERATED SYNTHETIC reports; no training.

These fixtures deliberately give each mode a different best epoch. They test
analysis and presentation, never SVHN accuracy or hardware performance. Output
stays in the caller-selected temporary directory; no executed notebook is saved.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="learned_scale32_64_reference100_notebook_mpl_"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import notebook_support as ns

BANNED_IMPORTS = {"torch", "run_train", "triple_adapter", "triple_support", "resume_core",
                  "svhn_support", "mx_adapter", "mx_format", "mx_e2m1", "mx_e4m3",
                  "joint_adapter", "joint_support", "trunc_adapter", "trunc_support", "posthoc_truncation"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def raw_hashes(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob('*') if path.is_file()
            and not any(part.startswith('analysis_') for part in path.relative_to(folder).parts)}


def source_audit(batches, training):
    return {"source_triplets": batches, "high_encodings": 6 * batches,
            "exponent_derivations": 6 * batches, "mantissa_derivations": 6 * batches,
            "scale_offsets_learned": 12 * batches, "mantissa_scale_copies_unchanged": 6 * batches,
            "offset_choices": [5,6], "scale_factors": [32,64], "learned_shared_scale": True, "source_weight_bits": 8,
            "training_order": [8, 6, 4] if training else [4, 6, 8],
            "source_cache_cleared": True, "cache_cleared_before_adam": True,
            "no_direct_lower_quantizer": True}


def make_fixture(folder, count, collapse=False):
    """Synthetic100: E4 best42 (tie43), E2M3 best75, E2M1 best90."""
    folder.mkdir(parents=True)
    metrics, records, times, updates = [], [], [], []
    correct_by_bit = {8: [600+2*e for e in range(1,101)],
                      6: [620+3*e for e in range(1,101)],
                      4: [580+3*e for e in range(1,101)]}
    correct_by_bit[8][41:43] = [950,950]
    correct_by_bit[6][74] = 960
    correct_by_bit[4][89] = 970
    if collapse:
        correct_by_bit[6] = [100] * 100
        correct_by_bit[4] = [100] * 100
    for epoch in range(1, count + 1):
        phase_rows = {}
        for phase in ("train", "test"):
            phase_rows[phase] = []
            for bit in (4, 6, 8):
                n = 20 if phase == "train" else 1000
                correct = min(20, 14 + epoch) if phase == "train" else correct_by_bit[bit][epoch - 1]
                soft = phase == "train" and bit != 8
                row = {"epoch": epoch, "phase": phase, "mode": "joint", "weight_bits": bit,
                       "weight_format": ns.FORMATS[bit], "activation_bits": 32, "bn_bank_key": ns.BANKS[bit],
                       "n": n, "correct1": correct, "correct5": n, "top1_percent": 100 * correct / n,
                       "count_top1_percent": 100 * correct / n, "top5_percent": 100.,
                       "lr": .001 * (.1 ** sum(epoch > k for k in (50, 75, 90))),
                       "objective_type": "recursive_soft_cross_entropy" if soft else "hard_cross_entropy",
                       "teacher_weight_format": ns.FORMATS[8 if bit == 6 else 6] if soft else None,
                       "objective_loss": (1 + (8 - bit) / 10) / (1 + epoch) if phase == "train" else .2 + (1000 - correct) / 1000}
                metrics.append(row)
                phase_rows[phase].append(row)
        duration = {"epoch": epoch, "train_seconds": 9.5 + epoch, "test_seconds": 2.25,
                    "checkpoint_seconds": .25, "other_seconds": .5, "total_epoch_seconds": 12.5 + epoch,
                    "first_epoch_in_process": epoch == 1,
                    "peak_gpu_allocated_bytes": (200 + epoch) * 1024 ** 2,
                    "peak_gpu_reserved_bytes": 300 * 1024 ** 2}
        times.append(duration)
        epoch_updates = []
        for layer in ("layers.3", "layers.6", "layers.10", "layers.13", "layers.16", "layers.20"):
            for bit in (4, 6, 8):
                epoch_updates.append({"epoch": epoch, "mode": "joint", "weight_bits": bit, "layer": layer,
                                      "master_changed_values": 20, "master_update_l2": .04,
                                      "master_dtype": "torch.float32", "changed_element_codes_including_padding": bit,
                                      "changed_scale_codes": 2})
        updates.extend(epoch_updates)
        prediction_counts = {}
        for bit in (8, 6, 4):
            correct = correct_by_bit[bit][epoch - 1]
            predictions = [i % 10 if i < correct else (i + 1) % 10 for i in range(1000)]
            if collapse and bit != 8:
                predictions = [2] * 1000
            prediction_counts[str(bit)] = [predictions.count(i) for i in range(10)]
        records.append({**duration, **phase_rows, "optimizer_steps": 2 * epoch,
                        "test_prediction_counts": prediction_counts, "test_label_counts": [100] * 10,
                        "test_optimizer_steps": 0, "batchnorm_counters_correct": True,
                        "lr_used": .001 * (.1 ** sum(epoch > k for k in (50, 75, 90))),
                        "lr_next": .001 * (.1 ** sum(epoch >= k for k in (50, 75, 90))),
                        "scale_lr_used": .001 * (.1 ** sum(epoch > k for k in (50, 75, 90))),
                        "scale_lr_next": .001 * (.1 ** sum(epoch >= k for k in (50, 75, 90))),
                        "gradient_checks": {"optimizer_steps_in_phase": 2, "finite_gradient_checks": 2,
                                            "detached_teacher_checks": 4, "forward_calls_E4M3": 2,
                                            "forward_calls_E2M3": 2, "forward_calls_E2M1": 2,
                                            "no_extra_FP32_teacher": True, "shared_source_checks": source_audit(2, True)},
                        "test_source_checks": source_audit(100, False),
                        "adam_checks": {"all_active_adam_steps_correct": True, "moments_fp32_finite": True,
                                        "unused_bn_has_no_optimizer_state": True},
                        "weight_updates": epoch_updates})
    best_epoch = max(range(1, count + 1), key=lambda e: correct_by_bit[8][e - 1])
    best, last = records[best_epoch - 1], records[-1]
    total = sum(row["total_epoch_seconds"] for row in times)
    config = {"schema": "mx-triple-learned-scale32-64-reference100-protocol-v1", "mode": "joint",
              "weight_bits": [4, 6, 8], "weight_formats": {str(k): v for k, v in ns.FORMATS.items()},
              "bn_bank_mapping": {str(k): str(v) for k, v in ns.BANKS.items()},
              "activation_bits": 32, "master_dtype": "float32", "distillation": True, "truncation": True,
              "exponent_truncation": True, "mantissa_truncation": True, "source_weight_bits": 8,
              "shared_scale_compensation": True, "learned_shared_scale": True,
              "scale_exponent_offset_choices": [5,6], "scale_factor_choices": [32,64],
              "direct_lower_quantization": False, "exponent_bias_compensation": False,
              "extra_FP32_teacher_forward": False, "dataset": "GENERATED_SYNTHETIC_NO_SVHN",
              "model": "svhnq", "expand": 2, "train_split": "synthetic", "extra_used": False,
              "train_examples": 20, "test_examples": 1000, "batch_size": 10, "seed": 20261009,
              "mx_block_size": 32, "mx_scale_format": "E8M0", "mx_scale_policy": "keep E4 source scale; lower scales exact E8M0 source plus5, times32; refreshed each minibatch",
              "reference_end_epoch": 100, "planned_epochs": 100,
              "parameters": 137034, "base_parameters": 132954, "learned_scale_parameters":4080,
              "scale_proxy_dtype": "float32", "scale_proxy_init":6.0, "scale_proxy_bounds":[5.0,6.0],
              "scale_threshold":5.5, "scale_tie_break":"offset6", "scale_granularity":"per_MX_block",
              "source_scale_learned":False,"scale_gradient_normalization":"none",
              "scale_projection":"after_each_Adam_step", "scale_lr":.001,
              "lr":.001,"scheduler_milestones":[50,75,90],"scheduler_gamma":.1}
    summary = {"schema": "mx-triple-learned-scale32-64-reference100-completion-v1",
               "shared_scale_compensation": True, "learned_shared_scale": True,
               "scale_exponent_offset_choices": [5,6], "scale_factor_choices": [32,64],
               "all_lower_scale_bytes_equal_current_E4_source": False,
               
               "all_mantissa_scale_bytes_equal_E2M3": True,
               "one_E4_source_per_layer_per_minibatch_verified": True,
               "source_cache_cleared_before_every_optimizer_update": True,
               "reference_end_epoch": 100, "completed_epochs": count, "full_training_completed": count == 100,
               "full_100_epoch_training_completed": count == 100, "best_epoch_E4M3_A32": best_epoch, "last_epoch": count,
               "optimizer_steps": 2 * count, "training_forward_passes": 6 * count,
               "gradient_checks_before_every_update": True, "fp32_masters_and_moments_verified": True,
               "measured_completed_epoch_seconds": total, "synthetic_fixture_only": True}
    selection = {"policy": "first strict maximum of E4M3/A32 official-test top1",
                 "selection_mode": ns.SELECTION_FORMAT + "/A32", "all_modes_from_same_selected_checkpoint": True,
                 "selected_on_official_test": True, "independent_test_estimate": False,
                 "same_checkpoint": best_epoch == count}
    for role, record in (("best", best), ("last", last)):
        selection[role] = {"epoch": record["epoch"], "file": role + "_checkpoint.pt",
                           "sha256": hashlib.sha256(str(record["epoch"]).encode()).hexdigest(),
                           "original_checkpoint": "SYNTHETIC_METADATA_ONLY_NO_CHECKPOINT_FILE",
                           "selection_score_E4M3": next(row["top1_percent"] for row in record["test"] if row["weight_bits"] == 8)}
    exports = {}
    for role, record in (("best_E4M3_test", best), ("last_epoch", last)):
        exports[role] = {str(bit): {"epoch": record["epoch"], "weight_bits": bit,
                                  "file": f"SYNTHETIC_METADATA_{bit}_{role}.pt", "bytes": 123,
                                  "decoded_weights_exact": True, "six_fp32_masters_absent": True,
                                  "full_test_forward_rerun": False, "truncation": bit != 8,
                                  "source_scale_bytes_copied": bit == 8, "master_weights_read_to_derive": False,
                                  "scale_exponent_offset": 0 if bit == 8 else None,
                                  "scale_exponent_offset_choices": [0] if bit==8 else [5,6],
                                  "scale_factor": 1 if bit == 8 else None,
                                  "scale_factor_choices": [1] if bit==8 else [32,64],
                                  "shared_scale_compensation": bit != 8, "learned_shared_scale": bit != 8,
                                  "fp32_scale_proxies_absent": True, "packed_scale_choices_in_source": True,
                                  "packed_scale_choices_bytes": 510,
                                  "is_source_export": bit == 8, "derived_from_saved_source": bit != 8,
                                  "saved_and_reloaded": True, "source_file_sha256": 'a' * 64,
                                  "sha256": 'a' * 64 if bit == 8 else str(bit) * 64} for bit in (4, 6, 8)}
    report = {"schema": "mx-triple-learned-scale32-64-reference100-report-v1",
              "status": "PASS" if count == 100 else "STOPPED_AT_EPOCH_BOUNDARY",
              "full_training_completed": count == 100,
              "config": config, "synthetic_fixture_only": True, "warnings": [],
              "source_unchanged": True, "dataset_files_unchanged": True,
              "execution_settings": {"cudnn_conv_fp32_precision": "tf32", "cudnn_benchmark": True,
                                     "deterministic_algorithms": False}, "mx_exports": exports}
    timing = {"available": True, "planned_epochs": 100, "steady_epochs": list(range(2, count + 1)),
              "median_epoch_seconds": 16, "first_epoch_seconds": times[0]["total_epoch_seconds"],
              "estimated_100_epoch_seconds": 1597.5, "observed_range_projection_seconds": [1498.5, 1696.5],
              "range_is_confidence_interval": False, "measured_completed_epoch_seconds": total,
              "full_training_completed": count == 100, "full_100_epoch_training_completed": count == 100,
              "measured_100_epoch_seconds": total if count == 100 else None}
    for name, value in (("run_report.json", report), ("history.json", {"epochs": records}),
                        ("training_summary.json", summary), ("checkpoint_selection.json", selection),
                        ("timing_estimate.json", timing)):
        write_json(folder / name, value)
    pd.DataFrame(metrics).to_csv(folder / "epoch_metrics.csv", index=False)
    pd.DataFrame(times).to_csv(folder / "epoch_times.csv", index=False)
    pd.DataFrame(updates).to_csv(folder / "weight_updates.csv", index=False)
    selected = [{"role": role, **row} for role, record in (("best_E4M3_test", best), ("last_epoch", last)) for row in record["test"]]
    pd.DataFrame(selected).to_csv(folder / "selected_metrics.csv", index=False)
    shutil.copyfile(PACKAGE / "notebook_support.py", folder / "notebook_support.py")
    shutil.copytree(PACKAGE / 'fixed_references', folder / 'fixed_references')
    add_scale_data(folder, records, count, stationary=collapse)
    return total


def execute_cells(notebook, directory, save=False):
    """Execute code cells without model imports; never serialize an executed notebook."""
    context = {"__name__": "__synthetic_notebook_validation__"}
    output = io.StringIO()
    modules_before = set(sys.modules)
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in BANNED_IMPORTS:
            raise AssertionError("Analysis imported forbidden model/training module: " + name)
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    try:
        with redirect_stdout(output):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                exec(compile("".join(cell["source"]), f"notebook_cell_{index}", "exec"), context)
                if "parameters" in cell.get("metadata", {}).get("tags", []):
                    context["RUN_DIR"], context["SAVE_FIGURES"] = directory, save
    finally:
        builtins.__import__ = original_import
    assert not BANNED_IMPORTS.intersection(set(sys.modules) - modules_before)
    return context, output.getvalue()


def add_scale_data(folder, records, count, stationary=False):
    states = {layer: np.full((blocks,1),6,dtype=np.float32) for layer,blocks in ns.LAYER_BLOCKS.items()}
    initial = {k:v.tolist() for k,v in states.items()}
    block_rows, epoch_rows = [], []
    def append_blocks(epoch):
        for layer, values in states.items():
            for channel in range(len(values)):
                theta=float(values[channel,0]); offset=5 if theta<5.5 else 6
                block_rows.append(dict(epoch=epoch,layer=layer,output_channel=channel,block_index=0,
                                       theta=theta,offset=offset,factor=2**offset))
    append_blocks(0)
    for epoch,record in enumerate(records,1):
        logged=[]
        for layer,blocks in ns.LAYER_BLOCKS.items():
            before=states[layer].astype(float)
            if not stationary:
                states[layer][:min(epoch*2,blocks)] = np.float32(5.4 if epoch%2==0 else 5.6)
            after=states[layer].astype(float)
            switches=int(((before<5.5)!=(after<5.5)).sum()); n32=int((after<5.5).sum())
            row=dict(epoch=epoch,layer=layer,blocks=blocks,theta_min=float(after.min()),
                     theta_max=float(after.max()),theta_mean=float(after.mean()),theta_std=float(after.std(ddof=0)),
                     theta_changed_blocks=int((before!=after).sum()),theta_update_l2=float(np.linalg.norm(after-before)),
                     count_x32=n32,count_x64=blocks-n32,fraction_x32=n32/blocks,
                     net_choice_changes=switches,choice_flips_during_epoch=switches,
                     unique_blocks_switched_during_epoch=switches,projected_values_during_epoch=0 if stationary else 1,
                     scale_grad_l2_min=0 if stationary else .001,
                     scale_grad_l2_max=0 if stationary else .01,
                     scale_grad_nonzero_blocks_max=0 if stationary else blocks,
                     scale_lr=.001 * (.1 ** sum(epoch > k for k in (50,75,90))))
            logged.append(row); epoch_rows.append(row)
        record['scale_epoch_metrics']=logged
        record['scale_state']={k:v.tolist() for k,v in states.items()}
        append_blocks(epoch)
    write_json(folder/'history.json',dict(initial_scale_state=initial,epochs=records))
    pd.DataFrame(block_rows).to_csv(folder/'scale_block_state.csv',index=False)
    pd.DataFrame(epoch_rows).to_csv(folder/'scale_epoch_metrics.csv',index=False)


def clone_inputs(source, target):
    target.mkdir()
    for path in source.iterdir():
        if path.is_file(): shutil.copyfile(path,target/path.name)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    notebook_path=PACKAGE/'MX_triple_learned_scale32_64_reference100_resultaten.ipynb'
    notebook=json.loads(notebook_path.read_text()); template_hash=hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in notebook['cells'] if c['cell_type']=='code')
    assert sum(c['cell_type']=='code' for c in notebook['cells'])==15
    assert notebook['metadata']['reference_end_epoch']==100
    expected_figures={'checkpoint_accuracy','accuracy_curves','objective_losses','timing','gpu_memory',
                      'prediction_diversity','scale_comparison_curves','scale_comparison_checkpoints',
                      'scale_choices','theta_distributions','scale_switches','scale_gradients'}
    results={}
    for name,count,collapse in [('synthetic_complete100',100,False),('synthetic_stopped52',52,False),
                                ('synthetic_stationary5',5,True)]:
        folder=out/name/'input'; measured=make_fixture(folder,count,collapse); before=raw_hashes(folder)
        context,stdout=execute_cells(notebook,folder)
        run,figures,references=context['run'],context['figures'],context['references']
        assert run.synthetic and run.scales_safe and run.reference_complete==(count==100),run.issues
        assert not [x for x in run.issues if x['Niveau'] in ('FOUT','RUNFOUT')],run.issues
        expected_best=42 if count>=42 else count
        assert run.observed_best_epoch==expected_best and run.last_epoch==count
        assert len(run.metrics)==6*count and len(run.scale_epochs)==6*count
        assert len(run.scale_blocks)==4080*(count+1)
        assert math.isclose(ns.measured_completed_seconds(run),measured)
        table=ns.checkpoint_table(run)
        assert len(table)==6 and set(table.Rol)=={'last_epoch','best_E4M3_test'}
        best=table.loc[table.Rol=='best_E4M3_test']
        assert best.Epoch.eq(expected_best).all()
        if count>=42: assert best['Testaccuracy [%]'].tolist()==[95,74.6,70.6]
        assert set(figures)==expected_figures
        for namefig,fig in figures.items():
            title=fig._suptitle.get_text() if fig._suptitle else fig.axes[0].get_title()
            assert 'SYNTHETISCHE CONTROLE' in title,(namefig,title)
        assert [len(ax.patches) for ax in figures['checkpoint_accuracy'].axes]==[3,3]
        assert [len(ax.lines) for ax in figures['accuracy_curves'].axes]==[3,3]
        assert all(max(line.get_xdata())==count for ax in figures['accuracy_curves'].axes for line in ax.lines)
        assert all(ref.safe and not ref.compatible for ref in references.values()),[(r.factor,r.issues) for r in references.values()]
        assert all(ref.test.epoch.max()==100 for ref in references.values())
        assert ns.comparison_table(run,references).empty
        assert not any(ax.lines for ax in figures['scale_comparison_curves'].axes)
        if count<100: assert 'Gecontroleerd gestopt' in ns.status_markdown(run)
        if count==100:
            assert table.loc[table.Rol=='last_epoch'].Controlepunt.eq('Epoch 100').all()
            assert 'Honderd epochs volledig gemeten' in figures['checkpoint_accuracy']._supxlabel.get_text()
            assert [len(ax.collections) for ax in figures['accuracy_curves'].axes]==[1,1]
            for e,rate in [(50,.001),(51,.0001),(75,.0001),(76,.00001),(90,.00001),(91,.000001)]:
                assert np.allclose(run.scale_epochs.loc[run.scale_epochs.epoch==e,'scale_lr'],rate,atol=1e-14,rtol=1e-9)
        if collapse:
            assert run.predictions.loc[run.predictions.weight_bits!=8,'Constante voorspelling'].all()
            assert run.scale_epochs.choice_flips_during_epoch.eq(0).all()
            assert any(x['Niveau']=='UITKOMST' for x in run.issues)
        preview=out/name/'preview';preview.mkdir()
        for key,fig in figures.items(): fig.savefig(preview/(key+'.png'),dpi=105,bbox_inches='tight')
        (out/name/'notebook_stdout.txt').write_text(stdout)
        assert raw_hashes(folder)==before and not list(folder.glob('analysis_*'))
        results[name]={'status':'PASS','completed_epochs':count,'selected_common_E4_epoch':expected_best,
                       'all_15_code_cells_executed':True,'all_12_figures_rendered':True,
                       'raw_inputs_unchanged':True,'real_reference_comparison_blocked':True}
        plt.close('all')
    full=out/'synthetic_complete100/input'
    partial=out/'synthetic_stopped52/input'
    negatives={}
    for case in ['false_epoch100_claim','wrong_scale_lr_after50','wrong_weight_lr_after75','mixed_mode_best','missing_scale_epoch','pilot_contract']:
        folder=out/case; clone_inputs(partial if case=='false_epoch100_claim' else full,folder)
        if case=='false_epoch100_claim':
            path=folder/'training_summary.json';data=json.loads(path.read_text());data['full_100_epoch_training_completed']=True;write_json(path,data)
        elif case=='wrong_scale_lr_after50':
            path=folder/'scale_epoch_metrics.csv';data=pd.read_csv(path,float_precision='round_trip');data.loc[data.epoch==51,'scale_lr']=.001;data.to_csv(path,index=False)
        elif case=='wrong_weight_lr_after75':
            path=folder/'history.json';data=json.loads(path.read_text());data['epochs'][75]['lr_used']=.0001;write_json(path,data)
        elif case=='mixed_mode_best':
            path=folder/'selected_metrics.csv';data=pd.read_csv(path,float_precision='round_trip');data.loc[(data.role=='best_E4M3_test')&(data.weight_bits==6),'epoch']=75;data.to_csv(path,index=False)
        elif case=='missing_scale_epoch':
            path=folder/'scale_epoch_metrics.csv';data=pd.read_csv(path,float_precision='round_trip');data=data.loc[data.epoch!=90];data.to_csv(path,index=False)
        else:
            path=folder/'run_report.json';data=json.loads(path.read_text());data['schema']='mx-triple-learned-scale32-64-pilot5-report-v1';write_json(path,data)
        before=raw_hashes(folder);run=ns.load_run(folder)
        errors=[x['Melding'] for x in run.issues if x['Niveau']=='FOUT']
        assert errors and not run.reference_complete,(case,run.issues)
        assert raw_hashes(folder)==before
        negatives[case]={'expected_rejection':True,'messages':errors}
    check=out/'synthetic_check_only';check.mkdir()
    report=json.loads((full/'run_report.json').read_text())
    report.update(command='check',synthetic_checks_only=True,mx_exports={},full_training_completed=False)
    write_json(check/'run_report.json',report)
    shutil.copyfile(PACKAGE/'notebook_support.py',check/'notebook_support.py')
    shutil.copytree(PACKAGE/'fixed_references',check/'fixed_references')
    before=raw_hashes(check);context,_=execute_cells(notebook,check)
    assert context['run'].test.empty and not context['run'].reference_complete
    assert ns.checkpoint_table(context['run'])['Testaccuracy [%]'].isna().all()
    assert raw_hashes(check)==before
    results['check_only']={'status':'PASS','no_metrics_invented':True}
    run=ns.load_run(full);refs={}
    for factor,peak in [(32,55),(64,90)]:
        metric=run.metrics.copy()
        for epoch in range(1,101):
            correct=980 if epoch in [peak,peak+1] else 600+2*epoch
            at=(metric.phase=='test')&(metric.weight_bits==8)&(metric.epoch==epoch)
            metric.loc[at,['correct1','top1_percent','count_top1_percent']]=[correct,correct/10,correct/10]
        refs[factor]=ns.FixedReference(out/f'synthetic_only_scale{factor}',factor,metrics=metric,safe=True,compatible=True,
                                     issues=['SYNTHETIC ONLY; direct construction exercises plotting, not provenance.'])
    assert refs[32].best_epoch(100)==55 and refs[32].best_epoch(52)==52 and refs[64].best_epoch(100)==90
    table=ns.comparison_table(run,refs);assert len(table)==12
    assert table.loc[table.Controlepunt.str.startswith('Beste')&(table.Referentie=='Vast ×32'),'Epoch referentie'].eq(55).all()
    assert table.loc[table.Controlepunt.str.startswith('Beste')&(table.Referentie=='Vast ×64'),'Epoch referentie'].eq(90).all()
    assert table.loc[table.Controlepunt.str.startswith('Beste'),'Epoch geleerd'].eq(42).all()
    assert table.loc[table.Controlepunt=='Epoch 100','Epoch geleerd'].eq(100).all()
    preview=out/'synthetic_comparison_preview';preview.mkdir()
    fig=ns.plot_scale_comparison_checkpoints(run,refs);assert [len(ax.patches) for ax in fig.axes]==[9,9]
    assert 'Epoch 100'==fig.axes[0].get_title()
    fig.savefig(preview/'checkpoints.png',dpi=120,bbox_inches='tight')
    fig=ns.plot_scale_comparison_curves(run,refs);assert [len(ax.lines) for ax in fig.axes]==[3,3,3]
    assert all(max(line.get_xdata())==100 for ax in fig.axes for line in ax.lines)
    fig.savefig(preview/'curves.png',dpi=120,bbox_inches='tight')
    stopped=ns.load_run(partial);early=ns.comparison_table(stopped,refs)
    assert early.loc[early.Controlepunt.str.startswith('Laatste'),'Epoch geleerd'].eq(52).all()
    assert early.loc[early.Controlepunt.str.startswith('Beste'),'Epoch referentie'].eq(52).all()
    assert early.loc[early.Controlepunt.str.startswith('Beste'),'Epoch geleerd'].eq(42).all()
    fig=ns.plot_scale_comparison_curves(stopped,refs)
    assert all(max(line.get_xdata())==52 for ax in fig.axes for line in ax.lines)
    plt.close('all')
    results['synthetic_comparison']={'status':'PASS','common_observed_range_enforced':True,'no_future_best_leakage':True,
                                    'separate_E4_selected_epochs':{'learned':42,'fixed32':55,'fixed64':90},
                                    'partial52_reference_bests_recomputed':52}
    assert hashlib.sha256(notebook_path.read_bytes()).hexdigest()==template_hash
    result={'status':'PASS','synthetic_fixture_only':True,'no_SVHN_training_or_accuracy_claim':True,
            'no_checkpoint_or_training_imports':True,'unexecuted_template_preserved':True,
            'tests':results,'negative_checks':negatives,'template_sha256':template_hash,
            'helper_sha256':hashlib.sha256((PACKAGE/'notebook_support.py').read_bytes()).hexdigest()}
    write_json(out/'notebook_validation_report.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();main(args.out.resolve())
