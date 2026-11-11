"""One hundred epochs: an independent absolute E8M0 scale for the E2 modes.

The E4M3 source retains its dynamic amax scale and original encoding.
A FP32 absolute log2-scale proxy phi is initialized once to the initial
E4 exponent plus six. Thereafter no relation to the current E4 scale is imposed.
E2M3 and E2M1 share round_half_up(phi), while their element codes still use
exponent-MSB and mantissa truncation from one common E4 source.
A32, FP32 masters, recursive distillation and the original100-epoch LR schedule
are unchanged. Weights and phi learn through epoch100 without a freeze.
Every run starts from fresh original weights; only this run contract may resume.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
import svhn_support as s
import trunc_support as t
import trunc_adapter as a

ROOT = Path(__file__).resolve().parent


def config(mode: str = 'joint') -> dict:
    s.require(mode == 'joint', 'Deze proef heeft exact drie gezamenlijke MX-modi uit één E4M3-bron.')
    return {
        'schema': 'mx-triple-independent-e2-scale-reference100-protocol-v1',
        'purpose': 'one hundred complete train-only epochs from fresh weights; learn an independent absolute E8M0 scale shared by the E2 modes through epoch100 without a freeze, initialized once to the initial E4 scale x64; unchanged E4 source and exponent/mantissa bit truncation',
        'mode': mode, 'weight_format': t.weight_format(mode), 'weight_bits': [4, 6, 8],
        'weight_formats': {'8': 'MXFP8_E4M3', '6': 'MXFP6_E2M3', '4': 'MXFP4_E2M1'},
        'codec_revision': {'8': 'e4m3-exact-binary-v2', '6': 'unchanged-E2M3-decode', '4': 'unchanged-E2M1-decode'},
        'weight_preprocess': 'original W32 G(W); detached mean(abs(W)), tanh/max normalization',
        'activation_bits': 32, 'activation_policy': 'original A32/ReLU in ALL THREE modes',
        'master_dtype': 'float32', 'gradient_dtype': 'float32', 'adam_moments_dtype': 'float32',
        'mx_block_size': 32, 'mx_scale_format': 'E8M0',
        'mx_group_axis': 'flatten Cin*Kh*Kw within each Cout, zero-pad tail',
        'mx_scale_policy': 'unchanged dynamic E4M3 source scale floor(log2(amax))-8; E2M3 and E2M1 share independent absolute E8M0 exponent round_half_up(phi); initialize once from initial E4 exponent+6, never recompute phi from E4; E8M0 finite exponents -127..127; reject nonfinite FP32 decode without clipping or wrap',
        'mx_rounding': 'E4M3 source nearest ties-to-even; finite FP32 source, scalar subnormal flush; lower modes exact bit mappings without additional rounding',
        'ste_policy': 'identity wrt G(W) all three modes; phi gradient ln(2)*sum_real_block(grad_q*q) from E2M3/E2M1; no normalization; no direct E4 phi gradient; source codes/scales detached; retain G(W) derivatives',
        'numeric_kernel': 'FP32 F.conv2d; codec FP64 intermediates; not native low-bit kernels',
        'truncation': True, 'mantissa_truncation': True, 'exponent_truncation': True,
        'independent_E2_shared_scale': True, 'learned_shared_scale': True,
        'scale_parameterization': 'independent_absolute_log2_E2_scale',
        'scale_proxy_dtype': 'float32', 'scale_proxy_name': 'phi',
        'scale_proxy_initialization': 'initial_E4_E8M0_exponent_plus_6_once_before_optimizer',
        'initial_relative_offset': 6, 'initial_relative_factor': 64,
        'scale_proxy_bounds': [-127.0, 127.0], 'scale_tie_break': 'higher_exponent',
        'scale_rounding': 'nearest_integer_ties_up; floor(phi)+(phi-floor(phi)>=0.5)',
        'e2_scale_bits_per_block': 8, 'e2_scale_bytes': 4080,
        'relative_factor_constrained_after_initialization': False,
        'fp32_decode_policy': 'reject overflow; exponents126/127 are valid E8M0 but not all E2 element codes remain finite in FP32',
        'observed_relative_offset_is_diagnostic_only': True,
        'scale_gradient': 'ln(2)*sum_real_block(grad_q*q); accumulate E2M3 and E2M1 only',
        'scale_gradient_normalization': 'none', 'scale_projection': 'after_each_Adam_step',
        'scale_granularity': 'per_MX_block', 'source_scale_learned': False,
        'quantization_source': 'one current E4M3 encoding per MX layer and minibatch, shared across all three modes',
        'source_weight_bits': 8,
        'exponent_mapping': 's e3 e2 e1 e0 m2 m1 m0 -> s e3 e2 m2 m1 m0; target bias1; no bias compensation',
        'mantissa_mapping': 'derived E2M3 uint8 codes >> 2; exact same per-block learned scale as E2M3',
        'source_cache_policy': 'detached integer source, absolute lower scale and derived codes; phi version fixed within triplet; cached q saved for backward; cleared before Adam',
        'direct_lower_quantization': False, 'exponent_bias_compensation': False,
        'distillation': True,
        'loss': 'E4M3 hard CE + E2M3 soft CE from detached E4M3 + E2M1 soft CE from detached E2M3; T=1; gradients summed, no division by 3',
        'passes_per_training_batch': 3, 'optimizer_steps_per_training_batch': 1,
        'training_mode_order': [8, 6, 4], 'test_mode_order': [4, 6, 8],
        'internal_upstream_selectors': [4, 6, 8], 'legacy_abits_and_wbits': 32,
        'model_bank_structure': s.BITS, 'bn_bank_mapping': {'8': '8', '6': '32', '4': '4'},
        'batchnorm_policy': 'normal training updates of three mode banks E4M3=8,E2M3=32,E2M1=4; parameters and running stats learn per mode; eval frozen; keys are storage IDs, all A32; banks1/2 unused',
        'extra_FP32_teacher_forward': False,
        'dataset': 'svhn', 'train_split': 'train', 'test_split': 'test', 'extra_used': False,
        'train_examples': 73257, 'test_examples': 26032, 'validation_examples': 0,
        'initialization_policy': 'fresh seed20261009; same untrained state as prior model construction; no warmstart',
        'model': 'svhnq', 'expand': 2, 'parameters': 137034,
        'base_parameters': 132954, 'learned_scale_parameters': 4080,
        'planned_epochs': 100, 'reference_end_epoch': 100,
        'scale_learning_end_epoch': 100, 'scale_freeze_epoch': None,
        'batch_size': 128, 'optimizer': 'adam', 'lr': .001, 'scale_lr': .001, 'weight_decay': 0.0,
        'optimizer_groups': ['weights', 'scale_proxy'],
        'scheduler_milestones': [50, 75, 90], 'scheduler_gamma': .1,
        'workers': 0, 'pin_memory': True, 'drop_last': False,
        'train_shuffle': True, 'test_shuffle': False, 'seed': 20261009,
        'evaluation_policy': 'all three modes on complete official test after every epoch; hard CE for each',
        'selection_policy': 'first strict maximum of E4M3/A32 official-test top1',
        'reporting_policy': 'all three modes at epoch100 or actual final epoch and at the same E4M3-best checkpoint; compare only matched observed epochs against historical references; partial runs never claim epoch100 completion',
        'backend_policy': 'github: cudnn.benchmark=True; no deterministic override',
        'rng_policy': 'global streams as upstream; sequential dropout draws per mode; saved at epoch boundaries',
        'checkpoint_policy': 'immutable FP32 training state per epoch and selected best/last copies',
        'export_policy': 'one E4M3 source plus one independent absolute E8M0 lower scale byte per group and BN auxiliary state; derive lower exports only from saved source codes and absolute lower scales; no MX masters or FP32 phi in compact exports; resume retains phi and Adam',
        'exact_historical_paper_reproduction_proven': False,
    }



NOTEBOOK = 'MX_triple_independent_e2_scale_reference100_resultaten.ipynb'


REFERENCE_DIRECTORIES = ('fixed_scale64_reference', 'relative_scale10_reference',
                         'relative_scale10_full100_reference', 'independent_pilot_reference')


def load_initialization_reference(directory=None):
    """Pinned historical relative-scale run; raw fields remain unchanged."""
    directory = Path(directory) if directory is not None else ROOT / 'relative_scale10_full100_reference'
    expected_epochs = 100 if directory.name == 'relative_scale10_full100_reference' else 5
    meta = json.loads((directory / 'REFERENCE.json').read_text(encoding='utf-8'))
    for name, digest in meta['files_sha256'].items():
        path = (directory / name).resolve()
        s.require(path.is_relative_to(directory.resolve()) and path.is_file()
                  and s.legacy.sha256(path) == digest, 'Historische referentie gewijzigd: ' + name)
    report = json.loads((directory / 'run_report.json').read_text(encoding='utf-8'))
    history = json.loads((directory / 'history.json').read_text(encoding='utf-8'))
    s.require(report['status'] == 'PASS' and report['config']['scale_proxy_init'] == 6.0,
              'Verwacht de geslaagde historische relatieve tienkeuzerun met start x64.')
    s.require([r['epoch'] for r in history['epochs']] == list(range(1, expected_epochs + 1)),
              'Historische referentie heeft onvolledige of onverwachte epochs.')
    s.require(history['initial_base_model_sha256'] == report['initial_base_model_sha256'],
              'Historische referentie heeft verschillende oorspronkelijke modelidentiteiten.')
    if expected_epochs == 100:
        c = report['config']
        s.require(report.get('full_training_completed') is True
                  and c['planned_epochs'] == c['scale_learning_end_epoch'] == 100
                  and c['scale_freeze_epoch'] is None,
                  'De volledige relatieve referentie moet100 epochs met leerbare schalen bevatten.')
    return dict(directory=directory, meta=meta, report=report, history=history,
                kind='relative10_full100' if expected_epochs == 100 else 'relative10_pilot5')


def archive_initialization_reference(out):
    for name in REFERENCE_DIRECTORIES:
        source, destination = ROOT / name, out / name
        s.require(source.is_dir(), 'Een vastgelegde historische referentie ontbreekt: ' + name)
        if destination.exists():
            for path in source.rglob('*'):
                if path.is_file():
                    other = destination / path.relative_to(source)
                    s.require(other.is_file() and other.read_bytes() == path.read_bytes(),
                              'Bestaande gearchiveerde referentie wijkt af: ' + str(other))
        else:
            shutil.copytree(source, destination)


def capture_source_scale_state(model):
    """Detached diagnostics only; never write or reinitialize phi."""
    diagnostics = a.scale_diagnostics(model)
    return {name: [[int(e) + 127 for e in row] for row in diagnostics[name]['e4_exponents']]
            for name in a.TARGET_LAYERS}


def validate_initialization(model, c, reference, report):
    import torch
    s.require(c['initial_relative_offset'] == a.INITIAL_RELATIVE_OFFSET == 6,
              'Deze proef moet eenmaal beginnen bij de aanvankelijke E4-schaal x64.')
    state = a.validate_scale_parameters(model)
    diagnostics = a.scale_diagnostics(model)
    s.require(state['total_blocks'] == 4080 and state['parameter_tensors'] == 6,
              'Ongeldig aantal onafhankelijke E2-schaalparameters.')
    for name, record in diagnostics.items():
        phi = torch.tensor(record['phi'], dtype=torch.float64)
        source = torch.tensor(record['e4_exponents'], dtype=torch.int32)
        lower = torch.tensor(record['e2_exponents'], dtype=torch.int32)
        s.require(bool((phi == source.to(torch.float64) + 6).all())
                  and bool((lower == source + 6).all()),
                  'De aanvankelijke absolute E2-schaal is niet exact de E4-beginschaal x64: ' + name)
    base = s.legacy.state_digest(a.base_state_dict(model))
    s.require(base == reference['history']['initial_base_model_sha256'],
              'FP32-gewichten of BN-initialisatie wijken af van de historische x64-referentie.')
    report['initialization_verification'] = dict(
        initial_relative_offset=6, initial_relative_factor=64,
        all4080_groups_start_at_initial_E4_scale_x64=True,
        initial_base_model_sha256=base, same_base_weights_and_BN_as_relative_reference=True,
        scale_parameterization_is_the_intended_training_change=True,
        source_scale_is_still_dynamic=True, absolute_E2_scale_independent_after_initialization=True,
        relative_factor_is_not_constrained=True)


def write_initialization_comparison(ctx, out, trace):
    if ctx.contract['synthetic_fixture_only'] and not hasattr(ctx, 'initialization_reference'):
        result = dict(synthetic_fixture_only=True, comparison_performed=False,
                      reason='Synthetic software fixture is not compared with real research data.')
        s.write_json(out / 'scale_parameterization_reference_report.json', result)
        ctx.report['scale_parameterization_reference'] = result
        return result
    reference = getattr(ctx, 'initialization_reference', None) or load_initialization_reference()
    old = reference['history']
    s.require(trace['initial_base_model_sha256'] == old['initial_base_model_sha256'],
              'De twee runs vertrekken niet van dezelfde basisgewichten/BN.')
    rows, matched, unmatched = [], [], []
    old_by_epoch = {record['epoch']: record for record in old['epochs']}
    for record in trace['epochs']:
        counterpart = old_by_epoch.get(record['epoch'])
        if counterpart is None:
            unmatched.append(record['epoch'])
            continue
        matched.append(record['epoch'])
        for phase in ('train', 'test'):
            s.require(record['coverage'][phase]['order_sha256'] == counterpart['coverage'][phase]['order_sha256'],
                      'Samplevolgorde verschilt van de historische x64-referentie.')
        bybit = {row['weight_bits']: row for row in counterpart['test']}
        for row in record['test']:
            other = bybit[row['weight_bits']]
            s.require(row['n'] == other['n'], 'Verschillend aantal testbeelden.')
            rows.append(dict(epoch=record['epoch'], weight_bits=row['weight_bits'],
                weight_format=row['weight_format'], independent_top1_percent=row['top1_percent'],
                relative_scale10_top1_percent=other['top1_percent'],
                difference_percentage_points=row['top1_percent']-other['top1_percent'],
                independent_correct1=row['correct1'], relative_scale10_correct1=other['correct1'],
                test_examples=row['n']))
    s.require(bool(rows), 'Geen gemeenschappelijke historische epochs voor de vergelijking.')
    s.write_csv(out / 'scale_parameterization_comparison.csv', rows)
    result = dict(synthetic_fixture_only=ctx.contract['synthetic_fixture_only'], comparison_performed=True,
        matched_epochs=matched, unmatched_epochs=unmatched,
        historical_reference_kind=reference['kind'], historical_reference_completed_epochs=len(old['epochs']),
        historical_reference_directory=reference['directory'].name,
        missing_historical_epochs_are_not_extrapolated=True,
        initial_base_model_sha256=trace['initial_base_model_sha256'],
        initial_base_weights_and_BN_match=True, all_sample_orders_match=True,
        initial_relative_factor_both_runs=64, parameterization_changed=True,
        new_parameterization='absolute E2 exponent independent of current E4 scale',
        reference_parameterization='relative E2 offset0..9 from current E4 scale',
        comparison_is_historical=True, exact_CUDA_repeat_not_claimed=True,
        source_reference_sha256=s.legacy.sha256(reference['directory']/'REFERENCE.json'))
    s.write_json(out / 'scale_parameterization_reference_report.json', result)
    ctx.report['scale_parameterization_reference'] = result
    return result


def write_notebook(out):
    notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'parameters' in cell.get('metadata', {}).get('tags', []):
            cell['source'] = ['from pathlib import Path\n', f'RUN_DIR = Path({str(out.resolve())!r})\n',
                              'SAVE_FIGURES = False\n']
    target = out / NOTEBOOK
    if target.exists():
        s.require(json.loads(target.read_text(encoding='utf-8')) == notebook,
                  'Bestaande notebook verschilt; niets overschrijven.')
    else:
        s.write_json(target, notebook)
    raw = (ROOT / 'notebook_support.py').read_bytes()
    target = out / 'notebook_support.py'
    if target.exists():
        s.require(target.read_bytes() == raw, 'Bestaande notebookhelper verschilt; niets overschrijven.')
    else:
        with target.open('xb') as handle: handle.write(raw)


def dependencies(out: Path, log) -> None:
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    (out / 'pip_freeze.txt').write_text(freeze.stdout + freeze.stderr, encoding='utf-8')
    s.require(freeze.returncode == 0, 'pip freeze mislukt; zie pip_freeze.txt.')
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True,
                           text=True, encoding='utf-8', errors='replace')
    (out / 'pip_check.txt').write_text(check.stdout + check.stderr, encoding='utf-8')
    s.require(check.returncode == 0, 'Afhankelijkheidsconflict; zie pip_check.txt. Geen automatische upgrades.')
    log('pip check: ' + check.stdout.strip())


def create_context(args, out: Path, report: dict, log, check_only: bool = False):
    package_hashes = s.verify_package()
    c = config(args.mode)
    reference = load_initialization_reference()
    archive_initialization_reference(out)
    source, source_info = s.legacy.prepare_source(out)
    report.update(config=c, source=source_info, package_sha256=package_hashes,
                  interpreter=sys.executable)
    s.write_json(out / 'training_change_manifest.json',
                 json.loads((ROOT / 'TRAINING_CHANGE_MANIFEST.json').read_text(encoding='utf-8')))
    dependencies(out, log)
    rt = s.runtime(c['seed'], args.device)
    modules = s.import_source(source)
    report.update(environment=rt.environment, execution_settings=rt.settings)
    log('GPU: ' + rt.environment['device_name'])
    log('Originele backend: cudnn.benchmark=True; geen bitexacte CUDA-hervatgarantie.')
    model = None
    if check_only:
        initialization_rng = rt.core.capture_rng({}, rt.device)
        import test_mx
        import test_e4m3
        import test_learned_scale
        report['codec_tests'] = {'E2M3': {}, 'E4M3': {}}
        try:
            report['codec_tests']['E2M3'] = test_mx.run_tests(modules, str(rt.device))
            test_e4m3.run_tests(str(rt.device), report=report['codec_tests']['E4M3'])
        except BaseException as exc:
            report['codec_tests'].update(status='FAIL', error=repr(exc))
            raise
        else:
            report['codec_tests']['status'] = 'PASS'
        finally:
            s.write_json(out / 'codec_report.json', report['codec_tests'])
        report['synthetic_training_checks'] = {}
        try:
            test_learned_scale.run_tests(modules, rt, source, log, checks=report['synthetic_training_checks'])
        except BaseException as exc:
            report['synthetic_training_checks'].update(status='FAIL', error=repr(exc))
            raise
        else:
            report['synthetic_training_checks']['status'] = 'PASS'
        finally:
            s.write_json(out / 'trunc_training_check_report.json', report['synthetic_training_checks'])
        _, forward_hash = t.load_joint_forward(source, modules.utils, log)
        report.update(forward_ast_sha256=forward_hash, dataset_downloaded=False,
                      synthetic_checks_only=True)
        report['notebook_dependencies'] = {}
        for name in ['pandas', 'matplotlib', 'ipykernel']:
            try:
                report['notebook_dependencies'][name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                report['notebook_dependencies'][name] = 'MISSING: notebook only'
        # Codec checks use their own seed; restore the actual experiment seed
        # before comparing fresh FP32 weights and BN with the historical full run.
        rt.core.restore_rng(initialization_rng, {}, rt.device)
        initial_model = t.build_model(modules, c['mode'], rt.device)
        validate_initialization(initial_model, c, reference, report)
        log('CHECK PASS: onafhankelijke absolute E2-schaal; start exact aanvankelijk x64; exponent- en mantissatruncatie; identity-mastergradient, absolute log-schaalgradient, E8M0-projectie en bronexport met twee schaalbanken. Geen SVHN-training.')
        return None

    start_data = time.perf_counter()
    info = s.ensure_dataset(args.data_root.resolve(), modules.data.SVHNMIX, args.download, log)
    report['data_preparation_seconds_including_download_and_checksums'] = time.perf_counter() - start_data
    report['dataset_files'] = info
    modules.data.data_paths['svhn'] = str(args.data_root.resolve())
    start_load = time.perf_counter()
    train_tf = modules.data.get_transform('svhn', 'train')
    test_tf = modules.data.get_transform('svhn', 'val')
    s.require(c['train_split'] == 'train' and c['extra_used'] is False,
              'Deze runner is uitsluitend voor de volledige train-split, zonder extra.')
    train = modules.data.get_dataset('svhn', c['train_split'], train_tf, download=False)
    test = modules.data.get_dataset('svhn', 'val', test_tf, download=False)
    s.require(train.split == 'train' and test.split == 'test', 'Onverwachte datasetsplit.')
    s.require(set(info['files']) == {'train_32x32.mat', 'test_32x32.mat'},
              'Train-only mag alleen train en test laden of controleren.')
    selection = {'train': s.validate_dataset(train, c['train_examples'], 'SVHN train-only'),
                 'test': s.validate_dataset(test, c['test_examples'], 'SVHN test'),
                 'train_split_order': ['train'], 'extra_used': False,
                 'held_out_validation_examples': 0}
    report['dataset_load_seconds'] = time.perf_counter() - start_load
    report['dataset_selection'] = selection
    report['transforms'] = {'train': repr(train_tf), 'test': repr(test_tf)}
    s.write_json(out / 'dataset_manifest.json', {'files': info, 'selection': selection,
                                              'transforms': report['transforms']})
    log(f'DATA PASS: train={len(train)}, test={len(test)}; extra NIET gebruikt; '
        'invoer 40x40; volledige train-split; geen validatiesplit.')
    # No explicit generators: match the global RNG policy of the source loaders.
    torch = rt.torch
    loaders = {
        'train': s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(train),
                  batch_size=c['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=False)),
        'test': s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(test),
                  batch_size=c['batch_size'], shuffle=False, num_workers=0, pin_memory=True, drop_last=False)),
    }
    log(f'MINIBATCHES per epoch: train={len(loaders["train"])}; test={len(loaders["test"])}; '
        f'batch={c["batch_size"]}, drop_last=False.')
    model = t.build_model(modules, c['mode'], rt.device)
    s.require(model.expand == c['expand'] and sum(p.numel() for p in model.parameters()) == c['parameters'],
              'Onverwacht model; verwacht svhnq expand=2.')
    validate_initialization(model, c, reference, report)
    optimizer = t.build_optimizer(modules, model, c)
    scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
    contract = {
        'config': c, 'environment': rt.environment, 'settings': rt.settings,
        'source_files': source_info['workcopy_files_sha256'], 'package_sha256': package_hashes,
        'data_files': info['files'], 'dataset_selection': selection, 'transforms': report['transforms'],
        'synthetic_fixture_only': False,
    }
    report['contract_sha256'] = rt.core.json_digest(contract)
    return SimpleNamespace(rt=rt, source=source, modules=modules, model=model, optimizer=optimizer,
                           scheduler=scheduler, loaders=loaders, contract=contract, c=c,
                           report=report, source_info=source_info, initialization_reference=reference)


def restore_timing_sidecar(rt, trace: dict) -> None:
    # The last checkpoint is written before its own write duration is known.
    # The immutable sidecar completes that timing without altering the checkpoint.
    if not trace['epochs']:
        return
    last = trace['epochs'][-1]
    path = Path(last['checkpoint_path']).parent.parent / 'epoch_summaries' / f"epoch_{last['epoch']:05d}.json"
    marker = path.with_suffix('.json.sha256')
    if path.is_file() and marker.is_file():
        s.require(rt.core.file_sha256(path) == marker.read_text().strip(), 'Epoch-sidecar checksum wijkt af.')
        full = json.loads(path.read_text(encoding='utf-8'))
        for key, value in last.items():
            s.require(value == full.get(key), 'Epoch-sidecar strookt niet met checkpoint: ' + key)
        trace['epochs'][-1] = full


def run_epochs(ctx, out: Path, log, stop_after: int, resume: Path | None = None,
               install_signal_handler: bool = True) -> dict:
    rt, c = ctx.rt, ctx.c
    s.require(c['planned_epochs'] == c['reference_end_epoch'] == 100,
              'Onjuist100-epochcontract.')
    s.require(c['scale_learning_end_epoch'] == 100 and c['scale_freeze_epoch'] is None,
              'Deze run leert gewichten en absolute E2-schalen tot epoch100, zonder freeze.')
    s.require(type(stop_after) is int and 1 <= stop_after <= c['planned_epochs'],
              'De stop-epoch moet tussen1 en100 liggen; de oorspronkelijke LR-mijlpalen blijven behouden.')
    torch, core, device = rt.torch, rt.core, rt.device
    model, optimizer, scheduler, loaders = ctx.model, ctx.optimizer, ctx.scheduler, ctx.loaders
    for directory in ['checkpoints', 'epoch_summaries', 'sample_order', 'batch_metrics']:
        (out / directory).mkdir(exist_ok=True)
    s.require(c['initial_relative_offset'] == a.INITIAL_RELATIVE_OFFSET == 6, 'Initbeleid en adapter verschillen.')
    trace = {'initial_model_sha256': s.legacy.state_digest(model.state_dict()),
             'initial_base_model_sha256': s.legacy.state_digest(a.base_state_dict(model)),
             'initial_scale_state': t.scale_state(model),
             'initial_source_scale_state': capture_source_scale_state(model), 'epochs': []}
    done, steps = 0, 0
    expected_steps = len(loaders['train'])
    s.require(expected_steps == math.ceil(c['train_examples'] / c['batch_size']), 'Verkeerd aantal minibatches.')
    resume_hash = None
    if resume is not None:
        resume = resume.resolve()
        state = core.load_verified(resume)
        s.require(0 < state['completed_epoch'] < stop_after, 'Dit checkpoint laat geen nieuwe referentie-epochs toe.')
        ctx.report['restore_checks'] = core.restore_checkpoint(state, model, optimizer, scheduler, {},
                                                              device, ctx.contract)
        trace = copy.deepcopy(state['trace'])
        restore_timing_sidecar(rt, trace)
        done, steps = state['completed_epoch'], state['optimizer_steps']
        s.require(steps == done * expected_steps, 'Opgeslagen aantal stappen is onjuist.')
        s.require(len(trace['epochs']) == done, 'Geschiedenis van epochs is onvolledig.')
        resume_hash = core.file_sha256(resume)
        ctx.report.update(resumed_from=str(resume), resume_input_sha256=resume_hash)
        log(f'HERVAT na epoch {done}; geen initialisatie van het model of LR-schema opnieuw.')
    else:
        log('NIEUWE initialisatie. Geen pilot- of ander trainingscheckpoint ingeladen.')
    first_epoch = done + 1
    soft_stop = {'requested': False}
    old_handler = None
    def handler(signum, frame):
        if soft_stop['requested']:
            raise KeyboardInterrupt('Tweede Ctrl+C: stop direct; alleen volledige checkpoints zijn hervatbaar.')
        soft_stop['requested'] = True
        log('STOP gevraagd: huidige epoch wordt volledig afgewerkt en opgeslagen. Tweede Ctrl+C stopt direct.')
    if install_signal_handler:
        old_handler = signal.signal(signal.SIGINT, handler)
    try:
        for epoch in range(done + 1, stop_after + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            lr = float(optimizer.param_groups[0]['lr'])
            scale_lr = float(optimizer.param_groups[1]['lr'])
            expected_lr = c['lr']
            for milestone in c['scheduler_milestones']:
                if milestone < epoch:
                    expected_lr *= c['scheduler_gamma']
            s.require(lr == expected_lr, f'LR-schema wijkt af in epoch {epoch}: {lr} vs {expected_lr}')
            expected_scale_lr = c['scale_lr']
            for milestone in c['scheduler_milestones']:
                if milestone < epoch: expected_scale_lr *= c['scheduler_gamma']
            s.require(scale_lr == expected_scale_lr, 'Schaal-LR wijkt af van het vastgelegde schema.')
            log(f'EPOCH {epoch}/{stop_after}; MODE={c["mode"]}; LR={lr:g}; oorspronkelijk schema={c["reference_end_epoch"]}')
            before_weights = t.layer_snapshot(model, c['mode'])
            train_rows, train_s, ast_hash, grad_audit = t.phase(rt, ctx.source, ctx.modules, model, optimizer,
                  loaders['train'], epoch, 'train', lr, out / 'batch_metrics', log, c['mode'])
            steps += expected_steps
            t.validate_gradients(model)
            update_rows = t.layer_updates(before_weights, t.layer_snapshot(model, c['mode']), c['mode'], epoch)
            if not any(row['master_changed_values'] > 0 for row in update_rows):
                log('DIAGNOSE: geen gewijzigde MX-masterwaarden in deze epoch; eindige gradients en Adamstappen worden afzonderlijk gecontroleerd.')
            adam_check = t.validate_adam_state(model, optimizer, steps)
            s.legacy.finite_state(model)
            state_before_eval = s.legacy.state_digest(model.state_dict())
            test_rows, test_s, test_ast_hash, test_audit = t.phase(rt, ctx.source, ctx.modules, model, optimizer,
                  loaders['test'], epoch, 'test', lr, out / 'batch_metrics', log, c['mode'])
            s.require(ast_hash == test_ast_hash, 'Train- en evaluatiefunctie zijn niet identiek.')
            s.require(s.legacy.state_digest(model.state_dict()) == state_before_eval,
                      'Evaluatie heeft modelgewichten of BatchNorm-statistieken gewijzigd.')
            coverage = {name: loader.validate(out / 'sample_order' / f'{name}_{epoch:05d}.npy')
                        for name, loader in loaders.items()}
            for rows, expected in [(train_rows, c['train_examples']), (test_rows, c['test_examples'])]:
                s.require(all(r['n'] == expected for r in rows), 'Metric bevat niet de volledige dataset.')
            bn_check = t.validate_bn(model, steps)
            scheduler.step()  # EXACT original position: after train and test.
            s.require(scheduler.last_epoch == epoch, 'Scheduler-epoch wijkt af.')
            checkpoint = out / 'checkpoints' / f'epoch_{epoch:05d}.pt'
            record = {
                'epoch': epoch, 'train': train_rows, 'test': test_rows,
                'lr_used': lr, 'lr_next': float(optimizer.param_groups[0]['lr']),
                'scale_lr_used': scale_lr, 'scale_lr_next': float(optimizer.param_groups[1]['lr']),
                'scale_state': t.scale_state(model), 'source_scale_state': capture_source_scale_state(model),
                'scale_epoch_metrics': grad_audit['scale_epoch_metrics'],
                'optimizer_steps': steps, 'coverage': coverage,
                'batchnorm_counters_correct': bn_check['all_counters_correct'],
                'gradient_checks': grad_audit, 'adam_checks': adam_check, 'weight_updates': update_rows,
                'test_optimizer_steps': test_audit['optimizer_steps_in_phase'],
                'test_source_checks': test_audit['shared_source_checks'],
                'test_prediction_counts': test_audit['prediction_counts'],
                'test_label_counts': test_audit['label_counts'],
                'model_sha256': state_before_eval, 'checkpoint_path': str(checkpoint.resolve()),
                'forward_ast_sha256': ast_hash, 'train_seconds': train_s, 'test_seconds': test_s,
                'first_epoch_in_process': epoch == first_epoch,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            }
            trace['epochs'].append(record)
            best = t.best_record(trace['epochs'])
            record['is_best_at_this_epoch'] = best['epoch'] == epoch
            record['best_epoch_so_far'] = best['epoch']
            t_save = time.perf_counter()
            payload = core.capture_checkpoint(model, optimizer, scheduler, {}, device, epoch, steps,
                                              ctx.contract, trace, {})
            # Empty validation_logits is intentional: scores are in trace; there
            # is NO reason to copy all test logits into every training checkpoint.
            payload['best_prec1'] = best['test'][-1]['top1_percent']
            core.atomic_save(checkpoint, payload)
            record['checkpoint_seconds'] = time.perf_counter() - t_save
            record['total_epoch_seconds'] = time.perf_counter() - t0
            record['other_seconds'] = record['total_epoch_seconds'] - train_s - test_s - record['checkpoint_seconds']
            epoch_file = out / 'epoch_summaries' / f'epoch_{epoch:05d}.json'
            s.write_json(epoch_file, record)
            epoch_file.with_suffix('.json.sha256').write_text(core.file_sha256(epoch_file) + '\n', encoding='ascii')
            ctx.report['completed_epoch'] = epoch
            log(f'KLAAR epoch {epoch}: train={train_s:.1f}s; test={test_s:.1f}s; '
                f'totaal={record["total_epoch_seconds"]:.1f}s; beste epoch={best["epoch"]}')
            log('  TEST ' + ' | '.join(f'{r["weight_format"]}/A32: '
                                    f'{r["top1_percent"]:.4f}%' for r in test_rows))
            for bit, counts in record['test_prediction_counts'].items():
                if max(counts) == sum(counts):
                    log(f'  DIAGNOSE {a.FORMATS[int(bit)]}: alle testbeelden krijgen klasse {counts.index(max(counts))}; meetresultaat, geen technische FAIL.')
            scale_rows = record['scale_epoch_metrics']
            flips = sum(row['exponent_flips_during_epoch'] for row in scale_rows)
            changed = sum(row['phi_changed_blocks'] for row in scale_rows)
            minimum = min(row['e2_exponent_min'] for row in scale_rows)
            maximum = max(row['e2_exponent_max'] for row in scale_rows)
            rel_min = min(row['observed_relative_offset_min'] for row in scale_rows)
            rel_max = max(row['observed_relative_offset_max'] for row in scale_rows)
            log(f'  E2-SCHALEN: absolute exponenten {minimum}..{maximum}; '
                f'gewijzigde FP32-hulpwaarden={changed}; omschakelingen in epoch={flips}; '
                f'geobserveerd verschil met actuele E4-exponent={rel_min}..{rel_max}.')
            estimate = s.estimate_time(trace['epochs'])
            if estimate['available']:
                log(f'  SCHATTING 100 epochs: {estimate["estimated_100_epoch_seconds"] / 3600:.2f} uur '
                    '(zonder download/data-inladen; geen nauwkeurigheidsvoorspelling).')
            if soft_stop['requested'] or (out / 'STOP_AFTER_EPOCH').exists():
                log('Gecontroleerd gestopt op epochgrens; brondata en checkpoints blijven behouden.')
                break
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
    if resume is not None:
        s.require(core.file_sha256(resume) == resume_hash, 'Invoercheckpoint is gewijzigd.')
    return trace


def write_scale_results(ctx, out, trace):
    """Validate absolute exponents, then report relative offsets only as diagnostics."""
    import torch
    records = trace['epochs']
    c_initial = int(ctx.c['initial_relative_offset'])
    rows = [row for record in records for row in record['scale_epoch_metrics']]
    block_rows = []
    states = [(0, trace['initial_scale_state'], trace['initial_source_scale_state'])]
    states += [(r['epoch'], r['scale_state'], r['source_scale_state']) for r in records]
    def exponents(value):
        # Exact FP32 values are representable in FP64, including either side of a half-tie.
        lower = torch.floor(value)
        return (lower + (value - lower >= .5)).to(torch.int64)
    previous = previous_source = None
    source_change_rows = []
    for epoch, state, source_state in states:
        s.require(tuple(state) == a.TARGET_LAYERS and tuple(source_state) == a.TARGET_LAYERS,
                  'Schaalstaat mist lagen of verandert volgorde.')
        for name in a.TARGET_LAYERS:
            value = torch.tensor(state[name], dtype=torch.float64)
            source_code = torch.tensor(source_state[name], dtype=torch.int64)
            expected_shape = a.block_shape(tuple(ctx.model.get_submodule(name).weight.shape))
            s.require(tuple(value.shape) == tuple(source_code.shape) == expected_shape
                      and bool(torch.isfinite(value).all())
                      and bool(((value >= a.PHI_MIN) & (value <= a.PHI_MAX)).all())
                      and bool(((source_code >= 0) & (source_code <= 254)).all()),
                      'Ongeldige opgeslagen absolute schaalstaat: ' + name)
            lower_exp = exponents(value)
            source_exp = source_code - 127
            relative = lower_exp - source_exp
            if epoch == 0 and not ctx.contract['synthetic_fixture_only']:
                s.require(bool((value == source_exp + c_initial).all()),
                          'Nieuwe run moet eenmaal beginnen bij de aanvankelijke E4-schaal x64.')
            for channel, values in enumerate(state[name]):
                for index, phi in enumerate(values):
                    lower = int(lower_exp[channel, index])
                    source = int(source_exp[channel, index])
                    delta = lower - source
                    block_rows.append({'epoch': epoch, 'layer': name, 'output_channel': channel,
                        'block_index': index, 'phi': phi, 'e2_exponent': lower,
                        'e2_scale_code': lower + 127, 'source_exponent': source,
                        'source_scale_code': source + 127, 'relative_offset': delta,
                        'relative_factor': math.ldexp(1.0, delta)})
            if epoch:
                layer_rows = [r for r in records[epoch-1]['scale_epoch_metrics'] if r['layer'] == name]
                s.require(len(layer_rows) == 1, 'Schaaldiagnostiek mist of dupliceert een laag.')
                row = layer_rows[0]
                before = torch.tensor(previous[name], dtype=torch.float64)
                before_exp = exponents(before)
                before_source = torch.tensor(previous_source[name], dtype=torch.int64) - 127
                counts = {str(k): int((lower_exp == k).sum()) for k in range(-127, 128)}
                relative_counts = {str(int(k)): int((relative == k).sum()) for k in torch.unique(relative)}
                net = int((lower_exp != before_exp).sum())
                s.require(row['epoch'] == epoch and row['blocks'] == value.numel()
                          and sum(counts.values()) == value.numel()
                          and row['e2_exponent_counts'] == counts
                          and row['observed_relative_offset_counts'] == relative_counts
                          and row['phi_changed_blocks'] == int((value != before).sum())
                          and row['net_exponent_changes'] == net
                          and net <= row['unique_blocks_switched_during_epoch'] <= value.numel()
                          and row['exponent_flips_during_epoch'] >= row['unique_blocks_switched_during_epoch'],
                          'Schaaldiagnostiek strookt niet met opgeslagen absolute blokstaat.')
                expected = {'phi_min': float(value.min()), 'phi_max': float(value.max()),
                            'phi_mean': float(value.mean()), 'phi_std': float(value.std(unbiased=False)),
                            'phi_update_l2': float((value - before).norm()),
                            'e2_exponent_min': int(lower_exp.min()), 'e2_exponent_max': int(lower_exp.max()),
                            'e2_exponent_mean': float(lower_exp.to(torch.float64).mean()),
                            'observed_relative_offset_min': int(relative.min()),
                            'observed_relative_offset_max': int(relative.max()),
                            'observed_relative_offset_mean': float(relative.to(torch.float64).mean())}
                s.require(all(math.isclose(row[key], val, rel_tol=1e-12, abs_tol=1e-12)
                              for key, val in expected.items()), 'Onjuiste phi/exponentstatistieken.')
                s.require(0 <= row['scale_grad_l2_min'] <= row['scale_grad_l2_max'] < float('inf')
                          and 0 <= row['scale_grad_nonzero_blocks_max'] <= value.numel()
                          and row['projected_values_during_epoch'] >= 0,
                          'Niet-eindige of ongeldige schaalgradientdiagnostiek.')
                s.require(row['phi_at_min_blocks'] == int((value == a.PHI_MIN).sum())
                          and row['phi_at_max_blocks'] == int((value == a.PHI_MAX).sum()),
                          'Onjuiste telling aan de absolute phi-grenzen.')
                s.require(row['exponent_jump_distance_during_epoch'] >= row['exponent_flips_during_epoch']
                          and row['exponent_jump_distance_during_epoch'] >= int((lower_exp - before_exp).abs().sum())
                          and 0 <= row['max_exponent_jump'] <= 254,
                          'Onjuiste spronggrootte van absolute exponenten.')
                source_change_rows.append({'epoch': epoch, 'layer': name, 'blocks': value.numel(),
                    'source_exponents_changed_since_previous_epoch': int((source_exp != before_source).sum()),
                    'e2_exponents_changed_since_previous_epoch': net,
                    'observed_relative_offsets_changed_since_previous_epoch':
                        int((relative != before_exp - before_source).sum())})
        previous, previous_source = state, source_state
    # Count dictionaries are JSON within CSV, not implementation-dependent Python repr.
    csv_rows = [{key: json.dumps(val, sort_keys=True) if isinstance(val, dict) else val
                 for key, val in row.items()} for row in rows]
    s.write_csv(out / 'scale_epoch_metrics.csv', csv_rows)
    s.write_csv(out / 'scale_block_state.csv', block_rows)
    s.write_csv(out / 'scale_source_changes.csv', source_change_rows)
    best, last = t.best_record(records), records[-1]
    selected = {}
    for role, record in [('best_E4M3_test', best), ('last_epoch', last)]:
        group = [r for r in block_rows if r['epoch'] == record['epoch']]
        e2_counts, relative_counts = {}, {}
        for block in group:
            e, d = str(block['e2_exponent']), str(block['relative_offset'])
            e2_counts[e] = e2_counts.get(e, 0) + 1
            relative_counts[d] = relative_counts.get(d, 0) + 1
        selected[role] = {'epoch': record['epoch'], 'blocks': len(group),
                         'e2_exponent_counts': e2_counts,
                         'observed_relative_offset_counts': relative_counts,
                         'relative_offsets_are_diagnostics_not_choices': True}
    scale_bytes = sum(math.prod(a.block_shape(tuple(ctx.model.get_submodule(n).weight.shape)))
                      for n in a.TARGET_LAYERS)
    result = {'schema': 'mx-triple-independent-e2-scale-reference100-scale-learning-v1',
              'learned_scale_parameters': ctx.c['learned_scale_parameters'],
              'initialization': ctx.c['scale_proxy_initialization'],
              'initial_relative_offset': 6, 'initial_relative_factor': 64,
              'tie_break': 'higher_exponent', 'phi_bounds': [a.PHI_MIN, a.PHI_MAX],
              'relative_factor_constrained_after_initialization': False,
              'scale_gradient': ctx.c['scale_gradient'], 'gradient_normalization': 'none',
              'scale_lr': ctx.c['scale_lr'], 'projection_after_every_Adam_update': True,
              'scale_learning_end_epoch': ctx.c['scale_learning_end_epoch'],
              'scale_freeze_epoch': ctx.c['scale_freeze_epoch'],
              'completed_optimizer_updates': last['optimizer_steps'],
              'phi_changed_in_any_epoch': any(r['phi_changed_blocks'] > 0 for r in rows),
              'nonzero_scale_gradient_observed': any(r['scale_grad_nonzero_blocks_max'] > 0 for r in rows),
              'total_exponent_flips_including_repeat_flips': sum(r['exponent_flips_during_epoch'] for r in rows),
              'total_exponent_jump_distance': sum(r['exponent_jump_distance_during_epoch'] for r in rows),
              'max_exponent_jump': max(r['max_exponent_jump'] for r in rows),
              'total_projected_values_including_repeats': sum(r['projected_values_during_epoch'] for r in rows),
              'hard_switch_required_for_technical_PASS': False,
              'zero_finite_scale_gradients_are_valid_diagnostics': True,
              'e4_scale_bytes': scale_bytes, 'independent_e2_scale_bytes': scale_bytes,
              'combined_scale_bytes': 2 * scale_bytes,
              'extra_export_bytes_vs_fixed_x64': scale_bytes,
              'extra_export_bytes_vs_fourbit_relative10': scale_bytes - (scale_bytes // 2),
              'analytical_FP32_phi_gradient_Adam_bytes': ctx.c['learned_scale_parameters'] * 16,
              'analytical_bytes_exclude_framework_and_temporaries': True,
              'selected': selected, 'synthetic_fixture_only': ctx.contract['synthetic_fixture_only']}
    s.write_json(out / 'scale_learning_summary.json', result)
    ctx.report['scale_learning_summary'] = result
    return result


def validate_history_export(path: Path, expected_trace: dict, core) -> dict:
    """Read back the complete saved trace before an export can succeed.

    This is an export-integrity check only; it changes no training computation.
    Read once so the digest describes exactly the bytes whose contents matched.
    """
    try:
        raw = path.read_bytes()
        restored_trace = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            'Opgeslagen history.json is onleesbaar of onvolledig; export wordt niet '
            'als geslaagd gerapporteerd: ' + str(path) + ' (' + str(exc) + ')') from exc
    mismatch = core.differences(expected_trace, restored_trace)
    s.require(not mismatch,
              'Opgeslagen history.json wijkt af van de volledige epochgeschiedenis: '
              + ', '.join(mismatch[:8]))
    return {'file': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'epochs': len(restored_trace['epochs']), 'exact_trace_match': True}


def export_results(ctx, out: Path, trace: dict, log):
    import torch
    records, c, core = trace['epochs'], ctx.c, ctx.rt.core
    s.require(records and [r['epoch'] for r in records] == list(range(1, records[-1]['epoch'] + 1)),
              'Onvolledige epochgeschiedenis.')
    expected_steps = math.ceil(c['train_examples'] / c['batch_size'])
    s.require(c['planned_epochs'] == c['reference_end_epoch'] == 100,
              'Onjuist100-epochcontract.')
    s.require(c['scale_learning_end_epoch'] == 100 and c['scale_freeze_epoch'] is None,
              'Deze run leert gewichten en absolute E2-schalen tot epoch100, zonder freeze.')
    s.require(0 < records[-1]['epoch'] <= c['planned_epochs'],
              'Deze run moet uiterlijk na100 epochs stoppen.')
    for record in records:
        epoch = record['epoch']
        s.require(record['optimizer_steps'] == expected_steps * epoch, 'Verkeerd aantal updates.')
        expected_lr = c['lr']
        for milestone in c['scheduler_milestones']:
            if milestone < epoch: expected_lr *= c['scheduler_gamma']
        s.require(record['lr_used'] == expected_lr, 'Onjuist LR-schema.')
        expected_next = c['lr']
        for milestone in c['scheduler_milestones']:
            if milestone <= epoch: expected_next *= c['scheduler_gamma']
        s.require(record['lr_next'] == expected_next, 'Onjuist LR voor de volgende epoch.')
        s.require(record['batchnorm_counters_correct'], 'BN-tellers niet correct.')
        s.require(record['test_optimizer_steps'] == 0, 'Evaluatie deed optimizerupdates.')
        s.require(record['adam_checks']['all_active_adam_steps_correct']
                  and record['adam_checks']['moments_fp32_finite']
                  and record['adam_checks']['unused_bn_has_no_optimizer_state'],
                  'Onjuiste Adamstaat.')
        audit = record['gradient_checks']
        s.require(audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == expected_steps,
                  'Onvolledige gradientcontroles.')
        s.require(audit['detached_teacher_checks'] == 2 * expected_steps,
                  'Er moeten twee losgekoppelde teachercontroles per minibatch zijn.')
        s.require(audit['forward_calls_E4M3'] == audit['forward_calls_E2M3']
                  == audit['forward_calls_E2M1'] == expected_steps, 'Verkeerd aantal forwards.')
        s.require(audit['cache_empty_before_all_optimizer_steps'],
                  'Broncache niet leeg voor optimizerupdate.')
        s.require(audit['scale_projection_after_every_update']
                  and audit['scale_optimizer_steps'] == expected_steps
                  and len(record['scale_epoch_metrics']) == len(a.TARGET_LAYERS),
                  'Onvolledige schaalupdates, projecties of diagnostiek.')
        expected_scale = c['scale_lr']
        for milestone in c['scheduler_milestones']:
            if milestone < epoch: expected_scale *= c['scheduler_gamma']
        expected_scale_next = c['scale_lr']
        for milestone in c['scheduler_milestones']:
            if milestone <= epoch: expected_scale_next *= c['scheduler_gamma']
        s.require(record['scale_lr_used'] == expected_scale
                  and record['scale_lr_next'] == expected_scale_next,
                  'Schaaloptimizer volgt niet het vastgelegde LR-schema.')
        for phase_name, count in [('train', c['train_examples']), ('test', c['test_examples'])]:
            source_audit = (audit['shared_source_checks'] if phase_name == 'train'
                            else record['test_source_checks'])
            batches = math.ceil(count / c['batch_size'])
            s.require(source_audit['source_triplets'] == batches
                      and source_audit['high_encodings'] == source_audit['exponent_derivations']
                      == source_audit['mantissa_derivations'] == batches * len(a.TARGET_LAYERS)
                      and source_audit['learned_absolute_scales_applied'] == 2 * batches * len(a.TARGET_LAYERS)
                      and source_audit['mantissa_scale_copies_unchanged'] == batches * len(a.TARGET_LAYERS)
                      and source_audit['e2_exponent_bounds'] == [-127, 127]
                      and source_audit['independent_absolute_lower_scale']
                      and source_audit['learned_shared_scale']
                      and source_audit['source_cache_cleared']
                      and source_audit['no_direct_lower_quantizer']
                      and source_audit['cache_cleared_before_adam'],
                      'Onvolledige gedeelde E4M3-broncontrole: ' + phase_name)
        s.require(set(record['test_prediction_counts']) == {'8', '6', '4'},
                  'Voorspellingsverdelingen ontbreken.')
        for counts in [record['test_label_counts'], *record['test_prediction_counts'].values()]:
            s.require(len(counts) == 10 and all(type(n) is int and n >= 0 for n in counts)
                      and sum(counts) == c['test_examples'], 'Onvolledige klasseverdeling.')
        for phase, count in [('train', c['train_examples']), ('test', c['test_examples'])]:
            s.require([row['weight_bits'] for row in record[phase]] == list(a.MODES), 'Onvolledige modi.')
            for row in record[phase]:
                bit = row['weight_bits']
                expected_loss = 'recursive_soft_cross_entropy' if phase == 'train' and bit != 8 else 'hard_cross_entropy'
                expected_teacher = a.FORMATS[8 if bit == 6 else 6] if phase == 'train' and bit != 8 else None
                s.require(row['n'] == count and row['activation_bits'] == 32 and row['bn_bank_key'] == a.BANK[bit]
                          and row['weight_format'] == a.FORMATS[bit] and row['objective_type'] == expected_loss
                          and row['teacher_weight_format'] == expected_teacher,
                          'Onjuiste modus, loss of datadekking.')
                s.require(abs(row['top1_percent'] - row['count_top1_percent']) < 1e-4,
                          'Accuracy strookt niet met correcte aantallen.')
    best, last = t.best_record(records), records[-1]
    selected, details, mx_exports = [], {}, {}
    for role, record in [('best_E4M3_test', best), ('last_epoch', last)]:
        path = Path(record['checkpoint_path'])
        state = core.load_verified(path)
        s.require(not core.differences(state['contract'], ctx.contract), 'Checkpointcontract wijkt af.')
        s.require(state['completed_epoch'] == record['epoch'], 'Verkeerde checkpointepoch.')
        s.require(s.legacy.state_digest(state['model']) == record['model_sha256'], 'Modelhash wijkt af.')
        s.require(not core.differences(state['trace']['epochs'][-1]['test'], record['test']), 'Testmetrics verschillen.')
        suffix = 'best' if role == 'best_E4M3_test' else 'last'
        filename = suffix + '_checkpoint.pt'
        target = out / filename
        with path.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
        digest = core.file_sha256(target)
        s.require(digest == core.file_sha256(path), 'Checkpointkopie wijkt af.')
        target.with_suffix('.pt.sha256').write_text(digest + '\n', encoding='ascii')
        details[role] = {'epoch': record['epoch'], 'file': filename, 'sha256': digest,
                         'original_checkpoint': str(path), 'selection_score_E4M3': record['test'][-1]['top1_percent']}
        selected.extend({'role': role, **row} for row in record['test'])
        # Encode and compare on the training device: G(W) can differ after moving to CPU.
        model = copy.deepcopy(ctx.model).to(ctx.rt.device).eval()
        model.load_state_dict(state['model'], strict=True)
        exports_for_role = {}
        provenance = {
            'training': 'joint independent absolute E2 block scale; E4 source -> exponent-MSB E2M3 -> mantissa E2M1; identity master STE plus ln2 absolute log-scale gradient; A32',
            'source_epoch': record['epoch'], 'checkpoint_sha256': digest,
            'encoding_device': str(ctx.rt.device),
            'synthetic_fixture_only': ctx.contract['synthetic_fixture_only']}
        source_payload, all_layer_rows = a.source_payload(model, provenance)
        source_file = out / f'mx_source_MXFP8_E4M3_{suffix}.pt'
        source_details = core.atomic_save(source_file, source_payload)
        # Derivation starts from the verified saved source. The compact source
        # has all BN/FP32 auxiliary state but no masters for the six MX layers.
        loaded_source = a.load_verified_payload(source_file)
        a.validate_source(loaded_source)
        source_digest = core.file_sha256(source_file)
        all_equal_scales = all(torch.equal(loaded_source['lower_scale_codes'][name].cpu(),
                                           loaded_source['mx_layers'][name]['scale_codes'].cpu())
                               for name in a.TARGET_LAYERS)
        for bit in (8, 6, 4):
            f = a.codec(bit)
            if bit == 8:
                mx_file, mx_details, loaded = source_file, source_details, loaded_source
            else:
                payload = a.derive_payload(loaded_source, bit, source_digest)
                mx_file = out / f'mx_derived_{a.FORMATS[bit]}_{suffix}.pt'
                mx_details = core.atomic_save(mx_file, payload)
                loaded = a.load_verified_payload(mx_file)
            a.select_mode(model, bit)
            for name in a.TARGET_LAYERS:
                encoded = f.MXTensor.from_payload(loaded['mx_layers'][name], ctx.rt.device)
                restored = encoded.decode()
                layer = model.get_submodule(name)
                with torch.no_grad():
                    live = layer.quantize_fn(layer.weight.detach())
                s.require(torch.equal(restored, live), 'Export verschilt van dezelfde trainingsroute.')
                source_scale = loaded_source['mx_layers'][name]['scale_codes']
                expected_scale = source_scale if bit == 8 else loaded_source['lower_scale_codes'][name]
                s.require(expected_scale.dtype == torch.uint8 and bool((expected_scale <= 254).all())
                          and torch.equal(encoded.scale_codes.cpu(), expected_scale.cpu()),
                          'Export volgt niet de opgeslagen absolute schaalbank voor zijn exponentbreedte.')
                s.require(name + '.weight' not in loaded['auxiliary_fp32_state'], 'FP32-master in export.')
            exports_for_role[str(bit)] = {**mx_details, 'epoch': record['epoch'], 'weight_bits': bit,
                                         'decoded_weights_exact': True, 'six_fp32_masters_absent': True,
                                         'full_test_forward_rerun': False, 'truncation': bit != 8,
                                         'source_weight_bits': 8, 'is_source_export': bit == 8,
                                         'derived_from_saved_source': bit != 8,
                                         'source_file_sha256': source_digest,
                                         'source_scale_bytes_copied': bit == 8 or all_equal_scales,
                                         'independent_absolute_lower_scales_in_source': True,
                                         'independent_e2_scale_bytes': sum(x.numel() for x in loaded_source['lower_scale_codes'].values()),
                                         'e2_scale_format': 'E8M0',
                                         'relative_factor_constrained': False,
                                         'fp32_scale_proxies_absent': True,
                                         'learned_shared_scale': bit != 8,
                                         'master_weights_read_to_derive': False,
                                         'master_weights_read_to_encode_source': bit == 8,
                                         'saved_and_reloaded': True,
                                         'encoding_and_comparison_device': str(ctx.rt.device),
                                         'export_validation_scope': 'exact weights versus live shared-source route on training device; no full test rerun'}
            layer_rows = [row for row in all_layer_rows if row['weight_bits'] == bit]
            s.require(len(layer_rows) == len(a.TARGET_LAYERS), 'Onvolledige exportdiagnostiek.')
            s.write_csv(out / f'mx_export_layers_{bit}_{suffix}.csv', layer_rows)
        s.require(core.file_sha256(source_file) == source_digest, 'Opgeslagen E4M3-bron gewijzigd tijdens afleiding.')
        mx_exports[role] = exports_for_role
    s.write_csv(out / 'selected_metrics.csv', selected)
    s.write_csv(out / 'epoch_metrics.csv', [row for r in records for phase in ('train', 'test') for row in r[phase]])
    s.write_csv(out / 'weight_updates.csv', [row for r in records for row in r['weight_updates']])
    selection = {'policy': c['selection_policy'], 'best': details['best_E4M3_test'], 'last': details['last_epoch'],
                 'same_checkpoint': best['epoch'] == last['epoch'], 'selected_on_official_test': True,
                 'independent_test_estimate': False, 'selection_mode': 'MXFP8_E4M3/A32',
                 'all_modes_from_same_selected_checkpoint': True}
    s.write_json(out / 'checkpoint_selection.json', selection)
    s.write_json(out / 'history.json', trace)
    prediction_rows = []
    for record in records:
        for bit in a.MODES:
            counts = record['test_prediction_counts'][str(bit)]
            prediction_rows.append({'epoch': record['epoch'], 'weight_bits': bit,
                'weight_format': a.FORMATS[bit], 'n': sum(counts),
                **{f'predicted_class_{k}': count for k, count in enumerate(counts)},
                'classes_predicted': sum(n > 0 for n in counts),
                'dominant_predicted_class': counts.index(max(counts)),
                'dominant_prediction_fraction': max(counts) / sum(counts)})
    s.write_csv(out / 'test_prediction_counts.csv', prediction_rows)
    scale_summary = write_scale_results(ctx, out, trace)
    elapsed = sum(r['total_epoch_seconds'] for r in records)
    pilot_completed = False  # This fresh full run is not a continuation of the prior pilot.
    full_completed = last['epoch'] == c['planned_epochs']
    epoch50_checkpoint = {'available': False, 'epoch': 50,
        'reason': 'Epoch50 is not yet present in this run history.'}
    if last['epoch'] >= 50:
        record50 = records[49]
        path50 = Path(record50['checkpoint_path'])
        s.require(record50['epoch'] == 50 and path50.is_file(),
                  'Het volledige checkpoint van epoch50 ontbreekt.')
        digest50 = core.file_sha256(path50)
        marker50 = path50.with_suffix('.pt.sha256')
        s.require(marker50.is_file() and marker50.read_text(encoding='ascii').strip() == digest50,
                  'Checksum van het bewaarde epoch50-checkpoint wijkt af.')
        epoch50_checkpoint = {'available': True, 'epoch': 50, 'original_checkpoint': str(path50),
            'sha256': digest50, 'optimizer_steps': record50['optimizer_steps'],
            'scale_lr_used': record50['scale_lr_used'], 'scale_lr_next': record50['scale_lr_next'],
            'full_training_state': True, 'checkpoint_in_feedback_bundle': False,
            'state_includes': ['FP32 weights', 'FP32 absolute scale proxies phi', 'both Adam groups',
                              'scheduler', 'Python/NumPy/Torch CPU/CUDA RNG', 'full epoch history'],
            'purpose': 'Optional future controlled freeze branch; this run continues learning phi through epoch100.'}
    s.write_json(out / 'epoch50_checkpoint.json', epoch50_checkpoint)
    summary = {'schema': 'mx-triple-independent-e2-scale-reference100-completion-v1',
               'planned_epochs': c['planned_epochs'], 'reference_end_epoch': c['reference_end_epoch'],
               'completed_epochs': last['epoch'],
               'pilot_completed': pilot_completed,
               'full_training_completed': full_completed,
               'full_100_epoch_training_completed': full_completed,
               'optimizer_steps': last['optimizer_steps'],
               'expected_optimizer_steps_at_reference_completion': expected_steps * c['reference_end_epoch'],
               'expected_training_forward_passes_at_reference_completion': expected_steps * c['reference_end_epoch'] * 3,
               'best_epoch_E4M3_A32': best['epoch'], 'last_epoch': last['epoch'],
               'lr_schedule_verified': True, 'all_epoch_modes_complete': True,
               'resumed': 'resumed_from' in ctx.report,
               'new_initialization_not_previous_experiment_warmstart': True,
               'epoch50_checkpoint': epoch50_checkpoint,
               'training_forward_passes': last['optimizer_steps'] * 3,
               'train_examples': c['train_examples'], 'test_examples': c['test_examples'],
               'mode': c['mode'], 'weight_formats': c['weight_formats'], 'activation_bits': 32,
               'distillation': True, 'truncation': True, 'source_weight_bits': 8,
               'exponent_truncation': True, 'mantissa_truncation': True,
               'independent_E2_shared_scale': True, 'learned_shared_scale': True,
               'scale_parameterization': c['scale_parameterization'],
               'scale_proxy_initialization': c['scale_proxy_initialization'],
               'initial_relative_factor': 64, 'scale_proxy_bounds': [-127.0, 127.0],
               'relative_factor_constrained_after_initialization': False,
               'learned_scale_parameters': c['learned_scale_parameters'],
               'scale_gradient': c['scale_gradient'], 'scale_gradient_normalization': 'none',
               'scale_projection_after_every_update': True, 'scale_lr': c['scale_lr'],
               'scale_learning_end_epoch': c['scale_learning_end_epoch'],
               'scale_freeze_epoch': c['scale_freeze_epoch'],
               'scale_learning_summary': scale_summary,
               'quantization_source': c['quantization_source'],
               'initial_model_sha256': trace['initial_model_sha256'],
               'initial_base_model_sha256': trace['initial_base_model_sha256'],
               'measured_completed_epoch_seconds': elapsed,
               'gradient_checks_before_every_update': True, 'fp32_masters_and_moments_verified': True,
               'one_E4_source_per_layer_per_minibatch_verified': True,
               'lower_scale_codes_depend_only_on_phi_after_initialization': True,
               'e4_source_scale_unchanged_policy': True,
               'all_mantissa_scale_bytes_equal_E2M3': True,
               'source_cache_cleared_before_every_optimizer_update': True,
               'batchnorm_learned_during_training_eval_frozen': True,
               'synthetic_fixture_only': ctx.contract['synthetic_fixture_only']}
    s.write_json(out / 'training_summary.json', summary)
    estimate = s.estimate_time(records, c['reference_end_epoch'])
    estimate['weight_mode'] = c['weight_format']
    estimate['includes'] = ['three-mode training', 'three-mode test', 'epoch diagnostics', 'checkpoint write']
    estimate['measured_completed_epoch_seconds'] = elapsed
    estimate['pilot_completed'] = pilot_completed
    estimate['full_training_completed'] = full_completed
    estimate['full_100_epoch_training_completed'] = full_completed
    estimate['measured_100_epoch_seconds'] = elapsed if full_completed else None
    estimate['interpretation'] = (
        'The measured 100-epoch time is the sum of all 100 recorded epoch durations, including restored epochs.'
        if full_completed else
        'Only completed epochs have measured durations. The 100-epoch estimate is a projection for a fresh '
        'run under comparable conditions, not measured or remaining time.')
    estimate['limitations'] = [
        'The epoch sum excludes dependency installation, dataset preparation/loading and final report export.',
        'The projection is distinct from measured duration and requires comparable laptop/backend/data loading conditions.',
        ('All 100 epochs, including both optimizer groups and the LR transitions at50/75/90, are represented '
         'in the measured history; this observed duration does not guarantee the duration of a future run.'
         if full_completed else
         'Only completed epochs are observed; future epochs, LR transitions not yet reached, '
         'thermal drift and changing background load can alter the projected duration.'),
    ]
    s.write_json(out / 'timing_estimate.json', estimate)
    s.write_csv(out / 'epoch_times.csv', [
        {'epoch': r['epoch'], **{k: r.get(k) for k in ['train_seconds', 'test_seconds', 'checkpoint_seconds',
          'other_seconds', 'total_epoch_seconds', 'first_epoch_in_process',
          'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes']}} for r in records])
    ctx.report.update(completed_epoch=last['epoch'], optimizer_steps=last['optimizer_steps'],
                      selected_best_epoch=best['epoch'], selected_last_epoch=last['epoch'],
                      initial_model_sha256=trace['initial_model_sha256'],
                      initial_base_model_sha256=trace['initial_base_model_sha256'], checkpoint_selection=selection,
                      training_summary=summary, timing_estimate=estimate, mx_exports=mx_exports,
                      synthetic_fixture_only=ctx.contract['synthetic_fixture_only'],
                      full_training_completed=full_completed, pilot_completed=pilot_completed,
                      full_100_epoch_training_completed=full_completed,
                      epoch50_checkpoint=epoch50_checkpoint,
                      training_forward_passes=last['optimizer_steps'] * 3)
    write_initialization_comparison(ctx, out, trace)
    write_notebook(out)
    archive_initialization_reference(out)
    ctx.report['history_export_validation'] = validate_history_export(
        out / 'history.json', trace, core)
    log(f'BEHOUDEN: alle drie modi uit beste E4M3-checkpoint {best["epoch"]}; laatste {last["epoch"]}.')
    log(f'ONAFHANKELIJKE E2-SCHAAL: {last["epoch"]}/100 epochs; {last["optimizer_steps"]} updates; '
        f'{last["optimizer_steps"] * 3} trainingspasses; gemeten epochtijd={elapsed / 60:.2f} min.')


def verify_unchanged(ctx, log):
    for name, expected in ctx.source_info['workcopy_files_sha256'].items():
        s.require(ctx.rt.core.file_sha256(ctx.source / name) == expected, 'Bronbestand gewijzigd: ' + name)
    s.verify_package()
    for name, expected in ctx.report['dataset_files']['files'].items():
        actual = s.hashes(Path(ctx.report['dataset_files']['root']) / name)
        s.require(all(actual[key] == expected[key] for key in actual), 'Ruwe dataset gewijzigd: ' + name)
    ctx.report.update(source_unchanged=True, dataset_files_unchanged=True)
    log('Eindcontrole: bronbestanden en ruwe datasetbestanden ongewijzigd.')


def feedback_bundle(out):
    names = ['run_report.json', 'epoch_metrics.csv', 'selected_metrics.csv', 'weight_updates.csv',
             'checkpoint_selection.json', 'timing_estimate.json', 'epoch_times.csv', 'history.json',
             'dataset_manifest.json', 'source_manifest.json', 'compatibility.patch',
             'pip_freeze.txt', 'pip_check.txt', 'run.log', 'warnings.log',
             'training_change_manifest.json', 'training_summary.json', 'epoch50_checkpoint.json',
             'scale_parameterization_reference_report.json', 'scale_parameterization_comparison.csv',
             'test_prediction_counts.csv', 'scale_epoch_metrics.csv', 'scale_block_state.csv', 'scale_learning_summary.json',
             'scale_source_changes.csv',
             'codec_report.json', 'trunc_training_check_report.json',
             'MX_triple_independent_e2_scale_reference100_resultaten.ipynb', 'notebook_support.py']
    names += [f'mx_export_layers_{bit}_{role}.csv' for bit in a.MODES for role in ('best', 'last')]
    for reference_directory in REFERENCE_DIRECTORIES:
        if (out / reference_directory).is_dir():
            names += [str(path.relative_to(out)).replace('\\', '/')
                      for path in sorted((out / reference_directory).rglob('*')) if path.is_file()]
    with ZipFile(out / 'feedback_bundle.zip', 'x', compression=ZIP_DEFLATED) as z:
        added = set()
        for name in names:
            path = out / name
            if path.is_file():
                z.write(path, name)
                added.add(name)
        # Keep the analysis notebook available even when check/run fails early.
        for name in ['MX_triple_independent_e2_scale_reference100_resultaten.ipynb', 'notebook_support.py']:
            if name not in added:
                z.write(ROOT / name, name)
                added.add(name)
        for reference_directory in REFERENCE_DIRECTORIES:
            if not (out / reference_directory).is_dir():
                for path in sorted((ROOT / reference_directory).rglob('*')):
                    if path.is_file():
                        name = str(path.relative_to(ROOT)).replace('\\', '/')
                        z.write(path, name)
                        added.add(name)
        # Small, reviewable runner snapshot; no checkpoints, dataset or optimizer tensors.
        source_names = [p.name for p in ROOT.iterdir() if p.is_file()
                        and p.suffix in ('.py', '.json', '.md', '.ipynb')]
        source_names.append('upstream/Any-Precision-DNNs-master.zip')
        for name in sorted(source_names):
            z.write(ROOT / name, 'runner_source/' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'resume'])
    parser.set_defaults(mode='joint')
    parser.add_argument('--data-root', type=Path, default=Path('data/svhn'))
    parser.add_argument('--output-root', type=Path, default=Path('mx_runs/mx_triple_independent_e2_scale_reference100'))
    parser.add_argument('--device', choices=['cuda', 'cuda:0'], default='cuda:0')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=100)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if not 1 <= args.stop_after_epoch <= 100:
        parser.error('De stop-epoch moet tussen1 en100 liggen; het oorspronkelijke100-epochschema blijft behouden.')
    if (args.command == 'resume') != (args.checkpoint is not None):
        parser.error('Alleen resume vereist --checkpoint; run begint altijd vanaf initialisatie.')
    # CHECK ONLY: configure cuBLAS before any CUDA context is initialized.
    # This Python process cannot modify the parent PowerShell environment.
    # The run/resume paths never set this variable.
    if args.command == 'check':
        import os
        previous_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if previous_workspace not in (':4096:8', ':16:8'):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_root / (stamp + '_' + args.mode + '_' + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mx-triple-independent-e2-scale-reference100-report-v1', 'status': 'RUNNING',
              'package_revision': 'mx_triple_independent_e2_scale_reference100_v1',
              'command': args.command, 'mode': args.mode, 'started_at': s.now(),
              'output_directory': str(out), 'argv': sys.argv, 'warnings': [],
              'limitations': ['train only; shared E4M3 source and two code-derived MX modes; this is not a paper reproduction',
                              'FP32 update master/gradients/Adam; A32; exponent MSB and mantissa truncation; both E2 modes share an independently learned absolute E8M0 scale per block',
                              'three forward/backwards: E4M3 hard CE, E2M3/E2M1 recursive detached soft CE, one Adam step',
                              'BN banks learn during training; absolute E2 log-scale STE; E4 source scale unchanged; independent scales do not imply independent element quantization',
                              'one hundred epochs from fresh original weights; initial lower scale equals initial E4 x64 only once; phi and weights learn through100 without a freeze; low accuracy or no hard switches are valid measured outcomes',
                              'official test used for checkpoint selection, not an independent test estimate',
                              'numerical weight-only MX emulation; no native MX kernel or energy claim',
                              'GPU bit-exact repeat/resume not guaranteed; fixed seed is not proof of determinism',
                              'one seed and one training run; no proof of general superiority or convergence']}
    if args.command == 'check':
        report['check_only_workspace'] = {
            'before': previous_workspace,
            'during_check_process': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'training_process_changed': False}
    log = s.legacy.Log(out / 'run.log')
    started = time.perf_counter()
    exit_code = 1
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        entry = {'category': category.__name__, 'message': str(message), 'file': filename, 'line': lineno}
        report['warnings'].append(entry)
        log('WARNING ' + str(message))
        with (out / 'warnings.log').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.showwarning = warning_handler
            log('RESULTATENMAP: ' + str(out))
            log('ONAFHANKELIJKE E2-SCHAAL:100 epochs vanaf nul; gewichten en phi blijven leren tot100; absolute E8M0-schaal per B32-groep, eenmaal gestart bij E4 x64; E4M3 -> E2M3 -> E2M1; A32; FP32-masters.')
            write_notebook(out)
            ctx = create_context(args, out, report, log, check_only=args.command == 'check')
            if ctx is None:
                report['status'] = 'PASS'
            else:
                trace = run_epochs(ctx, out, log, args.stop_after_epoch, args.checkpoint)
                export_results(ctx, out, trace, log)
                verify_unchanged(ctx, log)
                report['status'] = 'PASS' if report['full_training_completed'] else 'STOPPED_AT_EPOCH_BOUNDARY'
            exit_code = 0
    except BaseException as exc:
        report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
        report['error'], report['traceback'] = repr(exc), traceback.format_exc()
        log(report['status'] + ': ' + str(exc))
        log(report['traceback'])
        log('Geen bestaande data of eerdere resultaten gewijzigd. Deel de feedback_bundle.zip uit deze map.')
    finally:
        report['ended_at'] = s.now()
        report['session_wall_seconds'] = time.perf_counter() - started
        log('EINDSTATUS: ' + report['status'])
        log('Rapport: ' + str(out / 'run_report.json'))
        log.handle.close()
        s.write_json(out / 'run_report.json', report)
        feedback_bundle(out)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
