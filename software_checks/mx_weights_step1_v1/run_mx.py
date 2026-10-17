"""Step 1: read-only W32/A32 vs weight-only MXFP6 E2M3/A32 evaluation.

No optimizer, no calibration, no training, no precision search. The source
checkpoints and datasets are never overwritten. The actual training runner
is deliberately not patched until the representation has passed this check.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import warnings
import zipfile

import svhn_support as s
from mx_format import require

ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path(r'C:\Users\renzo\master-thesis-mx')
CONFIG = {
    'weight_format': 'MXFP6_E2M3', 'local_exponent_bits': 2, 'mantissa_bits': 3,
    'local_bias': 1, 'block_size': 32, 'scale_format': 'E8M0',
    'scale_rule': 'paper_algorithm1_floor_log2_amax_minus_2; clip[-127,127]',
    'layout': 'output_channel_rows; flatten_Cin_Kh_Kw; trailing_positive_zero_padding',
    'rounding': 'nearest_ties_to_even; saturate',
    'input_fp32_subnormals': 'flush_to_signed_zero', 'local_subnormals': 'supported',
    'NaN_Inf': 'reject_nonfinite_input', 'zero_block': 'scale_1_and_signed_zero_codes',
    'weight_preprocess': 'retain_original_W32_tanh_and_normalization',
    'activations': 'original_A32_ReLU', 'batchnorm': 'original_bank32_no_recalibration',
    'bias_first_last': 'unchanged_FP32', 'arithmetic': 'FP32_emulator_not_native_MX',
    'training': False, 'mantissa_truncation': False, 'optimizer_steps': 0,
    'checkpoint_policy': 'existing_reference_best_and_last; no_new_selection',
}

def new_directory(root: Path, name: str) -> Path:
    out = root / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + name)
    out.mkdir(parents=True, exist_ok=False)
    return out

def package_hashes() -> dict:
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        require(s.legacy.sha256(ROOT / name) == expected, 'Pakket gewijzigd: ' + name)
    return manifest

def dependencies(out: Path) -> None:
    for operation in ('freeze', 'check'):
        p = subprocess.run([sys.executable, '-m', 'pip', operation], capture_output=True,
                           text=True, encoding='utf-8', errors='replace')
        (out / ('pip_' + operation + '.txt')).write_text(p.stdout + p.stderr, encoding='utf-8')
        require(p.returncode == 0, 'pip ' + operation + ' faalt; zie rapport. Niets automatisch upgraden.')

def notebook(out: Path) -> None:
    n = json.loads((ROOT / 'MX_weights_resultaten.ipynb').read_text(encoding='utf-8'))
    for c in n['cells']:
        if 'parameters' in c.get('metadata', {}).get('tags', []):
            c['source'] = 'from pathlib import Path\nRUN_DIR = Path(' + repr(str(out.resolve())) + ')\n'
    s.write_json(out / 'MX_weights_resultaten.ipynb', n)

def feedback(out: Path) -> None:
    names = ['run_report.json', 'metrics.csv', 'layer_metrics.csv', 'pip_freeze.txt',
             'pip_check.txt', 'run.log', 'source_manifest.json', 'compatibility.patch']
    with zipfile.ZipFile(out / 'feedback_bundle.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            if (out / name).is_file():
                z.write(out / name, name)


def measure(models: dict, loader, device, utils) -> tuple[list[dict], dict]:
    import torch
    logits = {name: [] for name in models}
    targets = []
    meters = {name: [utils.AverageMeter() for _ in range(3)] for name in models}
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device, non_blocking=True)
            targets.append(labels.cpu())
            for name, model in models.items():
                output = model(inputs)
                require(bool(torch.isfinite(output).all()), 'Niet-eindige logits in ' + name)
                loss = criterion(output, labels)
                top1, top5 = utils.accuracy(output.data, labels, topk=(1, 5))
                for meter, v in zip(meters[name], (loss, top1, top5)):
                    meter.update(float(v), labels.numel())
                logits[name].append(output.cpu())
    target = torch.cat(targets)
    outputs = {name: torch.cat(parts) for name, parts in logits.items()}
    rows = []
    for name, output in outputs.items():
        pred = output.topk(5, 1, True, True)[1]
        correct1 = int((pred[:, 0] == target).sum())
        correct5 = int(pred.eq(target[:, None]).any(1).sum())
        rows.append({'mode': name, 'n': target.numel(), 'correct1': correct1,
                     'correct5': correct5, 'top1_percent': meters[name][1].avg,
                     'count_top1_percent': 100 * correct1 / target.numel(),
                     'hard_cross_entropy': meters[name][0].avg})
    return rows, {'target': target, 'logits': outputs}


def evaluate(args, rt, modules, source, source_info, report, out: Path, log) -> None:
    import torch
    import mx_adapter as a
    run = args.reference_run.resolve()
    require(run.is_dir(), 'Referentierun ontbreekt: ' + str(run))
    require(not out.resolve().is_relative_to(run), 'Resultaten mogen niet binnen de referentierun staan.')
    reference_file = run / 'run_report.json'
    r = json.loads(reference_file.read_text(encoding='utf-8'))
    require(r.get('schema') == 'anyprecision-svhn-trainonly-reference100-report-v1' and
            r.get('status') == 'PASS' and r.get('completed_epoch') == 100 and
            r.get('full_training_completed') is True, 'Verwacht geslaagde 100-epochreferentie.')
    require(r['config']['train_split'] == 'train' and not r['config']['extra_used'], 'Verkeerde trainingssplit.')
    require(r['source']['workcopy_files_sha256'] == source_info['workcopy_files_sha256'], 'Referentiebron wijkt af.')
    before_report = s.legacy.sha256(reference_file)
    report['reference'] = {'path': str(run), 'report_sha256': before_report,
                           'best_epoch': r['selected_best_epoch'], 'last_epoch': r['selected_last_epoch']}

    # Test only; no train split and no extra file is opened or downloaded here.
    data_root = args.data_root.resolve()
    cls = modules.data.SVHNMIX
    _, filename, expected_md5 = cls.split_list['test']
    data_file = data_root / filename
    require(data_file.is_file(), 'Bestaande testdata ontbreken; deze diagnose downloadt niets.')
    data_info = s.hashes(data_file)
    require(data_info['md5'] == expected_md5, 'Testbestand heeft afwijkende MD5.')
    modules.data.data_paths['svhn'] = str(data_root)
    tf = modules.data.get_transform('svhn', 'val')
    test = modules.data.get_dataset('svhn', 'val', tf, download=False)
    require(test.split == 'test', 'Verkeerde evaluatiesplit.')
    report['dataset'] = {'file': data_info, 'test': s.validate_dataset(test, 26032, 'SVHN test'),
                         'transform': repr(tf), 'train_read': False, 'extra_read': False}
    loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, num_workers=0,
                                        pin_memory=True, drop_last=False)
    rows_all, layers_all, checks = [], [], {}
    for role in ('best', 'last'):
        entry = r['checkpoint_selection'][role]
        path = run / entry['file']
        require(path.parent == run and path.name in ('best_checkpoint.pt', 'last_checkpoint.pt'), 'Onveilig checkpointpad.')
        digest = rt.core.file_sha256(path)
        require(digest == entry['sha256'], 'Checkpoint wijkt af van referentierapport.')
        saved = rt.core.load_verified(path)
        require(saved['completed_epoch'] == entry['epoch'], 'Checkpoint-epoch klopt niet.')
        require(saved['contract']['config'] == r['config'], 'Checkpointprotocol klopt niet.')
        require(saved['contract']['synthetic_fixture_only'] is False, 'Geen synthetisch checkpoint toegestaan.')
        expected_metric = [x for x in saved['trace']['epochs'][-1]['test'] if x['bits_w_and_a'] == 32][0]
        baseline = modules.models.svhnq(s.BITS, expand=2).to(rt.device)
        baseline.load_state_dict(saved['model'], strict=True)
        a.select_a32(baseline)
        baseline.eval().requires_grad_(False)
        baseline_hash = s.legacy.state_digest(baseline.state_dict())
        wrapped_control = a.adapt_copy(baseline, enabled=False).eval()
        online_mx = a.adapt_copy(baseline, enabled=True).eval()
        payload, layer_rows = a.export_payload(baseline, {
            'checkpoint_sha256': digest, 'reference_epoch': entry['epoch'], 'role': role,
            'reference_report_sha256': before_report, 'config': CONFIG})
        export_path = out / ('mx_source_' + role + '.pt')
        export_info = rt.core.atomic_save(export_path, payload)
        loaded_payload = torch.load(export_path, map_location='cpu', weights_only=True)
        require(rt.core.file_sha256(export_path) == export_info['sha256'], 'MX-exporthash klopt niet.')
        reloaded = a.load_exported_model(modules.models, loaded_payload, rt.device)
        # Exact tensor equality between source decoding and dynamic quantizer.
        decoded_exact = all(torch.equal(
            online_mx.get_submodule(name).quantize_fn(online_mx.get_submodule(name).weight),
            reloaded.get_submodule(name).decoded_weight) for name in a.TARGET_LAYERS)
        require(decoded_exact, 'Quantizer en opgeslagen codes decoderen niet naar dezelfde gewichten.')
        models = {'reference_W32_A32': baseline, 'wrapper_MX_off_A32': wrapped_control,
                  'MXFP6_E2M3_A32': online_mx, 'stored_MXFP6_E2M3_A32': reloaded}
        log(f'EVALUATIE {role}, epoch {entry["epoch"]}: vier controlepaden; geen training.')
        if rt.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        rows, arrays = measure(models, loader, rt.device, modules.utils)
        if rt.device.type == 'cuda':
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        y = arrays['logits']
        control_equal = torch.equal(y['reference_W32_A32'], y['wrapper_MX_off_A32'])
        export_equal = torch.equal(y['MXFP6_E2M3_A32'], y['stored_MXFP6_E2M3_A32'])
        control_close = torch.allclose(y['reference_W32_A32'], y['wrapper_MX_off_A32'], rtol=1e-5, atol=1e-6)
        export_close = torch.allclose(y['MXFP6_E2M3_A32'], y['stored_MXFP6_E2M3_A32'], rtol=1e-5, atol=1e-6)
        control_predictions = torch.equal(y['reference_W32_A32'].argmax(1), y['wrapper_MX_off_A32'].argmax(1))
        export_predictions = torch.equal(y['MXFP6_E2M3_A32'].argmax(1), y['stored_MXFP6_E2M3_A32'].argmax(1))
        reference_counts_match = rows[0]['correct1'] == expected_metric['correct1'] and rows[0]['correct5'] == expected_metric['correct5']
        unchanged = s.legacy.state_digest(baseline.state_dict()) == baseline_hash and rt.core.file_sha256(path) == digest
        checks[role] = {
            'epoch': entry['epoch'], 'checkpoint_unchanged': unchanged,
            'reference_counts_match': reference_counts_match, 'expected_correct1': expected_metric['correct1'],
            'quantized_weight_tensors_exact': decoded_exact,
            'control_logits_exact': control_equal, 'control_logits_close': control_close,
            'control_predictions_equal': control_predictions,
            'export_logits_exact': export_equal, 'export_logits_close': export_close,
            'export_predictions_equal': export_predictions,
            'control_max_abs_logit_difference': float((y['reference_W32_A32']-y['wrapper_MX_off_A32']).abs().max()),
            'export_max_abs_logit_difference': float((y['MXFP6_E2M3_A32']-y['stored_MXFP6_E2M3_A32']).abs().max()),
            'MX_vs_reference_prediction_changes': int((y['reference_W32_A32'].argmax(1) != y['MXFP6_E2M3_A32'].argmax(1)).sum()),
            'test_seconds_all_four_paths_including_instrumentation': duration,
            'source_export': export_info,
            'no_master_for_encoded_layers_in_export': True,
            'auxiliary_state_bytes': sum(v.numel()*v.element_size() for v in payload['auxiliary_fp32_state'].values()),
        }
        for item in rows:
            item.update(role=role, epoch=entry['epoch'], activation_bits=32)
        for item in layer_rows:
            item.update(role=role, epoch=entry['epoch'])
        rows_all.extend(rows)
        layers_all.extend(layer_rows)
        # Local per-example data for follow-up comparison, not put in feedback ZIP.
        with (out / ('predictions_' + role + '.pt')).open('xb') as f:
            torch.save(arrays, f)
        for item in rows:
            log(f'  {item["mode"]}: {item["count_top1_percent"]:.4f}% ({item["correct1"]}/{item["n"]})')
        del saved, baseline, wrapped_control, online_mx, reloaded, models, arrays, y
    s.write_csv(out / 'metrics.csv', rows_all)
    s.write_csv(out / 'layer_metrics.csv', layers_all)
    report['checks'] = checks
    report['metrics'] = rows_all
    report['reference_report_unchanged'] = s.legacy.sha256(reference_file) == before_report
    report['test_file_unchanged'] = s.hashes(data_file) == data_info
    report['source_unchanged'] = all(s.legacy.sha256(source / n) == h for n,h in source_info['workcopy_files_sha256'].items())
    status_fields = ['checkpoint_unchanged', 'reference_counts_match', 'quantized_weight_tensors_exact',
                     'control_logits_close', 'control_predictions_equal', 'export_logits_close', 'export_predictions_equal']
    report['status'] = 'PASS' if all(all(c[k] for k in status_fields) for c in checks.values()) and all(
        report[k] for k in ['reference_report_unchanged','test_file_unchanged','source_unchanged']) else 'REVIEW_REQUIRED'
    if report['status'] != 'PASS':
        report['warnings'].append('Controle wijkt af. Interpreteer MX-verschillen pas na controle van checks; geen automatische hertraining.')
    notebook(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check', 'evaluate'])
    p.add_argument('--reference-run', type=Path)
    p.add_argument('--data-root', type=Path, default=DEFAULT_PROJECT / 'data' / 'svhn')
    p.add_argument('--output-root', type=Path, default=DEFAULT_PROJECT / 'mx_runs' / 'mx_weights_step1')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.command == 'evaluate' and args.reference_run is None:
        p.error('evaluate vereist --reference-run; geen automatische checkpointkeuze.')
    if args.reference_run and args.output_root.resolve().is_relative_to(args.reference_run.resolve()):
        p.error('Resultaten mogen niet in de referentierun worden geschreven.')
    out = new_directory(args.output_root, args.command)
    log = s.legacy.Log(out / 'run.log')
    start = time.perf_counter()
    report = {'schema': 'mx-weights-step1-v1', 'status': 'STARTED', 'started_at': s.now(),
              'command': args.command, 'output_directory': str(out.resolve()), 'config': CONFIG,
              'warnings': [], 'optimizer_steps': 0, 'new_SVHN_training': False,
              'limitations': ['weight-only direct cast, no QAT accuracy evidence',
                             'original test-selected reference checkpoints; no independent holdout',
                             'FP32 execution buffers: no native MX speed/energy claim',
                             'format choice E2M3 is a first research configuration, not an optimum']}
    try:
        log('RESULTATEN: ' + str(out.resolve()))
        log('CONFIG: MXFP6 E2M3; groep=32; schaal=E8M0; A32/ReLU; BN32; geen training/truncatie.')
        report['package_sha256'] = package_hashes()
        dependencies(out)
        source, info = s.legacy.prepare_source(out)
        rt = s.runtime(20261009, args.device)
        modules = s.import_source(source)
        report.update(environment=rt.environment, execution_settings=rt.settings, source=info)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('once')
            if args.command == 'check':
                from test_mx import run_tests
                report['tests'] = run_tests(modules, str(rt.device))
                report['status'] = 'PASS'
            else:
                evaluate(args, rt, modules, source, info, report, out, log)
            report['warnings'].extend(str(w.message) for w in captured)
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        log(report['traceback'])
    report.update(ended_at=s.now(), session_seconds=time.perf_counter()-start)
    s.write_json(out / 'run_report.json', report)
    feedback(out)
    log('EINDSTATUS: ' + report['status'])
    return 0 if report['status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
