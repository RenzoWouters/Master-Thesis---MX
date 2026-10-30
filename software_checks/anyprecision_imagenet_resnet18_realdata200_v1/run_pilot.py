"""Bounded ImageNet-ResNet-18 pilot: 200 updates, never an automatic full run."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
import support as s


def parser():
    result = argparse.ArgumentParser(description='ResNet-18: precies 200 ImageNet-trainingsbatches en korte validatie.')
    result.add_argument('command', choices=['data', 'run', 'selftest', '_worker'], metavar='data|run|selftest')
    result.add_argument('--project-root', type=Path, default=s.default_project_root())
    result.add_argument('--output-root', type=Path, default=None)
    result.add_argument('--imagenet-root', type=Path, default=None,
                        help='Voorbereide ILSVRC2012-map met train/<wnid>/ en val/<wnid>/. Standaard project/data/imagenet.')
    result.add_argument('--workers', type=int, choices=range(0, 9), default=4,
                        help='Aantal dataloaderprocessen, standaard 4. Verandert de batchgrootte niet.')
    result.add_argument('--cpu-threads', type=int, choices=[4], default=4)
    result.add_argument('--run-directory', type=Path, help=argparse.SUPPRESS)
    return result


def verify_approved_check():
    """Reuse the actual, supplied R18 PASS rather than repeating a synthetic probe."""
    proof = json.loads((s.ROOT / 'approved_synthetic_check.json').read_text(encoding='utf-8'))
    gpu = proof.get('gpu_check', {})
    s.require(proof.get('package_id') == 'anyprecision_imagenet_resnet18_check_v1' and
              proof.get('status') == 'PASS' and gpu.get('status') == 'PASS' and
              gpu.get('optimizer_updates_completed') == 3 and
              gpu.get('finite_losses_and_gradients') is True and
              gpu.get('timed_out') is False and gpu.get('cleanup_failed') is False,
              'De eerder aangeleverde ResNet-18-geheugencontrole is niet geldig.')
    for name in ('bits', 'global_batch', 'bn_shards', 'execution', 'lr', 'momentum', 'weight_decay',
                 'amp', 'fp32_precision', 'cudnn_benchmark', 'activation_checkpointing', 'seed'):
        s.require(proof['config'].get(name) == s.CONFIG[name],
                  'De nieuwe proef wijkt af van de geslaagde GPU-configuratie: ' + name)
    pinned = proof.get('package_signature', {}).get('files', {})
    for name in ('engine.py', 'resnet18_model.py', 'test_engine.py', 'test_resnet18.py'):
        s.require(pinned.get(name) == s.sha256(s.ROOT / name),
                  'De model/enginecode wijkt af van de geslaagde geheugenproef: ' + name)
    return {'status': 'PASS', 'original_report': proof['output_directory'],
            'original_started_utc': proof['started_utc'], 'source': 'approved_synthetic_check.json',
            'synthetic_probe_repeated': False, 'unchanged_model_engine_and_recipe': True,
            'device': gpu.get('device'), 'packages': proof['environment']['packages']}


def cpu_selftests(sources, log):
    import engine
    import test_engine
    import test_resnet18
    import test_data
    import test_watchdog
    import test_pipeline
    s.configure(s.CONFIG['seed'], cuda=False, threads=4)
    result = {}
    log('CPU-controle: trainingslus, replica-BN en onafhankelijke ResNet-18-referentie.')
    result['engine'] = test_engine.run_selftests()
    result['resnet18'] = test_resnet18.run_tests(sources['anyprecision'], sources['truncquant'], engine_module=engine)
    log('CPU-controle: dataselectie, vaste 200-batchgrens, checkpoint en procesafsluiting.')
    result['data'] = test_data.run_tests()
    result['watchdog'] = test_watchdog.run_tests()
    result['pipeline'] = test_pipeline.run_tests()
    return result


def read_pilot_result(out):
    directory = Path(out) / 'pilot'
    for name in ('result.json', 'progress.json'):
        path = directory / name
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8'))
    return {'status': 'FAIL', 'error': 'Het pilotproces schreef geen rapport.'}


def validate_completed_result(result, out):
    s.require(result.get('optimizer_updates_completed') == 200 and result.get('train_samples') == 51200,
              'PASS vereist precies 200 optimizerupdates en 51.200 trainingsbeelden.')
    steps = result.get('train_steps', [])
    s.require(len(steps) == 200 and [step.get('step') for step in steps] == list(range(1, 201)),
              'De 200 afzonderlijke trainingsmetingen zijn onvolledig.')
    for step in steps:
        s.require(step.get('samples') == 256, 'Een trainingsmeting heeft geen batch256.')
        for key in ('data_wait_seconds', 'transfer_seconds', 'compute_seconds', 'total_seconds'):
            value = step.get(key)
            s.require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                      'Ongeldige tijdmeting: ' + key)
        s.require(step['compute_seconds'] > 0 and math.isclose(
            step['total_seconds'], sum(step[key] for key in
                                      ('data_wait_seconds', 'transfer_seconds', 'compute_seconds')),
            rel_tol=1e-7, abs_tol=1e-7), 'De tijdcomponenten tellen niet op tot de totale batchduur.')
    validation = result.get('validation', {})
    s.require(validation.get('status') == 'PASS' and validation.get('samples') == 5000 and
              validation.get('batches') == 79 and set(validation.get('by_bit', {})) == {str(b) for b in s.BITS},
              'De validatie van 5.000 beelden op alle vijf precisies is onvolledig.')
    for values in validation['by_bit'].values():
        s.require(values.get('samples') == 5000 and math.isfinite(values['loss']) and
                  0 <= values['top1'] <= values['top5'] <= 100,
                  'Ongeldige korte-validatiemetrics.')
    s.require(result.get('bn_training_check', {}).get('status') == 'PASS' and
              result.get('bn_validation_unchanged', {}).get('status') == 'PASS' and
              result.get('finite_final_state') is True and result.get('source_files_unchanged') is True,
              'De eindcontrole van BN, eindige toestand of bronintegriteit ontbreekt.')
    checkpoint = result.get('checkpoint', {})
    path = Path(out) / 'pilot' / 'checkpoint_step0200.pt'
    s.require(checkpoint.get('filename') == path.name and checkpoint.get('kind') == 'diagnostic_partial_epoch'
              and checkpoint.get('eligible_for_reference_resume') is False and path.is_file()
              and checkpoint.get('size_bytes') == path.stat().st_size and path.stat().st_size > 0,
              'Het bewaarde diagnosecheckpoint ontbreekt of is onvolledig.')
    s.require(result.get('completed_epochs') == 0 and result.get('eligible_for_reference_resume') is False,
              'Deze beperkte proef mag niet als een volledige epoch worden aangemerkt.')


def execute_pilot(args, out, log):
    from process_watchdog import run_bounded_process
    command = [sys.executable, '-u', str(s.ROOT / 'run_pilot.py'), '_worker',
               '--run-directory', str(Path(out).resolve()), '--imagenet-root', str(args.imagenet_root),
               '--workers', str(args.workers), '--cpu-threads', str(args.cpu_threads)]
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    log('Start korte ImageNet-proef: maximaal 200 updates; tijdslimiet voor dit proces 20 minuten.')
    watched = run_bounded_process(command, timeout_seconds=1200., heartbeat_seconds=15.,
                                  label='ResNet-18 echte data', log=log, env=environment)
    result = read_pilot_result(out)
    result.update(watched)
    result['eligible_for_reference_resume'] = False
    if watched['cleanup_failed']:
        result.update(status='FAIL', error='Niet alle eigen pilotprocessen konden worden gestopt; zie procesrapport.')
    elif watched['timed_out']:
        result.update(status='TIMEOUT', error='De 20-minutengrens is bereikt. Voltooide deelmetingen zijn behouden.',
                      peak_observation_scope='Only completed recorded boundaries; aborted-update final peak unknown.')
    else:
        s.require((result.get('status') == 'PASS') == (watched['process_returncode'] == 0),
                  'De exitcode en het pilotrapport spreken elkaar tegen.')
        s.require(result.get('status') in ('PASS', 'FAIL', 'OOM', 'INTERRUPTED'), 'Onvolledig piloteindrapport.')
        if result['status'] == 'PASS':
            validate_completed_result(result, out)
    (Path(out) / 'pilot').mkdir(exist_ok=True)
    s.json_write(Path(out) / 'pilot' / 'watchdog_result.json', result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == '_worker':
        s.verify_fixed_contract()
        s.require(args.run_directory is not None and args.imagenet_root is not None,
                  'Het interne pilotproces vereist de door de ouder voorbereide runmap en dataset.')
        from pilot_worker import run_worker
        return run_worker(args)

    out = s.new_output(args.project_root, args.command, args.output_root)
    log = s.Log(out / 'console.log')
    began = time.perf_counter()
    report = {'schema_version': 1, 'package_id': s.CONFIG['package_id'],
              'command': s.CONFIG['command_kind'], 'requested_command': args.command, 'status': 'RUNNING',
              'started_utc': s.utc_now(), 'output_directory': str(out), 'config': s.CONFIG,
              'runtime_options': {'workers': args.workers, 'cpu_threads': args.cpu_threads},
              'package_signature': s.package_signature(), 'completed_epochs': 0,
              'scope': 'Fresh diagnostic training prefix: 200 global batches and 5,000 stratified validation images.',
              'eligible_for_reference_resume': False}
    log('RESULTATENMAP: ' + str(out))
    log('RESNET-18 / IMAGENET: 200 batches256; BN64 x4; W/A 32,8,4,2,1; FP32; SGD lr=0.3.')
    log('Gedocumenteerde reconstructie; de volledige training begint later opnieuw vanaf nul.')
    try:
        s.verify_fixed_contract()
        s.verify_sources()
        s.copy_context(out)
        report['approved_synthetic_check'] = verify_approved_check()
        if args.command in ('data', 'run'):
            from data_support import audit_imagenet
            args.imagenet_root = (args.imagenet_root or args.project_root / 'data' / 'imagenet').expanduser().resolve()
            log('Eerst ImageNet-inventaris controleren: ' + str(args.imagenet_root))
            data_started = time.perf_counter()
            report['dataset'] = audit_imagenet(args.imagenet_root, strict=True, progress=log)
            report['dataset_audit_seconds'] = time.perf_counter() - data_started
            s.json_write(out / 'dataset_manifest.json', report['dataset'])
            s.json_write(out / 'run_report.json', report)
        if args.command == 'data':
            report['status'] = 'PASS'
            log('DATA PASS: volledige ImageNet-indeling en steekproefdecodes gecontroleerd; geen training uitgevoerd.')
        else:
            report['environment'] = s.check_environment(log)
            sources, report['source'] = s.prepare_sources(out)
            report['selftests'] = cpu_selftests(sources, log)
            s.json_write(out / 'run_report.json', report)
            gc.collect()
            if args.command == 'selftest':
                report['status'] = 'PASS'
                log('CPU-zelftests geslaagd; geen dataset- of GPU-training uitgevoerd.')
            else:
                report['pilot'] = execute_pilot(args, out, log)
                report['status'] = report['pilot']['status']
                reference = report['approved_synthetic_check']
                report['comparison_context'] = {
                    'same_gpu_as_synthetic_check': reference['device'] == report['pilot'].get('device'),
                    'same_package_versions': all(report['environment']['packages'][name] == reference['packages'][name]
                                                 for name in ('torch', 'torchvision', 'numpy', 'Pillow')),
                    'scope': 'Real input and data I/O now included; first 20 updates contain startup/warmup effects.'}
    except KeyboardInterrupt:
        report.update(status='INTERRUPTED', error='Proef onderbroken; voltooide deelmetingen blijven bewaard.')
        if (out / 'pilot').is_dir():
            report['pilot'] = read_pilot_result(out)
            report['pilot']['status'] = 'INTERRUPTED'
        log(report['error'])
    except Exception as exc:
        report.update(status='FAIL', error=str(exc), traceback=traceback.format_exc())
        if (out / 'pilot').is_dir():
            report['pilot'] = read_pilot_result(out)
            if report['pilot'].get('status') == 'PASS':
                report['pilot']['status'] = 'FAIL'
                report['pilot']['parent_validation_error'] = str(exc)
        log('FAIL: ' + str(exc))
        log(report['traceback'])
    finally:
        report['ended_utc'] = s.utc_now()
        report['wall_seconds'] = time.perf_counter() - began
        s.json_write(out / 'run_report.json', report)
        log('EINDSTATUS: ' + report['status'])
        log('Deel feedback_bundle.zip uit deze map. Het diagnosecheckpoint en ImageNet blijven lokaal.')
        log.close()
        s.feedback_bundle(out)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
