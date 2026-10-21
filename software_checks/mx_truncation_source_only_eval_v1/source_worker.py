"""Fresh-process inference. Receives a high source only, never a master or low file."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings
sys.dont_write_bytecode = True
import svhn_support as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding='utf-8'))
    out = args.output.resolve(); out.mkdir(exist_ok=False)
    started = time.perf_counter()
    log = s.legacy.Log(out/'worker.log')
    report = {'status': 'RUNNING', 'pid': os.getpid(), 'warnings': [],
              'synthetic_fixture_only': job['synthetic_fixture_only'], 'optimizer_steps': 0,
              'training_checkpoint_opened': False, 'saved_low_model_opened': False,
              'batchnorm_recalibrated': False}
    def on_warning(message, category, filename, lineno, file=None, line=None):
        row = {'category': category.__name__, 'message': str(message), 'file': filename, 'line': lineno}
        report['warnings'].append(row); log('WAARSCHUWING: ' + str(message))
    rc = 1
    try:
        import torch
        import numpy as np
        import resume_core as core
        import trunc_adapter as a
        import source_evaluation as ev
        from test_data import prepare_test_only
        from types import SimpleNamespace
        fixture = job['synthetic_fixture_only']
        s.require(type(fixture) is bool, 'Ongeldige fixturemarkering.')
        if fixture and job['device'] == 'cpu': torch.set_num_threads(1)
        with warnings.catch_warnings():
            warnings.simplefilter('default'); warnings.showwarning = on_warning
            report['package_sha256'] = s.verify_package()
            source_code, source_info = s.legacy.prepare_source(out)
            rt = s.runtime(job['seed'], job['device'], strict=not (fixture and job['device'] == 'cpu'))
            report.update(environment=rt.environment, execution_settings=rt.settings)
            # The synthetic CPU branch is only for our test harness, never an
            # automatic fallback when a real evaluation lacks a working GPU.
            if not fixture:
                for key in ('python','torch','torchvision','numpy','scipy','pillow','cuda_build','cudnn','device_name'):
                    s.require(rt.environment[key] == job['environment'][key], 'Workeromgeving wijkt af: ' + key)
                for key in ('cudnn_benchmark','cudnn_deterministic','deterministic_algorithms',
                            'matmul_fp32_precision','cudnn_conv_fp32_precision','AMP','torch_compile'):
                    s.require(rt.settings[key] == job['execution_settings'][key], 'Workerbackend wijkt af: ' + key)
            modules = s.import_source(source_code)
            high_path = args.job.resolve().parent / 'source_E2M3.pt'
            s.require(core.file_sha256(high_path) == job['source_sha256'], 'Onjuiste staged hoge bron.')
            before = core.file_sha256(high_path)
            if fixture:
                with np.load(args.job.resolve().parent/'synthetic_input.npz', allow_pickle=False) as data:
                    x = torch.from_numpy(data['images'].copy())
                    y = torch.from_numpy(data['labels'].copy())
                s.require(1 <= len(y) <= 32, 'Kunstmatige test moet klein blijven.')
                loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y,torch.arange(len(y))),
                           batch_size=3, shuffle=False, num_workers=0)
                n = len(y); test_file = None
            else:
                data_args = SimpleNamespace(data_root=Path(job['data_root']))
                loader, test_file, test_hash = prepare_test_only(data_args, modules, job, out, log)
                n = 26032
            with ev.inference_guards(high_path) as model_files:
                payload = a.load_verified_payload(high_path)
                a.validate_source(payload)
                p = payload['provenance']
                s.require(p['synthetic_fixture_only'] is fixture and p['source_epoch'] == job['epoch']
                          and p['checkpoint_sha256'] == job['checkpoint_sha256'], 'Onjuiste bronprovenance.')
                models, checks, layers = ev.build_models(modules.models, payload, rt.device)
                metrics, paired, data_checks = ev.evaluate_models(models, loader, n, rt.device,
                                             out, job['role'], job['epoch'], log)
                s.require(model_files == [str(high_path.resolve())], 'Onverwacht aantal geladen modellen.')
                report.update(model_files_loaded=model_files, model_file_count=1, **checks, **data_checks)
            if not fixture:
                s.require(data_checks['label_sha256'] == job['dataset_selection']['test']['label_sha256'],
                          'Verkeerde testlabelvolgorde.')
                s.require(s.hashes(test_file) == test_hash, 'Testbestand gewijzigd.')
                report['test_file_unchanged'] = True
            s.require(core.file_sha256(high_path) == before, 'Hoge bron gewijzigd.')
            s.require(all(core.file_sha256(source_code/n) == h for n,h in source_info['workcopy_files_sha256'].items()),
                      'Codewerkcopy gewijzigd.')
            s.verify_package()
            s.write_csv(out/'metrics.csv', metrics)
            s.write_csv(out/'paired_metrics.csv', [paired])
            s.write_csv(out/'layer_checks.csv', layers)
            report.update(status='PASS', metrics=metrics, source_file_unchanged=True,
                          no_direct_low_quantization=True, source_code_unchanged=True)
            rc = 0
    except BaseException as exc:
        report.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL',
                      error=repr(exc), traceback=traceback.format_exc())
        log(report['traceback'])
    finally:
        report['seconds'] = time.perf_counter() - started
        log('WORKER STATUS: ' + report['status']); log.handle.close()
        s.write_json(out/'worker_report.json', report)
    return rc


if __name__ == '__main__': raise SystemExit(main())
