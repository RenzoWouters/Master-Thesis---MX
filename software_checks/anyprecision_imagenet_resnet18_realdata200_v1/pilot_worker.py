"""Exactly 200 fresh ImageNet updates and a stratified 5,000-image evaluation."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
import time
import traceback

import support as s
import runtime_checks as checks


def _cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def save_diagnostic_checkpoint(out, model, optimizer, metadata):
    """Keep the diagnostic state locally; it is not a full-epoch resume point."""
    import numpy as np
    import torch
    began = time.perf_counter()
    destination = Path(out) / 'checkpoint_step0200.pt'
    s.require(not destination.exists(), 'Het diagnosecheckpoint bestaat al; overschrijven geweigerd.')
    temporary = destination.with_suffix('.pt.tmp')
    numpy_state = np.random.get_state()
    saved = {
        'schema_version': 1, 'package_id': s.CONFIG['package_id'],
        'kind': 'diagnostic_partial_epoch', 'completed_epochs': 0, 'global_step': 200,
        'eligible_for_reference_resume': False, 'full_training_must_start_from_zero': True,
        'model': _cpu_tree(model.state_dict()), 'optimizer': _cpu_tree(optimizer.state_dict()),
        'config': s.CONFIG, 'metadata': metadata,
        'rng': {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all(),
                'numpy': {'bit_generator': numpy_state[0], 'keys': numpy_state[1].tolist(),
                          'position': int(numpy_state[2]), 'has_gauss': int(numpy_state[3]),
                          'cached_gaussian': float(numpy_state[4])}},
        'resume_note': 'Diagnostic snapshot only; worker-prefetch state is not serialized and exact continuation is not claimed.'}
    with temporary.open('xb') as stream:
        torch.save(saved, stream)
    os.replace(temporary, destination)
    digest = s.sha256(destination)
    return {'filename': destination.name, 'path': str(destination.resolve()),
            'size_bytes': destination.stat().st_size, 'sha256': digest,
            'seconds': time.perf_counter() - began, 'kind': 'diagnostic_partial_epoch',
            'eligible_for_reference_resume': False, 'global_step': 200}


def write_csv(out, result):
    """CSV is supplementary; the atomically written JSON retains partial runs."""
    fields = ['step', 'samples', 'data_wait_seconds', 'transfer_seconds', 'compute_seconds', 'total_seconds']
    with (Path(out) / 'step_times.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: step[key] for key in fields} for step in result['train_steps'])
    with (Path(out) / 'train_metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['step', 'bit', 'samples', 'loss', 'top1', 'top5']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in result['train_steps']:
            for bit, value in step['by_bit'].items():
                n = value['samples']
                writer.writerow({'step': step['step'], 'bit': bit, 'samples': n, 'loss': value['loss'],
                                 'top1': 100 * value['top1_correct'] / n,
                                 'top5': 100 * value['top5_correct'] / n})


def run_worker(args):
    import torch
    import engine
    from data_support import build_loaders
    from resnet18_model import model_report

    run_directory = Path(args.run_directory).resolve()
    out = run_directory / 'pilot'
    out.mkdir(exist_ok=False)
    log = s.Log(out / 'worker.log')
    began = time.perf_counter()
    result = {
        'status': 'RUNNING', 'model': 'ResNet-18 reconstructed', 'synthetic_data': False,
        'global_batch': 256, 'batchnorm_shards': [64] * 4, 'bits': list(s.BITS),
        'optimizer_updates_completed': 0, 'train_samples': 0, 'completed_epochs': 0,
        'train_steps': [], 'memory_snapshots': [], 'validation': {'status': 'NOT_STARTED'},
        'eligible_for_reference_resume': False,
        'timing_scope': 'Per-step total = data wait + synchronized H2D + five-precision forward/backward/SGD; '
                        'excludes subsequent checks, JSON/logging and memory telemetry. train_loop_seconds includes those.',
        'accuracy_scope': 'Early diagnostic training metrics and stratified 5,000-image validation; no converged paper comparison.',
        'memory_counter_scope': 'PyTorch counters and CUDA reported free memory; not a direct Windows physical-residency measurement.',
        'peak_allocated_bytes': None, 'peak_reserved_bytes': None,
    }
    loader_iterator = None
    code = 1

    def save_progress(stage):
        result['stage'] = stage
        result['elapsed_seconds'] = time.perf_counter() - began
        s.json_write(out / 'progress.json', result)

    def memory_snapshot(phase, step):
        free, total = torch.cuda.mem_get_info()
        record = {'phase': phase, 'step': step,
                  'allocated_bytes': int(torch.cuda.memory_allocated()),
                  'reserved_bytes': int(torch.cuda.memory_reserved()), 'free_bytes': int(free),
                  'device_memory_bytes': int(total),
                  'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                  'peak_reserved_bytes': int(torch.cuda.max_memory_reserved())}
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            old = result[key]
            result[key] = record[key] if old is None else max(old, record[key])
        result['memory_snapshots'].append(record)
        return record

    try:
        sources = s.verify_source_copy(run_directory / 'source_workcopy')
        audit = json.loads((run_directory / 'dataset_manifest.json').read_text(encoding='utf-8'))
        s.require(audit.get('strict') is True and audit.get('train_count') == 1281167 and
                  audit.get('val_count') == 50000 and audit.get('class_count') == 1000,
                  'De volledige ImageNet-inventaris ontbreekt of is onjuist.')
        s.require(Path(audit['root']).resolve() == Path(args.imagenet_root).resolve(),
                  'Het datapad komt niet overeen met de voorafgaande controle.')
        save_progress('initialization')
        log('Start verse ResNet-18; exact 200 echte trainingsbatches, daarna 5.000 validatiebeelden.')
        start = time.perf_counter()
        result['runtime'] = s.configure(s.CONFIG['seed'], threads=args.cpu_threads)
        result['device'] = result['runtime']['device']
        free, total = torch.cuda.mem_get_info()
        result['device_memory_bytes'] = int(total)
        result['free_memory_before_model_bytes'] = int(free)
        log('GPU: ' + result['device']['name'])
        torch.cuda.reset_peak_memory_stats()
        model, optimizer, soft = checks.model_and_optimizer(sources, 'cuda:0')
        result['architecture'] = model_report(model)
        torch.cuda.synchronize()
        result['initialization_seconds'] = time.perf_counter() - start
        memory_snapshot('model', 0)
        save_progress('build_loaders')
        log('ImageFolder en begrensde dataloaders opbouwen; de volledige dataset blijft ongewijzigd.')
        start = time.perf_counter()
        train_loader, val_loader, selection = build_loaders(
            args.imagenet_root, workers=args.workers, seed=s.CONFIG['seed'], audit_manifest=audit, progress=log)
        result['data_loader_build_seconds'] = time.perf_counter() - start
        result['data_selection'] = selection
        s.require(len(train_loader) == 200 and len(train_loader.dataset) == 51200,
                  'De trainingsloader moet precies 200 batches van 256 bevatten.')
        s.require(len(val_loader) == 79 and len(val_loader.dataset) == 5000,
                  'De validatieloader moet precies 5.000 beelden in 79 batches bevatten.')
        s.require(train_loader.batch_size == 256 and val_loader.batch_size == 64,
                  'De vaste train- of validatiebatch is gewijzigd.')
        before_bn = checks.bn_snapshot(model)
        totals = checks.fresh_totals()
        save_progress('start_train_loader')
        log(f'Dataloader-workers starten: {args.workers}; daarna batch 1/200.')
        start = time.perf_counter()
        loader_iterator = iter(train_loader)
        result['loader_start_seconds'] = time.perf_counter() - start
        train_start = time.perf_counter()
        for step in range(1, 201):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            x, y = next(loader_iterator)
            t1 = time.perf_counter()
            s.require(x.shape[0] == 256 and tuple(x.shape[1:]) == (3, 224, 224),
                      'Onverwachte batchvorm; geen kleinere trainingsbatch toegestaan.')
            x = x.to('cuda:0', non_blocking=True)
            y = y.to('cuda:0', non_blocking=True)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            metrics = engine.train_step(model, optimizer, x, y, soft, mode='serial_dp4', validate=(step == 1))
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            checks.check_metrics(metrics, training=True, expected_samples=256)
            checks.add_totals(totals, metrics)
            result['train_steps'].append({
                'step': step, 'samples': 256, 'data_wait_seconds': t1 - t0,
                'transfer_seconds': t2 - t1, 'compute_seconds': t3 - t2,
                'total_seconds': t3 - t0, 'by_bit': metrics['by_bit']})
            result['optimizer_updates_completed'] = step
            result['train_samples'] = step * 256
            memory_snapshot('train', step)
            if step == 1:
                result['bn_first_update_check'] = checks.check_bn_training(before_bn, checks.bn_snapshot(model), 1)
            save_progress('training')
            if step == 1 or step % 10 == 0:
                top1 = 100 * totals['32']['top1_correct'] / totals['32']['samples']
                log(f'Train {step}/200: W32/A32 cumulatief top1={top1:.3f}%; '
                    f'laatste batch totaal {t3-t0:.3f} s, wachten {t1-t0:.3f} s, GPU-stap {t3-t2:.3f} s.')
            del x, y
        result['train_loop_seconds'] = time.perf_counter() - train_start
        # This bounded dataset has no 201st minibatch; exhaust once to let workers close normally.
        try:
            next(loader_iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError('De begrensde dataloader bevat onverwacht een 201ste batch.')
        loader_iterator = None
        result['train_summary'] = checks.summarize_totals(totals)
        result['bn_training_check'] = checks.check_bn_training(before_bn, checks.bn_snapshot(model), 200)
        checks.assert_finite_state(model, optimizer)
        save_progress('checkpoint')
        log('200 updates voltooid. Diagnosecheckpoint opslaan; dit is geen voltooide epoch.')
        result['checkpoint'] = save_diagnostic_checkpoint(out, model, optimizer, {
            'data_selection': selection, 'dataset_class_to_idx_sha256': audit['class_to_idx_sha256'],
            'source_manifest': json.loads((run_directory / 'source_manifest.json').read_text(encoding='utf-8')),
            'package_signature': s.package_signature()})
        save_progress('validation')
        log('Korte validatie: vijf beelden per klasse, alle vijf precisies, bevroren BatchNorm.')
        before_validation = checks.bn_snapshot(model)
        val_totals = checks.fresh_totals()
        result['validation'] = {'status': 'RUNNING', 'samples': 0, 'batches': 0}
        val_started = time.perf_counter()
        loader_iterator = iter(val_loader)
        for index, (x, y) in enumerate(loader_iterator, start=1):
            torch.cuda.reset_peak_memory_stats()
            x = x.to('cuda:0', non_blocking=True)
            y = y.to('cuda:0', non_blocking=True)
            metrics = engine.eval_batch(model, x, y)
            torch.cuda.synchronize()
            checks.check_metrics(metrics, training=False, expected_samples=int(x.shape[0]))
            checks.add_totals(val_totals, metrics)
            result['validation'].update(samples=val_totals['32']['samples'], batches=index,
                                         seconds=time.perf_counter() - val_started)
            memory_snapshot('validation', index)
            save_progress('validation')
            if index == 1 or index % 20 == 0 or index == 79:
                log(f'Validatie {index}/79: {val_totals["32"]["samples"]}/5000 beelden.')
            del x, y
        torch.cuda.synchronize()
        loader_iterator = None
        s.require(result['validation']['samples'] == 5000 and result['validation']['batches'] == 79,
                  'De korte validatie is niet volledig uitgevoerd.')
        result['validation'].update(status='PASS', seconds=time.perf_counter() - val_started,
                                     by_bit=checks.summarize_totals(val_totals),
                                     subset='Five images per each of the 1,000 classes; not the full 50,000-image validation.')
        result['bn_validation_unchanged'] = checks.check_bn_unchanged(before_validation, checks.bn_snapshot(model))
        checks.assert_finite_state(model, optimizer)
        result['finite_final_state'] = True
        s.verify_source_copy(run_directory / 'source_workcopy')
        result['source_files_unchanged'] = True
        result.update(status='PASS', total_worker_seconds=time.perf_counter() - began)
        save_progress('completed')
        code = 0
        log('PILOT PASS: 200 trainingsbatches en 5.000 validatiebeelden. Geen volledige epoch uitgevoerd.')
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='OOM', error=str(exc), traceback=traceback.format_exc())
        code = 3
        log('GPU-geheugenallocatie mislukt. De batch en precisies zijn niet automatisch gewijzigd.')
    except KeyboardInterrupt:
        result.update(status='INTERRUPTED', error='De korte proef is onderbroken.')
        log(result['error'])
    except Exception as exc:
        result.update(status='FAIL', error=str(exc), traceback=traceback.format_exc())
        log('PILOT FAIL: ' + str(exc))
        log(result['traceback'])
    finally:
        # Normal iterator destruction shuts down non-persistent DataLoader workers.
        # The parent additionally owns the whole process tree for forced termination.
        loader_iterator = None
        if torch.cuda.is_initialized():
            try:
                memory_snapshot('final', result['optimizer_updates_completed'])
            except Exception as exc:
                result['memory_report_error'] = str(exc)
        result['ended_utc'] = s.utc_now()
        result['total_worker_seconds'] = time.perf_counter() - began
        s.json_write(out / 'result.json', result)
        try:
            write_csv(out, result)
        except Exception as exc:
            log('CSV kon niet worden geschreven; JSON bevat de meetgegevens: ' + str(exc))
        log('WORKER STATUS: ' + result['status'])
        log.close()
    return code
