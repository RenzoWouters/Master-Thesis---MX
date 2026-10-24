"""Bounded, independent integration of the real B-trained post-hoc CLI.

Usage: python test_runner_integration.py NEW_OUTPUT SCALE64_FEEDBACK_ZIP

The feedback supplies JSON/CSV metadata only. All model exports and three
images are private synthetic fixtures. The original production pin is checked
against the real metadata, then a private pin copy permits exercising the
unchanged strict inspector with random exports. No production bypass exists.
No real checkpoint, SVHN data, optimizer, backward or BN update is used.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
import evaluation as ev
import joint_adapter as adapter
import mx_e2m1 as low
import mx_e4m3 as high
import mx_format as middle
import resume_core as core
import run_eval as runner
import scale64 as tr
import svhn_support as support
import trunc_adapter as trained

KIND = 'scale64'
ROLES = ('best_E4M3_test', 'last_epoch')
METADATA = ('run_report.json', 'checkpoint_selection.json', 'selected_metrics.csv',
            'epoch_metrics.csv', 'training_summary.json')
ORIGINAL_IMPORT = support.import_source
ORIGINAL_RUNTIME = support.runtime
ORIGINAL_READ = ev.read_export
ORIGINAL_ASSEMBLE = ev.assemble_paths
ORIGINAL_PRIOR = runner.attach_prior_results
ORIGINAL_TRAIN = torch.nn.Module.train


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def tree_hashes(path):
    return {p.relative_to(path).as_posix(): core.file_sha256(p)
            for p in sorted(Path(path).rglob('*')) if p.is_file()}


def fresh_source_import(source):
    for name in tuple(sys.modules):
        if name.split('.')[0] in ('models', 'datasets', 'optimizer', 'utils'):
            del sys.modules[name]
    return ORIGINAL_IMPORT(source)


def forbidden(*args, **kwargs):
    raise AssertionError('Forbidden training/master operation in post-hoc inference')


def eval_only(self, mode=True):
    assert mode is False, 'Evaluation entered training mode'
    return ORIGINAL_TRAIN(self, False)


def cpu_cuda_identity(self, *args, **kwargs):
    return self


def predict(model, images):
    with torch.no_grad():
        return torch.cat([model(images[:2]), model(images[2:])]).detach()


def write_exports(folder, original, sources):
    folder.mkdir(parents=True)
    (folder / 'SYNTHETIC_FIXTURE_ONLY.txt').write_text(
        'PRIVATE SOFTWARE VALIDATION: random exports; no research result.\n', encoding='utf-8')
    report = copy.deepcopy(original)
    report['fixture_synthetic_not_a_research_result'] = True
    payloads = {}
    for role in ROLES:
        payloads[role] = {}
        for bit in (8, 6, 4):
            item = report['mx_exports'][role][str(bit)]
            payload = (copy.deepcopy(sources[role]) if bit == 8 else
                       trained.derive_payload(payloads[role][8], bit, report['mx_exports'][role]['8']['sha256']))
            assert payload['provenance']['synthetic_fixture_only'] is True
            item.update(core.atomic_save(folder / item['file'], payload))
            item['source_file_sha256'] = report['mx_exports'][role]['8']['sha256']
            item['encoding_and_comparison_device'] = 'cpu; SYNTHETIC FIXTURE'
            payloads[role][bit] = payload
    return report, payloads


def main(out: Path, feedback: Path):
    started = time.perf_counter()
    out, feedback = Path(out).resolve(), Path(feedback).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    feedback_digest = core.file_sha256(feedback)
    package_hashes = support.verify_package()
    with ZipFile(feedback) as archive:
        metadata = {name: archive.read(name) for name in METADATA}
    original = json.loads(metadata['run_report.json'])
    production_pin = json.loads((ROOT / 'EXPECTED_REFERENCE.json').read_text())
    assert production_pin['run_report_sha256'] == hashlib.sha256(metadata['run_report.json']).hexdigest()
    for key in ('checkpoint_selection', 'mx_exports', 'initial_model_sha256'):
        assert production_pin[key] == original[key]
    assert original['status'] == 'PASS' and original['completed_epoch'] == 100
    assert original['synthetic_fixture_only'] is False
    assert original['config'] == json.loads((ROOT / 'SCALE64_REFERENCE_PROTOCOL.json').read_text())
    assert original['checkpoint_selection']['best']['epoch'] == 42
    assert original['checkpoint_selection']['last']['epoch'] == 100
    assert tr.TRAININGS == (KIND,) and len(tr.PATHS) == 9 and len(tr.COMPARISONS) == 12

    source_out = out / 'fixture_source'; source_out.mkdir()
    source, source_info = support.legacy.prepare_source(source_out)
    rt = ORIGINAL_RUNTIME(20261009, 'cpu', strict=False)
    modules = fresh_source_import(source)
    images = torch.rand((3, 3, 40, 40), generator=torch.Generator().manual_seed(8951)) * 2 - 1
    sources = {}
    with patch.object(torch.Tensor, 'cuda', cpu_cuda_identity):
        for i, role in enumerate(ROLES):
            torch.manual_seed(9991 + i)
            model = trained.adapt_copy(modules.models.svhnq([1, 2, 4, 8, 32], expand=2)).eval()
            with torch.no_grad():
                for layer in adapter.BN_LAYERS:
                    for j, bank_key in enumerate(('8', '32', '4')):
                        bank = model.get_submodule(layer).bn_dict[bank_key]
                        bank.weight.fill_(.9 + j*.07); bank.bias.fill_(j*.013)
                        bank.running_mean.fill_(j*.011); bank.running_var.fill_(1.1+j*.1)
            choice = original['checkpoint_selection']['best' if i == 0 else 'last']
            provenance = {'source_epoch': choice['epoch'], 'checkpoint_sha256': choice['sha256'],
                'synthetic_fixture_only': True, 'encoding_device': 'cpu',
                'fixture_notice': 'Private random model; not a research result'}
            sources[role] = trained.source_payload(model, provenance)[0]

    guard_folder = out / 'SYNTHETIC_guard_reference'
    guard_report, _ = write_exports(guard_folder, original, sources)
    for name, data in metadata.items(): (guard_folder / name).write_bytes(data)
    write_json(guard_folder / 'run_report.json', guard_report)
    # A private package mirror changes only the pin used by the fixture guard.
    # Actual numerical file hashes and archived protocol/package remain intact.
    guard_root = out / 'SYNTHETIC_guard_package'; guard_root.mkdir()
    for name in ('SCALE64_REFERENCE_PROTOCOL.json', 'mx_format.py', 'mx_e2m1.py', 'mx_e4m3.py',
                 'mx_adapter.py', 'joint_adapter.py', 'joint_support.py', 'svhn_support.py',
                 'smoke_support.py', 'resume_core.py', 'trunc_adapter.py', 'trunc_support.py', 'posthoc_truncation.py'):
        shutil.copy2(ROOT / name, guard_root / name)
    (guard_root / 'provenance').mkdir()
    shutil.copy2(ROOT / 'provenance/scale64_reference100_PACKAGE_SHA256.json',
                 guard_root / 'provenance/scale64_reference100_PACKAGE_SHA256.json')

    def fixture_pin(report):
        result = copy.deepcopy(production_pin)
        result['run_report_sha256'] = core.file_sha256(guard_folder / 'run_report.json')
        result.update({key: report[key] for key in ('checkpoint_selection', 'mx_exports', 'initial_model_sha256')})
        write_json(guard_root / 'EXPECTED_REFERENCE.json', result)

    fixture_pin(guard_report)
    guard_rt = SimpleNamespace(environment=original['environment'], settings=original['execution_settings'])
    guard_tests = {}
    try: runner.inspect_reference(guard_folder, guard_rt, KIND)
    except (AssertionError, RuntimeError) as exc: guard_tests['production_pin_rejects_private_exports'] = str(exc)
    else: raise AssertionError('Production pin accepted a private random reference')
    with patch.object(runner, 'ROOT', guard_root):
        accepted, fingerprints, selected = runner.inspect_reference(guard_folder, guard_rt, KIND)
        assert accepted['fixture_synthetic_not_a_research_result'] and len(selected) == 6
        assert all(core.file_sha256(guard_folder / name) == digest for name, digest in fingerprints.items())
        guard_tests['strict_inspector_accepts_private_pinned_export_fixture'] = True

        def reject_metadata(label, mutation, update_pin=True):
            altered = copy.deepcopy(guard_report); mutation(altered)
            write_json(guard_folder / 'run_report.json', altered)
            if update_pin: fixture_pin(altered)
            try: runner.inspect_reference(guard_folder, guard_rt, KIND)
            except (AssertionError, RuntimeError, ValueError, KeyError) as exc: guard_tests[label] = str(exc)
            else: raise AssertionError('Invalid reference accepted: ' + label)
            finally:
                write_json(guard_folder / 'run_report.json', guard_report); fixture_pin(guard_report)

        reject_metadata('report_identity_changed_without_repin', lambda r:r.update(fixture_extra=True), False)
        reject_metadata('wrong_training_schema', lambda r:r.update(schema='mx-triple-truncation-reference100-report-v1'))
        reject_metadata('unfinished_reference', lambda r:r.update(completed_epoch=99, full_training_completed=False))
        reject_metadata('synthetic_reference', lambda r:r.update(synthetic_fixture_only=True))
        reject_metadata('A16_protocol', lambda r:r['config'].update(activation_bits=16))
        reject_metadata('unsafe_export_path', lambda r:r['mx_exports'][ROLES[0]]['8'].update(file='../outside.pt'))
        reject_metadata('wrong_lower_source_hash', lambda r:r['mx_exports'][ROLES[0]]['6'].update(source_file_sha256='0'*64))
        reject_metadata('wrong_scale_offset', lambda r:r['mx_exports'][ROLES[0]]['6'].update(scale_exponent_offset=5))
        for which, epoch in [('best', 41), ('last', 99)]:
            altered = copy.deepcopy(guard_report); altered['checkpoint_selection'][which]['epoch'] = epoch
            write_json(guard_folder / 'run_report.json', altered); fixture_pin(altered)
            write_json(guard_folder / 'checkpoint_selection.json', altered['checkpoint_selection'])
            try: runner.inspect_reference(guard_folder, guard_rt, KIND)
            except (AssertionError, RuntimeError, ValueError, KeyError) as exc: guard_tests['resigned_wrong_'+which] = str(exc)
            else: raise AssertionError('Changed checkpoint selection accepted')
            finally:
                write_json(guard_folder / 'run_report.json', guard_report); fixture_pin(guard_report)
                (guard_folder / 'checkpoint_selection.json').write_bytes(metadata['checkpoint_selection.json'])
        history_path = guard_folder / 'epoch_metrics.csv'
        for field, value in [('correct1', '999999'), ('lr', '.9')]:
            rows = csv_rows(history_path); rows[0][field] = value
            history_path.unlink(); support.write_csv(history_path, rows)
            try: runner.inspect_reference(guard_folder, guard_rt, KIND)
            except (AssertionError, RuntimeError, ValueError, KeyError) as exc: guard_tests['invalid_history_'+field] = str(exc)
            else: raise AssertionError('Invalid historical metric accepted')
            finally: history_path.write_bytes(metadata['epoch_metrics.csv'])

    # Prior-result attachment is exercised against actual complete metadata.
    prior_out = out / 'prior_attachment'; prior_out.mkdir()
    prior_result = ORIGINAL_PRIOR(original, prior_out)
    assert prior_result['historical_measurements_reused'] is True
    assert prior_result['new_model_evaluations'] == 0
    for field in ('initial_model_sha256', 'transforms'):
        bad = copy.deepcopy(original); bad[field] = 'changed'
        target = out / ('bad_prior_' + field); target.mkdir()
        try: ORIGINAL_PRIOR(bad, target)
        except (AssertionError, RuntimeError) as exc: guard_tests['prior_mismatch_'+field] = str(exc)
        else: raise AssertionError('Mismatched historical comparison accepted')

    eval_folder = out / 'SYNTHETIC_evaluation_reference'
    eval_report, payloads = write_exports(eval_folder, original, sources)
    expected_logits = {}
    with patch.object(torch.Tensor, 'cuda', cpu_cuda_identity):
        for role in ROLES:
            expected_logits[role] = {bit: predict(trained.load_exported_model(modules.models, payloads[role][bit], 'cpu'), images)
                                     for bit in (8, 6, 4)}
    labels = expected_logits[ROLES[0]][8].topk(5, dim=1, largest=True, sorted=True).indices[:, 0].clone()
    label_hash = hashlib.sha256(labels.numpy().astype(np.int64).tobytes()).hexdigest()
    data_file = out / 'SYNTHETIC_three_images.pt'; torch.save({'images':images, 'labels':labels}, data_file)
    data_hashes = support.hashes(data_file)
    loader = torch.utils.data.DataLoader(support.IndexedDataset(torch.utils.data.TensorDataset(images, labels)),
        batch_size=2, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    expected_metrics = {}
    for role in ROLES:
        expected_metrics[role] = {}
        for bit, logits in expected_logits[role].items():
            top5 = logits.topk(5, dim=1, largest=True, sorted=True).indices
            c1, c5 = int((top5[:,0]==labels).sum()), int((top5==labels[:,None]).any(dim=1).sum())
            expected_metrics[role][bit] = dict(n=3, correct1=c1, correct5=c5, top1_percent=100*c1/3,
                count_top1_percent=100*c1/3, top5_percent=100*c5/3,
                objective_loss=float(F.cross_entropy(logits, labels, reduction='none').double().mean()))
    eval_report['synthetic_fixture_only'] = True
    eval_report['config']['test_examples'] = 3
    eval_report['training_summary'].update(test_examples=3, synthetic_fixture_only=True)
    eval_report['dataset_selection']['test'].update(n=3, label_sha256=label_hash,
        labels_histogram=np.bincount(labels.numpy(), minlength=10).tolist())
    eval_report['dataset_files']['files'] = {'SYNTHETIC_three_images.pt':data_hashes}
    eval_report['transforms']['test'] = 'SYNTHETIC already transformed images; no SVHN'
    history = list(csv.DictReader(io.StringIO(metadata['epoch_metrics.csv'].decode('utf-8-sig'))))
    for row in history:
        if row['phase'] != 'test': continue
        row.update(n=3, correct1=0, correct5=0, top1_percent=0., count_top1_percent=0., top5_percent=0., objective_loss=2.3)
        for role in ROLES:
            choice = eval_report['checkpoint_selection']['best' if role==ROLES[0] else 'last']
            if int(row['epoch']) == choice['epoch']: row.update(expected_metrics[role][int(row['weight_bits'])])
    selected = []
    for role in ROLES:
        choice = eval_report['checkpoint_selection']['best' if role==ROLES[0] else 'last']
        choice['selection_score_E4M3'] = expected_metrics[role][8]['top1_percent']
        selected.extend({'role':role, **row} for row in history if row['phase']=='test' and int(row['epoch'])==choice['epoch'])
    support.write_csv(eval_folder / 'epoch_metrics.csv', history)
    support.write_csv(eval_folder / 'selected_metrics.csv', selected)
    for name, value in [('run_report.json',eval_report), ('checkpoint_selection.json',eval_report['checkpoint_selection']),
                        ('training_summary.json',eval_report['training_summary'])]: write_json(eval_folder/name,value)
    before_guard, before_eval = tree_hashes(guard_folder), tree_hashes(eval_folder)
    source_paths = {eval_folder / item['file'] for items in eval_report['mx_exports'].values() for item in items.values()}

    def fixture_inspection(ref, runtime, kind):
        assert ref == eval_folder and kind == KIND
        names = list(METADATA)
        for exports in eval_report['mx_exports'].values():
            for item in exports.values(): names.extend([item['file'], item['file']+'.sha256'])
        return copy.deepcopy(eval_report), {name:core.file_sha256(ref/name) for name in names}, csv_rows(ref/'selected_metrics.csv')

    def fixture_data(args, current_modules, reference, path, log):
        support.write_json(path/'dataset_manifest.json', {'synthetic_fixture_only':True,
            'test_file':{'path':str(data_file),**data_hashes}, 'selection':reference['dataset_selection']['test'],
            'train_read':False,'extra_read':False})
        return loader, data_file, data_hashes

    def fixture_prior(reference, path):
        assert reference['synthetic_fixture_only'] is True
        return ORIGINAL_PRIOR(original, path)

    def fixture_dependencies(path, log):
        (path/'pip_freeze.txt').write_text('Isolated CPU validation; dependencies checked separately.\n')
        (path/'pip_check.txt').write_text('Dependency subprocess substituted only inside test harness.\n')

    def fixture_runtime(seed, requested_device, strict=True):
        assert requested_device in ('cuda','cuda:0')
        return ORIGINAL_RUNTIME(seed,'cpu',strict=False)

    def fixture_read(path, digest):
        payload = ORIGINAL_READ(path,digest)
        if Path(path) in source_paths:
            assert payload['provenance']['synthetic_fixture_only'] is True
            payload['provenance']['synthetic_fixture_only'] = False
        return payload

    def fixture_assemble(models, incoming, hashes, path, role, device, kind):
        for payload in incoming.values():
            assert payload['provenance']['synthetic_fixture_only'] is False
            payload['provenance']['synthetic_fixture_only'] = True
        return ORIGINAL_ASSEMBLE(models,incoming,hashes,path,role,device,kind)

    original_evaluate = runner.evaluate
    def marked_evaluate(args,path,report,runtime,current_modules,log):
        report['limitations'].append('SYNTHETIC CPU SOFTWARE VALIDATION; three artificial images; no SVHN result.')
        try: return original_evaluate(args,path,report,runtime,current_modules,log)
        finally: report['synthetic_fixture_only'] = True

    cases = {}
    def invoke(command, name=None, status='PASS'):
        name = name or command
        args = [str(ROOT/'run_eval.py'),command,'--output-root',str(out/name)]
        if command=='evaluate': args += ['--scale64-run',str(eval_folder)]
        with ExitStack() as stack:
            for target, field, replacement in [
                (sys,'argv',args),(runner,'dependencies',fixture_dependencies),
                (support,'runtime',fixture_runtime),(support,'import_source',fresh_source_import),
                (torch.Tensor,'cuda',cpu_cuda_identity),(torch.optim.Optimizer,'__init__',forbidden),
                (torch.Tensor,'backward',forbidden),(torch.autograd,'backward',forbidden)]:
                stack.enter_context(patch.object(target,field,replacement))
            if command=='evaluate':
                for target, field, replacement in [
                    (runner,'inspect_reference',fixture_inspection),(runner,'prepare_test_only',fixture_data),
                    (runner,'attach_prior_results',fixture_prior),(runner,'evaluate',marked_evaluate),
                    (ev,'read_export',fixture_read),(ev,'assemble_paths',fixture_assemble),
                    (torch.nn.Module,'train',eval_only),(adapter,'effective_reference_weight',forbidden),
                    (adapter.original_adapter,'effective_reference_weight',forbidden),(trained,'effective_reference_weight',forbidden),
                    (high,'encode_tensor',forbidden),(middle,'encode_tensor',forbidden),(low,'encode_tensor',forbidden),
                    (high,'encode_elements',forbidden),(low,'encode_elements',forbidden)]:
                    stack.enter_context(patch.object(target,field,replacement))
            with (out/(name+'_stdout.log')).open('x',encoding='utf-8') as log:
                stack.enter_context(redirect_stdout(log)); code = runner.main()
        paths = list((out/name).glob('*_'+command)); assert len(paths)==1
        path=paths[0]; report=json.loads((path/'run_report.json').read_text())
        assert code==(0 if status=='PASS' else 1) and report['status']==status, (name,report.get('error'),report.get('traceback'))
        assert report['optimizer_steps']==0 and report['batchnorm_recalibration'] is False
        assert report['new_checkpoint_selection'] is False and report['synthetic_fixture_only'] is True
        with ZipFile(path/'feedback_bundle.zip') as archive:
            assert archive.testzip() is None and len(archive.namelist())==len(set(archive.namelist()))
        cases[name]={'status':report['status'],'expected_status':status,'output_directory':str(path),'error':report.get('error')}
        return path,report

    check_path, check_report = invoke('check')
    eval_path, result = invoke('evaluate')
    for key in ('input_files_unchanged','test_file_unchanged','original_checkpoint_selection_unchanged',
                'historical_correct_counts_all_match','derived_models_all_evaluated_from_saved_and_reloaded_codes'):
        assert result[key] is True, key
    metrics,pairs=csv_rows(eval_path/'metrics.csv'),csv_rows(eval_path/'paired_metrics.csv')
    assert len(metrics)==18 and len(pairs)==24
    assert len(csv_rows(eval_path/'weight_pair_metrics.csv'))==216
    assert len(csv_rows(eval_path/'layer_code_metrics.csv'))==108
    assert len(csv_rows(eval_path/'historical_reproduction.csv'))==6
    assert set(result['roles'])=={KIND} and result['train_and_extra_read'] is False
    assert result['references'][KIND]['checkpoint_files_opened'] is False
    assert result['references'][KIND]['FP32_master_weights_opened'] is False
    for role in ROLES:
        detail=result['roles'][KIND][role]
        for flag in ('all_model_and_bn_states_unchanged','all_models_no_grad','all_paths_use_frozen_target_precision_bn',
                     'six_FP32_master_parameters_absent_all_paths','B_same_frozen_state_as_trained',
                     'trained_lower_codes_exact_B_routes','trained_lower_scale_codes_exact_B_routes'):
            assert detail[flag] is True, flag
        assert detail['n_test']==3 and detail['label_sha256']==label_hash
        with np.load(eval_path/detail['per_image_file'],allow_pickle=False) as arrays:
            assert np.array_equal(arrays['indices'],np.arange(3)) and np.array_equal(arrays['labels'],labels.numpy())
            for row in (r for r in metrics if r['role']==role):
                key=row['path']; logits=arrays[key+'__logits']; pred=arrays[key+'__predictions']; ce=arrays[key+'__per_image_ce']
                assert int(row['correct1'])==int((pred==labels.numpy()).sum())
                assert math.isclose(float(row['hard_cross_entropy']),float(ce.astype(np.float64).mean()),rel_tol=0,abs_tol=1e-9)
                if key in ('source8','trained6','trained4'):
                    assert np.array_equal(logits,expected_logits[role][tr.PATH_SPEC[key]['target_bits']].numpy())
            for bit in (6,4):
                assert np.array_equal(arrays[f'B{bit}__logits'],arrays[f'trained{bit}__logits'])
                assert detail['B_same_run_reproduction'][str(bit)]['logits_exact'] is True
            for row in (r for r in pairs if r['role']==role):
                r,o=arrays[row['reference_path']+'__predictions'],arrays[row['other_path']+'__predictions']
                rc,oc=r==labels.numpy(),o==labels.numpy()
                truth={'wrong_to_correct':int((~rc&oc).sum()),'correct_to_wrong':int((rc&~oc).sum()),
                       'both_correct':int((rc&oc).sum()),'both_wrong':int((~rc&~oc).sum()),
                       'prediction_changes':int((r!=o).sum()),'wrong_to_other_wrong':int((~rc&~oc&(r!=o)).sum())}
                assert all(int(row[k])==v for k,v in truth.items())
    notebook=json.loads((eval_path/runner.NOTEBOOK).read_text())
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in notebook['cells'] if c['cell_type']=='code')
    with ZipFile(eval_path/'feedback_bundle.zip') as archive:
        assert all(result['roles'][KIND][r]['per_image_file'] in archive.namelist() for r in ROLES)
        assert all('prior_posthoc/'+name in archive.namelist() for name in ('REFERENCE.json','metrics.csv','paired_metrics.csv','run_report.json'))

    def payload_failure(name,bit,mutation,fragment):
        item=eval_report['mx_exports'][ROLES[0]][str(bit)]; path=eval_folder/item['file']; marker=path.with_suffix('.pt.sha256')
        old_bytes,old_marker,old_item=path.read_bytes(),marker.read_bytes(),copy.deepcopy(item)
        payload=torch.load(path,map_location='cpu',weights_only=True);mutation(payload)
        path.unlink();marker.unlink();item.update(core.atomic_save(path,payload))
        try:
            failed_path,failed_report=invoke('evaluate',name,'FAIL')
            assert fragment.lower() in failed_report['error'].lower(),(name,failed_report['error'])
            assert not (failed_path/'metrics.csv').exists()
        finally:
            path.write_bytes(old_bytes);marker.write_bytes(old_marker);item.clear();item.update(old_item)

    payload_failure('reject_checkpoint_provenance',8,lambda p:p['provenance'].update(checkpoint_sha256='0'*64),'checkpointprovenance')
    payload_failure('reject_hidden_master',8,lambda p:p['auxiliary_fp32_state'].update({adapter.TARGET_LAYERS[0]+'.weight':torch.zeros(1)}),'master')
    payload_failure('reject_B_source_hash',6,lambda p:p['derivation'].update(source_sha256='0'*64),'bronhash')
    def alter_code(payload):
        name=adapter.TARGET_LAYERS[0];mx=middle.MXTensor.from_payload(payload['mx_layers'][name])
        mx.codes.reshape(-1)[0]^=1;payload['mx_layers'][name]=mx.payload()
    payload_failure('reject_B_lower_codes',6,alter_code,'exacte B-afleiding')
    def alter_scale(payload):
        name=adapter.TARGET_LAYERS[0];mx=middle.MXTensor.from_payload(payload['mx_layers'][name])
        mx.scale_codes.reshape(-1)[0]-=1;payload['mx_layers'][name]=mx.payload()
    payload_failure('reject_B_lower_scale',6,alter_scale,'exacte B-afleiding')

    bad_cli=[['run'],['check','--device','cpu'],['check','--download'],['evaluate'],
        ['check','--scale64-run',str(eval_folder)],['evaluate','--direct-run',str(eval_folder)],
        ['evaluate','--scale64-run',str(eval_folder),'--output-root',str(eval_folder/'inside')],
        ['check','--synthetic'],['check','--epochs','5']]
    cli=[]
    for args in bad_cli:
        completed=subprocess.run([sys.executable,str(ROOT/'run_eval.py'),*args],cwd=out,capture_output=True,text=True,timeout=45)
        assert completed.returncode==2,(args,completed.stdout,completed.stderr)
        cli.append({'arguments':args,'exit_code':completed.returncode})
    assert tree_hashes(guard_folder)==before_guard and tree_hashes(eval_folder)==before_eval
    assert support.hashes(data_file)==data_hashes and core.file_sha256(feedback)==feedback_digest
    assert support.verify_package()==package_hashes
    report={'schema':'mx-triple-scale64-trained-posthoc-runner-validation-v1','status':'PASS',
        'synthetic_fixture_only':True,'production_CUDA_or_SVHN_tested':False,
        'scope':'Real check/evaluate main; two random B-trained export fixtures; three images in batches2+1',
        'no_real_checkpoint_or_SVHN_loaded':True,'no_training_or_BN_recalibration':True,
        'production_bypass_added':False,'production_pin_matches_actual_feedback_metadata':True,
        'fixture_only_adaptations':[
            'Isolated CPU runtime; Tensor.cuda initialization alias only inside harness.',
            'Private pin/package mirror for strict-inspector positive fixtures; original pin rejects fixture.',
            'Dataset, dependency commands and already separately tested inspector substituted for three-image CLI run.',
            'Historical attachment tested with original metadata; fixture wrapper supplies that metadata to real attachment function.',
            'At-rest and derived payloads remain synthetic=True; transient read/assemble provenance substitution only inside harness.',
            'Training mode, optimizer, backward, G(W), tensor quantizers and unrelated element quantizers forbidden during evaluate.',
            'No production source modified; reports explicitly synthetic on PASS and expected FAIL.'],
        'runtime_environment':rt.environment,'cases':cases,'reference_guard_tests':guard_tests,'CLI_guards':cli,
        'metric_rows':18,'paired_rows':24,'weight_rows':216,'code_rows':108,'reproduction_rows':6,
        'per_image_files':2,'B_logits_exactly_equal_stored_mode':True,
        'all_source_feedback_dataset_and_package_files_unchanged':True,'notebook_blank_only':True,
        'feedback_sha256':feedback_digest,'test_harness_sha256':core.file_sha256(Path(__file__)),
        'successful_evaluation_directory':str(eval_path),'check_directory':str(check_path),
        'seconds':time.perf_counter()-started}
    write_json(out/'integration_report.json',report)
    print(json.dumps({'status':'PASS','report':str(out/'integration_report.json'),
        'evaluation_directory':str(eval_path),'seconds':report['seconds']},indent=2))
    return report


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
