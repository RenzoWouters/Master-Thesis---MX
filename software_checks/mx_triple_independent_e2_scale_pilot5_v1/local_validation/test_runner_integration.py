"""Integration-only CPU fixtures, never SVHN measurements or production overrides.

Run this file with an unused output directory. A separate oracle process first
constructs the unmodified author model on CPU. The actual check entry uses that
CPU oracle only inside a mocked in-memory reference, because its initialization
hash differs from the pinned Windows/CUDA observation. Raw references and the
production guard are not modified. The isolated CPU venv has no pip, so its
distribution inventory replaces only the production pip subprocess checks in
this fixture; pip health and strict Windows/CUDA environment gates are not tested.
Three synthetic images exercise the actual
main/loop/optimizer/scheduler/export/checkpoint/resume paths.
"""
from __future__ import annotations
from contextlib import redirect_stdout, ExitStack
import argparse
import copy
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import torch
import run_train as r
import svhn_support as s
import trunc_adapter as a
import trunc_support as t
import resume_core as core


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def cpu_cuda(tensor, *args, **kwargs):
    assert tensor.device.type == 'cpu'
    return tensor


def forbidden(*args, **kwargs):
    raise AssertionError('A direct lower encoder was invoked by the truncation route.')


def oracle_process(destination):
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source, source_info = s.legacy.prepare_source(destination)
    modules = s.import_source(source)
    rt = s.runtime(r.config()['seed'], 'cpu', strict=False)
    original = modules.models.svhnq(s.BITS, num_classes=10, expand=2).to(rt.device)
    result = {'scope': 'unmodified author model initialization on CPU, test-only oracle',
              'initial_base_model_sha256': s.legacy.state_digest(original.state_dict()),
              'source_files_sha256': source_info['workcopy_files_sha256'],
              'environment': rt.environment}
    write_json(destination / 'oracle.json', result)


def check_entry(destination, oracle_file):
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    original_runtime, original_reference = s.runtime, r.load_initialization_reference
    oracle = read_json(oracle_file)
    package_before = s.verify_package()
    historical = original_reference()
    historical_base = historical['history']['initial_base_model_sha256']

    def cpu_runtime(seed, requested_device):
        assert requested_device in ('cuda', 'cuda:0')
        return original_runtime(seed, 'cpu', strict=False)

    def cpu_oracle_reference(*args, **kwargs):
        reference = copy.deepcopy(original_reference(*args, **kwargs))
        reference['history']['initial_base_model_sha256'] = oracle['initial_base_model_sha256']
        reference['report']['initial_base_model_sha256'] = oracle['initial_base_model_sha256']
        reference['kind'] = 'explicit_test_only_unmodified_author_CPU_oracle'
        return reference

    def fixture_dependencies(out, log):
        inventory = {dist.metadata['Name']: dist.version for dist in metadata.distributions()
                     if dist.metadata.get('Name')}
        write_json(out / 'validation_dependency_inventory.json', inventory)
        log('CPU VALIDATIE: metadata-inventaris; pip-subprocesschecks niet uitgevoerd (deze testvenv bevat geen pip).')

    with (destination / 'stdout.log').open('x', encoding='utf-8') as stream:
        with redirect_stdout(stream), patch.object(s, 'runtime', cpu_runtime), \
             patch.object(r, 'dependencies', fixture_dependencies), \
             patch.object(r, 'load_initialization_reference', cpu_oracle_reference), \
             patch.object(torch.Tensor, 'cuda', cpu_cuda), \
             patch.object(sys, 'argv', [str(ROOT/'run_train.py'), 'check', '--output-root', str(destination/'output')]):
            result = r.main()
    directories = list((destination/'output').glob('*_joint_check'))
    assert len(directories) == 1
    path = directories[0]
    report = read_json(path/'run_report.json')
    assert result == 0 and report['status'] == 'PASS', (report.get('error'), report.get('traceback'))
    assert report['synthetic_checks_only'] and report['dataset_downloaded'] is False
    assert report['initialization_verification']['initial_base_model_sha256'] == oracle['initial_base_model_sha256']
    assert report['synthetic_training_checks']['status'] == 'PASS'
    assert (path / r.NOTEBOOK).is_file() and (path/'feedback_bundle.zip').is_file()
    assert s.verify_package() == package_before
    assert original_reference()['history']['initial_base_model_sha256'] == historical_base
    write_json(destination/'check_entry_report.json', {
        'status': 'PASS', 'scope': 'actual main/check on CPU, test-only runtime and explicit CPU oracle substitution',
        'production_initialization_guard_unchanged': True, 'historical_reference_unchanged': True,
        'production_dependency_checks_unchanged': True, 'pip_health_not_exercised_in_CPU_fixture': True,
        'reason_for_test_dependency_inventory': 'isolated CPU validation venv contains no pip module',
        'historical_Windows_initial_hash': historical_base,
        'test_only_CPU_oracle_hash': oracle['initial_base_model_sha256'],
        'source_module_imports_isolated_in_fresh_process': True,
        'actual_output': str(path), 'test_report': report['synthetic_training_checks'],
        'no_SVHN_training': True})


def verify_exports(path, report, modules):
    count = 0
    for role in ('best_E4M3_test', 'last_epoch'):
        details = report['mx_exports'][role]
        payloads = {bit: a.load_verified_payload(path/details[str(bit)]['file']) for bit in (8,6,4)}
        lower = payloads[8]['lower_scale_codes']
        assert sum(v.numel() for v in lower.values()) == 4080
        for name in a.TARGET_LAYERS:
            high, mid, low = [a.codec(bit).MXTensor.from_payload(payloads[bit]['mx_layers'][name])
                              for bit in (8,6,4)]
            u = high.codes.long()
            expected = ((u >> 7) << 5) | (((u >> 5) & 3) << 3) | (u & 7)
            assert torch.equal(mid.codes.long(), expected)
            assert torch.equal(low.codes.long(), expected >> 2)
            assert lower[name].dtype == torch.uint8 and torch.equal(mid.scale_codes, lower[name])
            assert torch.equal(low.scale_codes, lower[name])
            for payload in payloads.values():
                aux = payload['auxiliary_fp32_state']
                assert name+'.weight' not in aux and name+'.quantize_fn.phi' not in aux
            count += 1
        images = torch.linspace(-1,1,2*3*40*40).reshape(2,3,40,40)
        with ExitStack() as stack:
            stack.enter_context(patch.object(a, 'effective_reference_weight', forbidden))
            for bit in (8,6,4):
                stack.enter_context(patch.object(a.codec(bit), 'encode_tensor', forbidden))
                stack.enter_context(patch.object(a.codec(bit), 'encode_elements', forbidden))
            for bit in (8,6,4):
                source_model = a.load_exported_model(modules.models, payloads[8], torch.device('cpu'), bit)
                own_model = a.load_exported_model(modules.models, payloads[bit], torch.device('cpu'), bit)
                with torch.no_grad():
                    assert torch.equal(source_model(images), own_model(images))
    return {'selected_source_layer_pairs_checked': count,
            'absolute_E2_scale_bytes': 4080, 'source_only_reload_logits_exact': True,
            'masters_and_phi_absent': True}


def verify_scale_tables(path, trace):
    states = {0: trace['initial_scale_state'], **{r['epoch']:r['scale_state'] for r in trace['epochs']}}
    sources = {0: trace['initial_source_scale_state'], **{r['epoch']:r['source_scale_state'] for r in trace['epochs']}}
    with (path/'scale_block_state.csv').open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4080 * len(states)
    seen = set()
    for row in rows:
        e, n, c, b = int(row['epoch']), row['layer'], int(row['output_channel']), int(row['block_index'])
        key = e,n,c,b
        assert key not in seen
        seen.add(key)
        phi = states[e][n][c][b]
        source = sources[e][n][c][b]-127
        # Python float represents the original FP32 value exactly.
        lower = math.floor(phi)+(phi-math.floor(phi)>=.5)
        assert float(row['phi']) == phi and int(row['e2_exponent']) == lower
        assert int(row['source_exponent']) == source
        assert int(row['relative_offset']) == lower-source
        assert float(row['relative_factor']) == math.ldexp(1., lower-source)
    return {'rows': len(rows), 'all_absolute_and_observed_relative_values_match_trace': True}


def main(destination):
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.perf_counter()
    # Separate processes prevent an earlier source workcopy polluting the real check import guard.
    for command, extra in [('oracle', []), ('check-entry', [str(destination/'oracle/oracle.json')])]:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), command,
                               str(destination/command), *extra], capture_output=True, text=True)
        assert proc.returncode == 0, (command, proc.stdout[-4000:], proc.stderr[-6000:])
    config, package_hashes = r.config(), s.verify_package()
    assert config['planned_epochs'] == config['reference_end_epoch'] == 100 and config['pilot_stop_epoch'] == 5
    assert config['scheduler_milestones'] == [50,75,90] and config['activation_bits'] == 32
    assert config['scale_proxy_bounds'] == [-127.,127.] and not config['relative_factor_constrained_after_initialization']
    references = {p.relative_to(ROOT).as_posix(): core.file_sha256(p)
                  for directory in r.REFERENCE_DIRECTORIES for p in (ROOT/directory).rglob('*') if p.is_file()}
    prepared = destination/'source'; prepared.mkdir()
    source, source_info = s.legacy.prepare_source(prepared)
    modules = s.import_source(source)
    oracle = read_json(destination/'oracle/oracle.json')
    generator = torch.Generator().manual_seed(4837)
    images = torch.rand(3,3,40,40,generator=generator)*2-1
    labels = torch.tensor([0,1,2])
    datasets = {phase: torch.utils.data.TensorDataset(images.clone(),labels.clone()) for phase in ('train','test')}
    cases, contexts = {}, {}

    def fixture_context(args, out, report, log, check_only=False):
        assert not check_only and args.stop_after_epoch == 5 and not args.download
        rt = s.runtime(config['seed'], 'cpu', strict=False)
        c = r.config(); c.update(train_examples=3, test_examples=3, batch_size=2,
                                 purpose='SYNTHETIC_CPU_INDEPENDENT_E2_INTEGRATION_ONLY')
        model = t.build_model(modules, 'joint', rt.device)
        assert s.legacy.state_digest(a.base_state_dict(model)) == oracle['initial_base_model_sha256']
        optimizer = t.build_optimizer(modules, model, c)
        scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
        name = out.parent.name
        cases[name]['fresh_model_sha256'] = s.legacy.state_digest(model.state_dict())
        def count_step(opt, hook_args, hook_kwargs):
            cases[name]['updates'] += 1
        optimizer.register_step_post_hook(count_step)
        loaders = {phase:s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(dataset),
                   batch_size=2,shuffle=phase=='train',num_workers=0,pin_memory=False,drop_last=False))
                   for phase,dataset in datasets.items()}
        contract = {'config':c,'environment':rt.environment,'settings':rt.settings,
                    'source_files':source_info['workcopy_files_sha256'],'package_sha256':package_hashes,
                    'synthetic_fixture_only':True}
        report.update(config=c,environment=rt.environment,execution_settings=rt.settings,source=source_info,
                      package_sha256=package_hashes,contract_sha256=core.json_digest(contract),
                      synthetic_fixture_only=True,dataset_downloaded=False)
        ctx = SimpleNamespace(rt=rt,c=c,source=source,source_info=source_info,modules=modules,model=model,
                              optimizer=optimizer,scheduler=scheduler,loaders=loaders,contract=contract,report=report)
        contexts[name] = ctx
        return ctx

    def verify_fixture_unchanged(ctx, log):
        assert all(core.file_sha256(source/name)==digest for name,digest in source_info['workcopy_files_sha256'].items())
        assert s.verify_package()==package_hashes
        assert all(torch.equal(dataset.tensors[0],images) and torch.equal(dataset.tensors[1],labels)
                   for dataset in datasets.values())
        ctx.report.update(source_unchanged=True,package_unchanged=True,synthetic_inputs_unchanged=True)

    def run_case(name, checkpoint=None, stop=None, expected='PASS'):
        cases[name] = {'updates':0}
        args = ['resume' if checkpoint else 'run','--output-root',str(destination/name)]
        if checkpoint: args += ['--checkpoint',str(checkpoint)]
        original_write = s.write_json
        def write_then_stop(path, value):
            result=original_write(path,value)
            if stop and path.parent.name=='epoch_summaries' and path.name==f'epoch_{stop:05d}.json':
                (path.parent.parent/'STOP_AFTER_EPOCH').touch()
            return result
        print('CPU integration case:',name,flush=True)
        with (destination/(name+'_stdout.log')).open('x',encoding='utf-8') as stream:
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(stream))
                stack.enter_context(patch.object(sys,'argv',[str(ROOT/'run_train.py'),*args]))
                stack.enter_context(patch.object(r,'create_context',fixture_context))
                stack.enter_context(patch.object(r,'verify_unchanged',verify_fixture_unchanged))
                stack.enter_context(patch.object(s,'write_json',write_then_stop))
                stack.enter_context(patch.object(torch.Tensor,'cuda',cpu_cuda))
                for bit in (6,4):
                    stack.enter_context(patch.object(a.codec(bit),'encode_tensor',forbidden))
                    stack.enter_context(patch.object(a.codec(bit),'encode_elements',forbidden))
                code = r.main()
        matches=list((destination/name).glob('*_joint_*')); assert len(matches)==1
        path=matches[0]; report=read_json(path/'run_report.json')
        assert report['status']==expected, (name,report.get('error'),report.get('traceback'))
        assert code==(1 if expected=='FAIL' else 0)
        assert (path/'feedback_bundle.zip').is_file() and (path/r.NOTEBOOK).is_file()
        cases[name].update(status=report['status'],output_directory=str(path))
        return path,report

    continuous, report = run_case('continuous5')
    assert report['completed_epoch']==5 and report['pilot_completed'] and not report['full_training_completed']
    assert report['optimizer_steps']==cases['continuous5']['updates']==10 and report['training_forward_passes']==30
    assert report['timing_estimate']['measured_100_epoch_seconds'] is None
    assert report['history_export_validation']['exact_trace_match']
    trace=read_json(continuous/'history.json')
    scale_check=verify_scale_tables(continuous,trace)
    export_check=verify_exports(continuous,report,modules)
    scores=[rec['test'][-1]['top1_percent'] for rec in trace['epochs']]
    assert report['selected_best_epoch']==scores.index(max(scores))+1
    assert report['selected_last_epoch']==5
    with ZipFile(continuous/'feedback_bundle.zip') as archive:
        contents=set(archive.namelist())
        assert {r.NOTEBOOK,'notebook_support.py','history.json','scale_block_state.csv','scale_source_changes.csv'} <= contents
        assert not any(n.endswith('.pt') for n in contents)
        for name,digest in references.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    notebook=read_json(continuous/r.NOTEBOOK)
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in notebook['cells'] if c['cell_type']=='code')
    selected_parameters=[c for c in notebook['cells'] if 'parameters' in c.get('metadata',{}).get('tags',[])]
    assert len(selected_parameters)==1 and str(continuous.resolve()) in ''.join(selected_parameters[0]['source'])
    final=core.load_verified(continuous/'checkpoints/epoch_00005.pt')

    stopped, partial=run_case('stop2',stop=2,expected='STOPPED_AT_EPOCH_BOUNDARY')
    assert partial['completed_epoch']==2 and not partial['pilot_completed'] and cases['stop2']['updates']==4
    checkpoint=stopped/'checkpoints/epoch_00002.pt'; digest=core.file_sha256(checkpoint)
    partial_state=core.load_verified(checkpoint)
    continuous2=core.load_verified(continuous/'checkpoints/epoch_00002.pt')
    for key in ('model','optimizer','scheduler','rng'):
        assert not core.differences(partial_state[key],continuous2[key]),key
    resumed,resumed_report=run_case('resume2_to5',checkpoint=checkpoint)
    resumed_state=core.load_verified(resumed/'checkpoints/epoch_00005.pt')
    assert cases['resume2_to5']['updates']==6 and resumed_report['pilot_completed']
    for key in ('model','optimizer','scheduler','rng'):
        assert not core.differences(final[key],resumed_state[key]),key
    for before,after in zip(final['trace']['epochs'],resumed_state['trace']['epochs']):
        for key in ('train','test','model_sha256','scale_state','source_scale_state','scale_epoch_metrics'):
            assert before[key]==after[key],(before['epoch'],key)
        for phase in ('train','test'):
            assert before['coverage'][phase]['order_sha256']==after['coverage'][phase]['order_sha256']
    for filename in ('scale_block_state.csv','scale_epoch_metrics.csv','scale_source_changes.csv'):
        assert (continuous/filename).read_bytes()==(resumed/filename).read_bytes(),filename
    assert core.file_sha256(checkpoint)==digest

    foreign=copy.deepcopy(partial_state)
    foreign['contract']['config']['schema']='mx-triple-learned-scale10-pilot5-protocol-v1'
    foreign_path=destination/'foreign_fixture.pt'; core.atomic_save(foreign_path,foreign)
    rejected,reject_report=run_case('reject_foreign_contract',checkpoint=foreign_path,expected='FAIL')
    assert 'Hervatprotocol/omgeving/bron verschilt' in reject_report['error']
    assert cases['reject_foreign_contract']['updates']==0 and not list((rejected/'checkpoints').glob('*.pt'))
    assert s.legacy.state_digest(contexts['reject_foreign_contract'].model.state_dict())==cases['reject_foreign_contract']['fresh_model_sha256']
    for args in (['run','--stop-after-epoch','6'],['run','--device','cpu'],['resume']):
        proc=subprocess.run([sys.executable,str(ROOT/'run_train.py'),*args],cwd=destination,capture_output=True,text=True)
        assert proc.returncode==2,(args,proc.stdout,proc.stderr)
    assert s.verify_package()==package_hashes
    assert all(core.file_sha256(ROOT/name)==digest for name,digest in references.items())
    result={'status':'PASS','synthetic_fixture_only':True,'no_SVHN_data_loaded_or_downloaded':True,
            'not_a_SVHN_accuracy_or_GPU_timing_result':True,
            'scope':'actual check entry plus main/loop/export/resume on three CPU synthetic images',
            'test_duration_seconds':time.perf_counter()-started,'cases':cases,
            'scale_tables':scale_check,'exports':export_check,
            'continuous5_equals_stop2_resume5_model_optimizer_scheduler_RNG_sample_orders_exact_CPU':True,
            'partial_run_not_marked_PASS':True,'foreign_contract_rejected_before_restore_or_updates':True,
            'historical_raw_references_unchanged':True,'production_config':config,
            'test_file_sha256':core.file_sha256(Path(__file__)),
            'check_entry_report':read_json(destination/'check-entry/check_entry_report.json')}
    write_json(destination/'runner_integration_report.json',result)
    print(json.dumps({'status':'PASS','report':str(destination/'runner_integration_report.json'),
                      'seconds':result['test_duration_seconds']},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','oracle','check-entry'])
    parser.add_argument('destination',type=Path)
    parser.add_argument('oracle',nargs='?',type=Path)
    args=parser.parse_args()
    if args.command=='oracle': oracle_process(args.destination.resolve())
    elif args.command=='check-entry': check_entry(args.destination.resolve(),args.oracle)
    else: main(args.destination.resolve())
