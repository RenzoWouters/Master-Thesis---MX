#!/usr/bin/env python3
"""CPU-only, read-only checkpoint verification and memory-traffic modelling."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import unittest
import zipfile

HERE=Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def package_hashes():
    return {str(p.relative_to(HERE)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and p.suffix not in ('.zip','.png','.svg')
            and 'local_validation' not in p.parts and p.name!='SHA256.json'}


def run_self_tests():
    output=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    details=dict(status='PASS' if result.wasSuccessful() else 'FAIL',
                 tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                 log=output.getvalue())
    if result.testsRun == 0: details['status']='FAIL'
    return details


def write_tables(out, results):
    tables=out/'tables'; tables.mkdir()
    for key in ('storage','initial_loading','mode_payloads','summary','trace','activations'):
        rows=results.get(key,[])
        if not rows: continue
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (tables/(key+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def copy_analysis_files(out):
    for name in ('MX_precisiewissels_geheugentransfers.ipynb','analysis_helpers.py','README.md'):
        p=HERE/name
        if p.is_file(): shutil.copyfile(p,out/name)
    dest=out/'data'; dest.mkdir(exist_ok=True)
    for name in ('accuracy_context.json','source_catalog.json'):
        p=HERE/'data'/name
        if p.is_file(): shutil.copyfile(p,dest/name)


def bundle(out):
    dest=out/'feedback_bundle.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p!=dest and '__pycache__' not in p.parts:
                z.write(p,p.relative_to(out))
        for p in sorted(HERE.glob('*.py')):
            z.write(p,Path('experiment_code')/p.name)
        z.write(HERE/'experiment_config.json','experiment_code/experiment_config.json')
    return dest


def parse_args():
    p=argparse.ArgumentParser(description='Initiële loading en precisiewissels: zes bestaande modellen, geen training.')
    p.add_argument('command',choices=['check','run','reference'])
    p.add_argument('--project-root',type=Path,default=HERE.parents[1])
    p.add_argument('--output-root',type=Path,default=None)
    p.add_argument('--sources',type=Path,default=None,help='Optionele JSON: model-id -> bestaande runmap.')
    p.add_argument('--config',type=Path,default=HERE/'experiment_config.json')
    p.add_argument('--self-test-only',action='store_true',help='Alleen bij check: geen lokale checkpointcontrole.')
    args=p.parse_args()
    if args.self_test_only and args.command!='check': p.error('--self-test-only hoort uitsluitend bij check')
    return args


def main():
    args=parse_args(); start=time.perf_counter()
    root=args.project_root.resolve()
    parent=args.output_root.resolve() if args.output_root else root/'mx_runs'/'mx_precision_switch_transfer_eval'
    # Never overwrite: unique run folder even for failed checks.
    out=parent/(dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.command)
    out.mkdir(parents=True,exist_ok=False)
    lines=[]
    def log(msg):
        msg=str(msg); lines.append(msg); print(msg,flush=True)
    report=dict(schema='mx_precision_switch_transfer_report_v1',command=args.command,status='RUNNING',
                created_at=dt.datetime.now().astimezone().isoformat(),project_root=str(root),
                result_directory=str(out),training_started=False,dataset_accessed=False,
                checkpoint_tensors_deserialized=False,hardware_measured=False,
                python=sys.version,package_hashes=package_hashes())
    exit_code=0
    try:
        log('RESULTATENMAP: '+str(out))
        log('GEHEUGENTRANSFERPROEF: initiële loading + precisiewissels; zes bestaande modellen; CPU; geen training.')
        report['self_tests']=run_self_tests()
        (out/'self_tests.txt').write_text(report['self_tests']['log'],encoding='utf-8')
        if report['self_tests']['status']!='PASS': raise RuntimeError('Zelftest faalt; zie self_tests.txt')
        log(f"ZELFTEST PASS: {report['self_tests']['tests_run']} tests.")
        from source_catalog import verify_sources, reference_source_summary
        if args.command=='reference' or args.self_test_only:
            report['source_validation']=reference_source_summary() if args.command=='reference' else dict(status='NOT_CHECKED',reason='explicit self-test-only')
        else:
            overrides=json.loads(args.sources.read_text(encoding='utf-8-sig')) if args.sources else None
            log('BRONCONTROLE: bestaande epoch-100-checkpoints/exports zoeken en SHA-256 controleren.')
            report['source_validation']=verify_sources(root,overrides=overrides,hash_files=True)
            write_json(out/'source_validation.json',report['source_validation'])
            if report['source_validation'].get('status')!='PASS':
                for message in report['source_validation'].get('errors',[]): log('BRON: '+message)
                raise RuntimeError('Niet alle zes bestaande bronnen zijn bevestigd; zie source_validation.json. Gebruik --sources bij verplaatste runs.')
            log('BRONCONTROLE PASS: zes bestaande volledige trainingen bevestigd.')
        config=json.loads(args.config.read_text(encoding='utf-8-sig'))
        write_json(out/'experiment_config.json',config)
        if args.command!='check':
            from experiment import run_experiment
            results=run_experiment(config,progress=log)
            results['status']='REFERENCE_ONLY' if args.command=='reference' else 'PASS'
            results['source_validation']=report['source_validation']
            results['created_at']=report['created_at']
            write_json(out/'run_results.json',results)
            write_tables(out,results)
            copy_analysis_files(out)
            report['simulation_seconds']=results['simulation_seconds']
            report['simulation_count']=len(results['summary'])
            report['results_file']='run_results.json'
            report['status']=results['status']
            log(f"SIMULATIE KLAAR: {len(results['summary'])} combinaties in {results['simulation_seconds']:.2f} s.")
        else:
            report['status']='SELF_TEST_ONLY' if args.self_test_only else 'PASS'
            log('CHECK KLAAR. Er is geen training of netwerk-inference uitgevoerd.')
    except Exception as exc:
        report['status']='FAIL'; report['error']=str(exc); report['traceback']=traceback.format_exc()
        log('FAIL: '+str(exc)); log(report['traceback']); exit_code=1
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        write_json(out/'run_report.json',report)
        log('EINDSTATUS: '+report['status'])
        log('Rapport: '+str(out/'run_report.json'))
        log('Feedback: '+str(out/'feedback_bundle.zip'))
        (out/'run.log').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        bundle(out)
    return exit_code


if __name__=='__main__': raise SystemExit(main())
