"""CPU-only component/integration tests; all inputs here are synthetic.

No SVHN files, trained checkpoints, or network-accuracy measurements are used.
The compact fixture is constructed locally with synthetic_fixture_only=True;
the production source verifier MUST reject it. Direct component tests may use
it to test frozen reconstruction and inference without relaxing that guard.

Run with the existing PyTorch 2.10.0 / torchvision 0.25.0 environment:
    python -m unittest discover -s tests -p test_probe.py -v
"""
from __future__ import annotations

import copy
import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import run_probe as probe
import mantissa_rules as rules


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def independent_value(code, bits):
    # Explicit codebooks, independent of both the Fraction formula and the
    # legacy tensor decoders used by the evaluator.
    source = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
              16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60)
    target = (0,4,8,12,16,24,32,48)
    levels, sign_bit = (source,32) if bits == 6 else (target,8)
    result = Fraction(levels[code % sign_bit],8)
    return -result if code & sign_bit else result


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = probe.import_runtime('cpu')
        cls.torch, cls.a = cls.rt.torch, cls.rt.a
        cls.torch.set_num_threads(1)
        cls.torch.manual_seed(1729)
        reference = cls.rt.models.svhnq([1,2,4,8,32], expand=2)
        trainable = cls.a.adapt_copy(reference).cpu()
        for parameter in cls.a.scale_parameters(trainable):
            parameter.requires_grad_(False)
        provenance = {
            'source_epoch': 100,
            'checkpoint_sha256': 'a' * 64,
            'synthetic_fixture_only': True,
            'absolute_E2_scale_frozen_at_this_checkpoint': True,
            'fixture_note': 'Untrained synthetic source; epoch is guard-test metadata, not a training result.',
        }
        cls.payload, _ = cls.a.source_payload(trainable, provenance)
        generator = cls.torch.Generator().manual_seed(731)
        for name in cls.a.TARGET_LAYERS:
            old = cls.a.codec(8).MXTensor.from_payload(cls.payload['mx_layers'][name])
            rows, columns = old.shape[0], math.prod(old.shape[1:])
            logical = cls.torch.randint(0,64,(rows,columns),generator=generator,dtype=cls.torch.int64)
            logical.reshape(-1)[:64] = cls.torch.arange(64)
            middle_codes = cls.torch.zeros_like(old.codes, dtype=cls.torch.int64)
            middle_codes.reshape(rows,-1)[:, :columns] = logical
            # Inverse of the existing exponent-MSB extraction: E4 low exponent
            # bits are zero, while sign and all three mantissa bits are kept.
            source_codes = (((middle_codes & 56) << 2) | (middle_codes & 7)).to(cls.torch.uint8)
            rebuilt = cls.a.codec(8).MXTensor(source_codes,old.scale_codes.clone(),old.shape)
            cls.payload['mx_layers'][name] = rebuilt.payload()
            cls.payload['lower_scale_codes'][name] = cls.torch.full_like(old.scale_codes,119)
        cls.a.validate_source(cls.payload)
        cls.models, cls.chains = probe.build_variants(cls.payload, cls.rt)
        cls.inputs = cls.torch.linspace(-1.,1.,steps=3*3*40*40).reshape(3,3,40,40)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mx_probe_synthetic_test_')
        self.tmp = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save_source(self, payload=None, name='synthetic_source.pt'):
        payload = self.payload if payload is None else payload
        path = self.tmp / name
        self.torch.save(payload,path)
        path.with_suffix('.pt.sha256').write_text(probe.sha256(path)+'\n',encoding='ascii')
        reference = {
            'schema': 'synthetic-source-guard-test-only',
            'source_bytes': path.stat().st_size,
            'source_sha256': probe.sha256(path),
            'checkpoint_sha256': 'a'*64,
        }
        return path,reference

    def test_tensor_rules_and_legacy_decoders_match_all64_exact_codes(self):
        audit = probe.audit_tensor_rules(self.rt)
        self.assertTrue(audit['all64_codes_match_exact_reference'])
        self.assertTrue(audit['all_methods_match_existing_decoders'])
        codes = self.torch.arange(64,dtype=self.torch.uint8).reshape(2,32)
        for method in rules.METHODS:
            result = probe.recode_tensor(codes,method,self.torch)
            self.assertEqual(result.dtype,self.torch.uint8)
            self.assertEqual(result.shape,codes.shape)
            self.assertEqual(result.reshape(-1).tolist(),list(rules.lookup(method)))
            for source_code,target_code in zip(codes.reshape(-1).tolist(),result.reshape(-1).tolist()):
                self.assertEqual(rules.decode_code(source_code,6),independent_value(source_code,6))
                decoded = self.a.codec(4).decode_elements(self.torch.tensor([target_code],dtype=self.torch.uint8))[0]
                self.assertEqual(decoded.item(),float(independent_value(target_code,4)))
        for bad in (self.torch.tensor([64],dtype=self.torch.uint8),
                    self.torch.tensor([0],dtype=self.torch.int64)):
            with self.assertRaises(RuntimeError):
                probe.recode_tensor(bad,'lsb_trunc',self.torch)

    def test_baseline_is_the_existing_frozen_loader_and_only_mx_weights_change(self):
        baseline = self.a.load_exported_model(self.rt.models,self.payload,self.rt.device,bit=4)
        self.assertEqual(probe.state_digest(baseline),probe.state_digest(self.models['lsb_trunc']))
        baseline_state = baseline.state_dict()
        decoded_keys = {name+'.decoded_weight' for name in self.a.TARGET_LAYERS}
        for method in rules.METHODS:
            model = self.models[method]
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            state = model.state_dict()
            self.assertEqual(set(state),set(baseline_state))
            changed = {key for key in state if not self.torch.equal(state[key],baseline_state[key])}
            self.assertTrue(changed <= decoded_keys)
            if method != 'lsb_trunc':
                self.assertTrue(changed,'The synthetic fixture must distinguish the alternatives.')
            for name in self.a.BN_LAYERS:
                self.assertEqual(model.get_submodule(name).weight_mode,4)
        for name,chain in self.chains.items():
            self.assertTrue(self.torch.equal(chain[6].scale_codes,self.payload['lower_scale_codes'][name]))
            self.assertTrue(self.torch.equal(chain[4].scale_codes,chain[6].scale_codes))
        for bit,key in ((8,'source8'),(6,'source6')):
            loaded = self.a.load_exported_model(self.rt.models,self.payload,self.rt.device,bit=bit)
            self.assertEqual(probe.state_digest(loaded),probe.state_digest(self.models[key]))

    def test_three_real_cnn_forwards_per_variant_are_finite_and_read_only(self):
        before = {name:probe.state_digest(model) for name,model in self.models.items()}
        with self.torch.inference_mode():
            for name,model in self.models.items():
                logits = model(self.inputs)
                self.assertEqual(tuple(logits.shape),(3,10),name)
                self.assertTrue(bool(self.torch.isfinite(logits).all()),name)
        self.assertEqual(before,{name:probe.state_digest(model) for name,model in self.models.items()})
        report = probe.check_source_models(self.payload,self.rt)
        self.assertEqual(report,{'synthetic_images':3,'variants':5,'all_outputs_finite':True,
                                 'model_and_bn_state_unchanged':True,'accuracy_measured':False})

    def test_weight_diagnostics_exclude_padding_and_use_existing_shared_scales(self):
        # Six logical weights, but 64 storage slots: an erroneous inclusion of
        # padding strongly changes counts, MAE/MSE, and the zero histogram.
        codes = self.torch.zeros((2,1,32),dtype=self.torch.uint8)
        codes[0,0,:3] = self.torch.tensor([1,6,30],dtype=self.torch.uint8)
        codes[1,0,:3] = self.torch.tensor([33,38,62],dtype=self.torch.uint8)
        scales = self.torch.tensor([[127],[128]],dtype=self.torch.uint8)
        middle = self.a.codec(6).MXTensor(codes,scales,(2,3))
        rows,frequencies = probe.weight_diagnostics({'synthetic_padding_layer':{6:middle}},self.rt)
        self.assertEqual(sum(row['count'] for row in frequencies),6)
        self.assertEqual(next(row['count'] for row in frequencies if row['source_code']==0),0)
        logical = (1,6,30,33,38,62)
        factors = (1,1,1,2,2,2)
        for method in rules.METHODS:
            source = [independent_value(code,6)*factor for code,factor in zip(logical,factors)]
            target = [independent_value(rules.map_code(code,method),4)*factor
                      for code,factor in zip(logical,factors)]
            errors = [abs(t-s) for s,t in zip(source,target)]
            for layer in ('synthetic_padding_layer','ALL_MX_LAYERS'):
                row = next(row for row in rows if row['layer']==layer and row['method']==method)
                self.assertEqual(row['n_weights'],6)
                self.assertEqual(row['zero_source'],0)
                self.assertEqual(row['mae_vs_e2m3'],float(sum(errors,Fraction(0))/6))
                self.assertEqual(row['mse_vs_e2m3'],float(sum((e*e for e in errors),Fraction(0))/6))
                self.assertEqual(row['max_abs_error'],float(max(errors)))
                self.assertEqual(row['exact_values'],sum(e==0 for e in errors))
                self.assertEqual(row['zero_target'],sum(value==0 for value in target))
                self.assertTrue(row['padding_excluded'])
                self.assertTrue(row['shared_scale_unchanged'])

    def test_evaluator_real_cnn_counts_match_independent_forwards(self):
        labels = self.torch.tensor([0,1,2],dtype=self.torch.int64)
        dataset = self.torch.utils.data.TensorDataset(self.inputs,labels)
        expected = {}
        with self.torch.inference_mode():
            for name,model in self.models.items():
                logits = model(self.inputs)
                top5 = logits.topk(5,dim=1).indices
                expected[name] = {'predictions':top5[:,0].tolist(),
                                  'correct1':int((top5[:,0]==labels).sum()),
                                  'correct5':int((top5==labels[:,None]).any(dim=1).sum())}
        rows,audit = probe.evaluate_dataset(self.models,dataset,self.rt,self.tmp,lambda _:None)
        self.assertEqual(audit['examples'],3)
        self.assertTrue(audit['model_and_bn_state_unchanged'])
        self.assertFalse(audit['real_training_performed'])
        for row in rows:
            self.assertEqual(row['n'],3)
            for key in ('correct1','correct5'):
                self.assertEqual(row[key],expected[row['variant']][key])
        saved = read_csv(self.tmp/'predictions.csv')
        for index,row in enumerate(saved):
            self.assertEqual(int(row['index']),index)
            self.assertEqual(int(row['label']),int(labels[index]))
            for name in probe.VARIANTS:
                self.assertEqual(int(row[name]),expected[name]['predictions'][index])

    def test_paired_counts_and_batch_tail_use_actual_torch_forwards(self):
        # Actual nn.Module forwards with controlled, unique logits exercise all
        # four paired outcomes, including disagreement when both are wrong.
        # 129 inputs also exercise the batch128 boundary and a one-image tail.
        torch = self.torch
        count = 129
        patterns = {'source8':(2,2,2,2),'source6':(0,0,0,0),
                    'lsb_trunc':(0,0,1,1),'rne':(0,1,0,2),'msb_bestexp':(1,0,0,1)}
        predictions = {name:[pattern[i%4] for i in range(count)] for name,pattern in patterns.items()}
        class BufferedScores(torch.nn.Module):
            def __init__(self,chosen):
                super().__init__()
                scores = torch.zeros((count,10),dtype=torch.float32)
                scores[torch.arange(count),torch.tensor(chosen)] = 1.
                self.register_buffer('scores',scores)
            def forward(self,inputs):
                return self.scores[inputs[:,0,0,0].long()]
        models = {name:BufferedScores(predictions[name]).eval() for name in probe.VARIANTS}
        images = torch.zeros((count,3,40,40),dtype=torch.float32)
        images[:,0,0,0] = torch.arange(count,dtype=torch.float32)
        labels = torch.zeros(count,dtype=torch.int64)
        dataset = torch.utils.data.TensorDataset(images,labels)
        rows,audit = probe.evaluate_dataset(models,dataset,self.rt,self.tmp,lambda _:None)
        self.assertEqual(audit['examples'],count)
        self.assertTrue(audit['model_and_bn_state_unchanged'])
        for row in rows:
            self.assertEqual(row['n'],count)
            self.assertEqual(row['correct1'],sum(p==0 for p in predictions[row['variant']]))
        saved = read_csv(self.tmp/'predictions.csv')
        self.assertEqual([int(row['index']) for row in saved],list(range(count)))
        paired = {row['variant']:row for row in read_csv(self.tmp/'paired_predictions.csv')}
        for name in ('rne','msb_bestexp'):
            expected = dict(both_correct=0,baseline_correct_only=0,variant_correct_only=0,both_wrong=0,prediction_disagreements=0)
            for baseline,variant in zip(predictions['lsb_trunc'],predictions[name]):
                correct_base,correct_variant = baseline==0,variant==0
                if correct_base and correct_variant:key='both_correct'
                elif correct_base:key='baseline_correct_only'
                elif correct_variant:key='variant_correct_only'
                else:key='both_wrong'
                expected[key]+=1
                expected['prediction_disagreements']+=baseline!=variant
            self.assertEqual(int(paired[name]['n']),count)
            for key,value in expected.items():
                self.assertEqual(int(paired[name][key]),value,(name,key))
            self.assertTrue(all(expected[key]>0 for key in ('both_correct','baseline_correct_only','variant_correct_only','both_wrong')))

    def test_source_guards_reject_synthetic_modified_provenance_and_wrong_checkpoint(self):
        path,reference = self.save_source()
        self.a.load_verified_payload(path)  # Structural payload is valid.
        with self.assertRaisesRegex(RuntimeError,'Geen echte voltooide'):
            probe.verify_source(path,reference,self.rt)
        wrong = dict(reference,checkpoint_sha256='b'*64)
        with self.assertRaisesRegex(RuntimeError,'bedoelde epoch100-checkpoint'):
            probe.verify_source(path,wrong,self.rt)
        payload = copy.deepcopy(self.payload)
        payload['provenance']['source_epoch'] = 99
        wrong_epoch,epoch_ref = self.save_source(payload,'synthetic_epoch99.pt')
        with self.assertRaisesRegex(RuntimeError,'bedoelde epoch100-checkpoint'):
            probe.verify_source(wrong_epoch,epoch_ref,self.rt)
        payload = copy.deepcopy(self.payload)
        payload['provenance']['hidden_tensor'] = self.torch.zeros(1)
        bad_metadata,metadata_ref = self.save_source(payload,'synthetic_bad_metadata.pt')
        with self.assertRaisesRegex(ValueError,'provenance'):
            probe.verify_source(bad_metadata,metadata_ref,self.rt)
        raw = bytearray(path.read_bytes())
        raw[len(raw)//2] ^= 1
        modified = self.tmp/'modified_source.pt'
        modified.write_bytes(raw)
        modified.with_suffix('.pt.sha256').write_text(probe.sha256(modified)+'\n',encoding='ascii')
        with self.assertRaisesRegex(RuntimeError,'SHA-256 verschilt'):
            probe.verify_source(modified,reference,self.rt)
        path.with_suffix('.pt.sha256').write_text('0'*64+'\n',encoding='ascii')
        with self.assertRaisesRegex(ValueError,'checksum'):
            probe.verify_source(path,reference,self.rt)

    def test_dataset_guards_reject_wrong_bytes_before_loading(self):
        data_root = self.tmp/'synthetic_bad_dataset'
        data_root.mkdir()
        path = data_root/'test_32x32.mat'
        path.write_bytes(b'synthetic bytes, deliberately not an SVHN dataset')
        reference = {'dataset':{'files':{'files':{'test_32x32.mat':{
            'bytes':path.stat().st_size,'sha256':'0'*64}}}}}
        with patch.object(self.rt.data,'get_dataset',side_effect=AssertionError('Dataset loader must not be reached.')) as loader:
            with self.assertRaisesRegex(RuntimeError,'SVHN-testbestand wijkt af'):
                probe.load_test_data(data_root,self.tmp/'source.pt',reference,self.rt)
            loader.assert_not_called()

    def test_dataset_guard_rejects_wrong_label_order(self):
        data_root = self.tmp/'synthetic_labels'
        data_root.mkdir()
        path = data_root/'test_32x32.mat'
        path.write_bytes(b'synthetic guard-test placeholder; parser is not called')
        transform = self.rt.data.get_transform('svhn','val')
        fake_dataset = type('SyntheticDataset',(),{
            'split':'test','labels':self.rt.np.array([0,1,2]),'__len__':lambda self:3})()
        reference = {'dataset':{
            'files':{'files':{'test_32x32.mat':{'bytes':path.stat().st_size,'sha256':probe.sha256(path)}}},
            'transforms':{'test':repr(transform)},
            'selection':{'test':{'n':3,'label_sha256':'0'*64,'labels_histogram':[1,1,1,0,0,0,0,0,0,0]}}}}
        old_path = self.rt.data.data_paths['svhn']
        try:
            with patch.object(self.rt.data,'get_dataset',return_value=fake_dataset) as loader:
                with self.assertRaisesRegex(RuntimeError,'Testlabels of hun volgorde verschillen'):
                    probe.load_test_data(data_root,self.tmp/'source.pt',reference,self.rt)
                self.assertFalse(loader.call_args.kwargs['download'])
        finally:
            self.rt.data.data_paths['svhn'] = old_path

    def isolated_package(self,reference,with_vendor=False):
        package = self.tmp/('isolated_full_package' if with_vendor else 'isolated_numeric_package')
        package.mkdir()
        for name in ('run_probe.py','mantissa_rules.py'):
            shutil.copy2(ROOT/name,package/name)
        if with_vendor:
            shutil.copytree(ROOT/'vendor',package/'vendor',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (package/'REFERENCE.json').write_text(json.dumps(reference),encoding='utf-8')
        manifest = {p.relative_to(package).as_posix():probe.sha256(p)
                    for p in sorted(package.rglob('*')) if p.is_file()}
        (package/'PACKAGE_SHA256.json').write_text(json.dumps(manifest),encoding='utf-8')
        return package

    def test_cli_numeric_needs_no_tensor_runtime_and_writes_no_accuracy(self):
        package = self.isolated_package({'schema':'synthetic-cli-numeric-test'})
        output = self.tmp/'numeric_output'
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH',None)
        result = subprocess.run([sys.executable,'-S',str(package/'run_probe.py'),'numeric','--output',str(output)],
                                cwd=package,env=env,text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report = json.loads((output/'run_report.json').read_text())
        self.assertIn(report['status'],('PASS','PASS_WITH_WARNINGS'))
        self.assertFalse(report['network_evaluated'])
        self.assertFalse(report['training_performed'])
        self.assertEqual(len(read_csv(output/'codebook_mapping.csv')),192)
        self.assertFalse((output/'accuracy.csv').exists())
        self.assertFalse((output/'predictions.csv').exists())
        with zipfile.ZipFile(output/'feedback_bundle.zip') as archive:
            self.assertIn('numeric_summary.json',archive.namelist())
            self.assertNotIn('accuracy.csv',archive.namelist())

    def test_cli_check_rejects_synthetic_source_without_fake_accuracy(self):
        source,reference = self.save_source()
        package = self.isolated_package(reference,with_vendor=True)
        output = self.tmp/'rejected_check'
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable,str(package/'run_probe.py'),'check','--device','cpu',
                                 '--source-export',str(source),'--output',str(output)],
                                cwd=package,env=env,text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        report = json.loads((output/'run_report.json').read_text())
        self.assertEqual(report['status'],'FAIL')
        self.assertIn('Geen echte voltooide freeze50-run',report['error'])
        self.assertFalse(report['network_evaluated'])
        self.assertFalse(report['training_performed'])
        self.assertNotIn('accuracy',report)
        self.assertFalse((output/'accuracy.csv').exists())
        self.assertFalse((output/'predictions.csv').exists())
        with zipfile.ZipFile(output/'feedback_bundle.zip') as archive:
            self.assertIn('run_report.json',archive.namelist())
            self.assertNotIn('accuracy.csv',archive.namelist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
