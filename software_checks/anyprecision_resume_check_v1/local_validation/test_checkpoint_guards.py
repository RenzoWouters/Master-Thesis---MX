"""Small CPU unit tests for checkpoint safety and exact RNG restoration.

Uses a tiny artificial model, not an accuracy experiment. Temporary files only.
"""
from pathlib import Path
import json
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import resume_core as core


class CheckpointGuards(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        np.random.seed(71)
        random.seed(71)
        self.device = torch.device('cpu')
        self.generators = {'train':torch.Generator().manual_seed(91),
                           'validation':torch.Generator().manual_seed(92)}
        self.model = torch.nn.Linear(4,2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [1,2], gamma=.1)
        y=self.model(torch.randn(3,4)).square().mean()
        y.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.contract = {'test':True,'device':'cpu','seed':71}
        self.state = core.capture_checkpoint(self.model, self.optimizer, self.scheduler,
            self.generators, self.device, 1, 1, self.contract, {'epochs':[]}, {})

    def test_rng_roundtrip_with_cached_gaussian(self):
        random.gauss(0,1)
        np.random.normal(size=3)
        state=core.capture_rng(self.generators,self.device)
        expected=core.rng_probe(self.generators,self.device)
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        for g in self.generators.values():g.manual_seed(1)
        core.restore_rng(state,self.generators,self.device)
        self.assertEqual(expected,core.rng_probe(self.generators,self.device))
        self.assertFalse(core.differences(state,core.capture_rng(self.generators,self.device)))

    def test_full_restore_and_next_optimizer_update(self):
        model2=torch.nn.Linear(4,2)
        opt2=torch.optim.Adam(model2.parameters(),lr=.001)
        sch2=torch.optim.lr_scheduler.MultiStepLR(opt2,[1,2],gamma=.1)
        checks=core.restore_checkpoint(self.state,model2,opt2,sch2,self.generators,self.device,self.contract)
        self.assertTrue(all(checks.values()))
        x=torch.arange(12,dtype=torch.float32).reshape(3,4)/10
        for model,opt,sch in [(self.model,self.optimizer,self.scheduler),(model2,opt2,sch2)]:
            opt.zero_grad()
            model(x).square().mean().backward()
            opt.step()
            sch.step()
        self.assertFalse(core.differences(self.model.state_dict(),model2.state_dict()))
        self.assertFalse(core.differences(self.optimizer.state_dict(),opt2.state_dict()))
        self.assertEqual(float(opt2.param_groups[0]['lr']),.00001)

    def test_reject_contract_change_before_loading(self):
        previous=core.cpu_copy(self.model.state_dict())
        with self.assertRaisesRegex(RuntimeError,'Hervatprotocol'):
            core.restore_checkpoint(self.state,self.model,self.optimizer,self.scheduler,
                self.generators,self.device,{'test':True,'device':'cpu','seed':72})
        self.assertFalse(core.differences(previous,self.model.state_dict()))

    def test_no_overwrite_and_weights_only_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'epoch_001.pt'
            core.atomic_save(p,self.state)
            original=p.read_bytes()
            with self.assertRaisesRegex(RuntimeError,'overschreven'):
                core.atomic_save(p,self.state)
            self.assertEqual(original,p.read_bytes())
            self.assertFalse(core.differences(self.state,core.load_verified(p)))

    def test_reject_corrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'epoch_001.pt'
            core.atomic_save(p,self.state)
            # Deliberate corruption of a disposable test checkpoint only.
            with p.open('r+b') as f:
                f.seek(20)
                x=f.read(1)
                f.seek(20)
                f.write(bytes([x[0]^1]))
            with self.assertRaisesRegex(RuntimeError,'checksum'):
                core.load_verified(p)

    def test_reject_incomplete_checkpoint_without_marker(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'epoch_001.pt'
            torch.save(self.state,p)
            with self.assertRaisesRegex(RuntimeError,'checksum ontbreekt'):
                core.load_verified(p)

    def test_missing_rng_is_detected(self):
        invalid=core.cpu_copy(self.state)
        del invalid['rng']['loader_generators']['train']
        with self.assertRaisesRegex(RuntimeError,'generators verschillen'):
            core.restore_rng(invalid['rng'],self.generators,self.device)

    def test_negative_control_omitting_rng_changes_probe(self):
        expected=core.rng_probe(self.generators,self.device)
        torch.manual_seed(655)
        actual=core.rng_probe(self.generators,self.device)
        self.assertNotEqual(expected,actual)


if __name__=='__main__':
    unittest.main(verbosity=2)
