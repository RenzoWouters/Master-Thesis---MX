"""CPU checkpoint tests, including continuation in a fresh Python process.

Synthetic fixtures are reachable only by these explicit self-test functions;
they never replace CIFAR-10 in the training runner.
"""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

import checkpoint_support as checkpoint
from engine import create_optimizer, train_step


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 8)
        self.bn = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        return self.head(self.dropout(torch.tanh(self.bn(self.linear(x.mean((2, 3)))))))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="checkpoint_unittest_")
        self.root = Path(self.temporary.name)
        self.protocol = checkpoint._fixture_protocol()
        self.runtime = {"torch": torch.__version__, "device": {"name": "CPU"}}
        self.identity = {"fixture": True, "train_examples": 4}
        self.train_generator = torch.Generator().manual_seed(31)
        self.val_generator = torch.Generator().manual_seed(32)
        random.seed(21); np.random.seed(22); torch.manual_seed(23)
        self.model = _TinyModel()
        self.optimizer = create_optimizer(self.model)
        for _ in range(2):
            train_step(self.model, self.optimizer, torch.randn(2, 3, 32, 32), torch.tensor([1, 3]))

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, name="epoch001.pt", **overrides):
        kwargs = dict(epoch_completed=1, global_updates=2,
            history=[{"epoch": 1, "global_updates": 2}],
            train_generator=self.train_generator, val_generator=self.val_generator,
            protocol=self.protocol, package_fingerprint="a" * 64,
            runtime=self.runtime, dataset_identity=self.identity, best_epoch=1, best_top1=20.0)
        kwargs.update(overrides)
        return checkpoint.save_checkpoint(self.root / name, self.model, self.optimizer, **kwargs)

    def load(self, name="epoch001.pt", **overrides):
        kwargs = dict(expected_experiment=self.protocol["experiment"], expected_fingerprint="a" * 64)
        kwargs.update(overrides)
        return checkpoint.load_checkpoint(self.root / name, **kwargs)

    def test_roundtrip_weights_only_and_fp32(self):
        saved = self.save()
        payload = self.load()
        self.assertEqual(saved["sha256"], checkpoint._file_hash(saved["path"]))
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        self.assertEqual(payload["epoch_completed"], 1)
        self.assertEqual(payload["global_updates"], 2)
        other = _TinyModel()
        opt = create_optimizer(other)
        report = checkpoint.restore_checkpoint(payload, other, opt,
            torch.Generator(), torch.Generator(), device="cpu",
            expected_runtime=self.runtime, expected_dataset_identity=self.identity)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(checkpoint._compare_nested(self.model.state_dict(), other.state_dict(), atol=0, rtol=0)["exact"])
        self.assertTrue(checkpoint._compare_nested(self.optimizer.state_dict(), opt.state_dict(), atol=0, rtol=0)["exact"])
        self.assertTrue(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all())
            for p in opt.state.values() for t in p.values() if torch.is_tensor(t)))

    def test_all_rng_and_loader_generator_states_restored(self):
        self.save()
        expected = (random.random(), np.random.random(), torch.rand(4),
                    torch.randperm(9, generator=self.train_generator),
                    torch.randperm(9, generator=self.val_generator))
        model = _TinyModel(); opt = create_optimizer(model)
        train, val = torch.Generator(), torch.Generator()
        checkpoint.restore_checkpoint(self.load(), model, opt, train, val, device="cpu")
        actual = (random.random(), np.random.random(), torch.rand(4),
                  torch.randperm(9, generator=train), torch.randperm(9, generator=val))
        self.assertTrue(checkpoint._compare_nested(expected, actual, atol=0, rtol=0)["exact"])

    def test_existing_checkpoint_is_never_overwritten(self):
        self.save(); before = (self.root / "epoch001.pt").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "niet overschreven"):
            self.save()
        self.assertEqual(before, (self.root / "epoch001.pt").read_bytes())

    def test_partial_timing_checkpoint_rejected(self):
        torch.save({"partial_epoch": True, "resume_supported": False}, self.root / "timing.pt")
        with self.assertRaisesRegex(RuntimeError, "timingcheckpoint"):
            self.load("timing.pt")

    def test_wrong_experiment_and_fingerprint_rejected(self):
        self.save()
        with self.assertRaisesRegex(RuntimeError, "ander experiment"):
            self.load(expected_experiment="different")
        with self.assertRaisesRegex(RuntimeError, "andere pakketversie"):
            self.load(expected_fingerprint="b" * 64)

    def test_history_and_adam_update_inconsistency_rejected_before_write(self):
        for extra in ({"history": [{"epoch": 2}]}, {"global_updates": 3},
                      {"epoch_completed": 2, "global_updates": 4,
                       "history": [{"epoch": 1}, {"epoch": 2}]}):
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                self.save(**extra)
            self.assertFalse((self.root / "epoch001.pt").exists())

    def test_bad_dataset_runtime_and_tensor_rejected_before_model_mutation(self):
        self.save()
        for cause in ("dataset", "runtime", "tensor", "lr"):
            payload = self.load()
            target = _TinyModel(); optimizer = create_optimizer(target)
            before = checkpoint._cpu_clone(target.state_dict())
            kwargs = dict(device="cpu", expected_runtime=self.runtime,
                          expected_dataset_identity=self.identity)
            if cause == "dataset": kwargs["expected_dataset_identity"] = {"fixture": "changed"}
            if cause == "runtime": kwargs["expected_runtime"] = {"torch": "changed"}
            if cause == "tensor": payload["model_state"]["linear.weight"][0, 0] = float("nan")
            if cause == "lr": payload["optimizer_state"]["param_groups"][0]["lr"] = 0.1
            with self.subTest(cause=cause), self.assertRaises(RuntimeError):
                checkpoint.restore_checkpoint(payload, target, optimizer,
                                              torch.Generator(), torch.Generator(), **kwargs)
            self.assertTrue(checkpoint._compare_nested(before, target.state_dict(), atol=0, rtol=0)["exact"])

    def test_completed_five_epoch_checkpoint_cannot_continue(self):
        self.save(); payload = self.load()
        payload.update(epoch_completed=5, global_updates=10,
            history=[{"epoch": e, "global_updates": 2 * e} for e in range(1, 6)],
            lr_schedule_state=checkpoint._schedule(self.protocol, 5))
        with self.assertRaisesRegex(RuntimeError, "al voltooid"):
            checkpoint.restore_checkpoint(payload, self.model, self.optimizer,
                self.train_generator, self.val_generator, device="cpu")

    def test_parent400_schedule_boundaries_are_not_compressed_for_pilot(self):
        rates = {e: checkpoint.learning_rate_for_completed_epochs(e)
                 for e in (0, 4, 5, 149, 150, 249, 250, 349, 350, 400)}
        for e in (0, 4, 5, 149): self.assertEqual(rates[e], .001)
        for e in (150, 249): self.assertAlmostEqual(rates[e], .0001)
        for e in (250, 349): self.assertAlmostEqual(rates[e], .00001)
        for e in (350, 400): self.assertAlmostEqual(rates[e], .000001)
        checkpoint.apply_learning_rate(self.optimizer, 150)
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], .0001)

    def test_actual_mobilenet_continuation_exact_on_cpu(self):
        result = checkpoint.run_resume_diagnostic("cpu")
        self.assertEqual(result["status"], "PASS", result)
        self.assertTrue(result["rng_exact"] and result["model_exact"] and result["optimizer_exact"])

    def test_actual_mobilenet_continuation_in_fresh_process(self):
        from mobilenet_model import build_model
        model = build_model(); optimizer = create_optimizer(model)
        train, val = torch.Generator().manual_seed(61), torch.Generator().manual_seed(62)
        checkpoint._fixture_epoch(model, optimizer, train, val, "cpu")
        path, output = self.root / "process.pt", self.root / "continued.pt"
        checkpoint.save_checkpoint(path, model, optimizer, epoch_completed=1, global_updates=2,
            history=[{"epoch": 1}], train_generator=train, val_generator=val,
            protocol=self.protocol, package_fingerprint="a" * 64, runtime=self.runtime,
            dataset_identity=self.identity, best_epoch=1, best_top1=0.)
        checkpoint._fixture_epoch(model, optimizer, train, val, "cpu")
        expected = {"model": checkpoint._cpu_clone(model.state_dict()),
                    "optimizer": checkpoint._cpu_clone(optimizer.state_dict()),
                    "rng": checkpoint.capture_rng(train, val, device="cpu")}
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                 "_continue", str(path), str(output)],
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = torch.load(output, map_location="cpu", weights_only=True)
        compared = checkpoint._compare_nested(expected, actual, atol=0, rtol=0)
        self.assertTrue(compared["exact"], compared)


def _continue_in_process(path, output):
    from mobilenet_model import build_model
    torch.set_num_threads(4)
    protocol = checkpoint._fixture_protocol()
    model = build_model(); optimizer = create_optimizer(model)
    train, val = torch.Generator(), torch.Generator()
    payload = checkpoint.load_checkpoint(path, expected_experiment=protocol["experiment"],
                                         expected_fingerprint="a" * 64)
    checkpoint.restore_checkpoint(payload, model, optimizer, train, val, device="cpu")
    checkpoint._fixture_epoch(model, optimizer, train, val, "cpu")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": checkpoint.capture_rng(train, val, device="cpu")}, output)


def run_tests(progress=None):
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CheckpointTests)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report = {"status": "PASS" if result.wasSuccessful() else "FAIL",
                  "tests_run": result.testsRun, "output": stream.getvalue(),
                  "scope": "CPU only; real MobileNet synthetic continuation including a fresh process; no CIFAR training or CUDA claim"}
        if not result.wasSuccessful():
            raise AssertionError(report["output"])
        if progress:
            progress(f"CPU checkpointtests PASS: {result.testsRun}; inclusief hervatten in nieuw proces.")
        return report
    finally:
        torch.set_num_threads(previous)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "_continue":
        _continue_in_process(sys.argv[2], sys.argv[3])
    else:
        print(json.dumps(run_tests(), indent=2))
