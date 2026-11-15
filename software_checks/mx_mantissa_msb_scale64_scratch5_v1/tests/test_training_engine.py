"""CPU-only scratch training checks; all images are synthetic, no parent file.

Run with the existing Torch environment:
    python tests/test_training_engine.py --output /new/directory/outside/package

The production historical initialization pin remains unchanged. This private
CPU test derives its separate pin from the unchanged original CPU constructor;
the observed CPU/Windows-CUDA hash difference is explicitly reported. Historical
initial RNG is unavailable; RNG equality below compares fresh LSB/MSB construction
and exact CPU resume, not an invented historical snapshot.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import training_engine as engine

torch, core = engine.torch, engine.core
support, legacy, training = engine.support, engine.legacy_adapter, engine.legacy_training


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        generator = torch.Generator().manual_seed(446 if split == "train" else 887)
        self.data = torch.randn((3, 3, 40, 40), generator=generator)
        self.labels = torch.tensor([0, 1, 2], dtype=torch.int64)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index])


def make_loaders():
    return {name: support.TrackedLoader(torch.utils.data.DataLoader(
        support.IndexedDataset(SyntheticDataset(name)), batch_size=2,
        shuffle=name == "train", num_workers=0, drop_last=False)) for name in ("train", "test")}


def prepare_fixture(destination):
    """Create one original source workcopy and import its modules once/process."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    reference = json.loads((ROOT / "reference_lsb_scale64" / "REFERENCE.json").read_text())
    report = json.loads((ROOT / "reference_lsb_scale64" / "run_report.json").read_text())
    config = copy.deepcopy(report["config"])
    config.update(train_examples=3, test_examples=3, batch_size=2)
    source_out = destination / "source"
    source_out.mkdir()
    source, source_info = support.legacy.prepare_source(source_out)
    rt = support.runtime(config["seed"], "cpu", strict=False)
    modules = support.import_source(source)
    assert reference["initial_model_sha256"] == report["initial_model_sha256"]
    # This is an original LSB CPU construction, not a revised production pin.
    # The main research runner still requires the historical reference hash.
    original_cpu = training.build_model(modules, "joint", rt.device)
    cpu_initial_hash = support.legacy.state_digest(original_cpu.state_dict())
    return SimpleNamespace(rt=rt, source=source, modules=modules, source_info=source_info,
        config=config, expected_initial_hash=cpu_initial_hash,
        historical_initial_hash=reference["initial_model_sha256"], synthetic_fixture_only=True)


class ScratchTrainingTests(unittest.TestCase):
    output_path = None
    evidence = {}

    @classmethod
    def setUpClass(cls):
        if cls.output_path is None:
            raise unittest.SkipTest("Voer dit bestand uit met --output voor de CPU-integratie.")
        cls.fixture = prepare_fixture(cls.output_path)
        cls.cuda_patch = patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
        cls.cuda_patch.start()
        cls.reference_hashes = {p.relative_to(ROOT).as_posix(): core.file_sha256(p)
            for directory in (ROOT / "vendor", ROOT / "reference_lsb_scale64")
            for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts}

    @classmethod
    def tearDownClass(cls):
        cls.cuda_patch.stop()
        assert all(core.file_sha256(ROOT / name) == digest for name, digest in cls.reference_hashes.items())
        assert all(core.file_sha256(cls.fixture.source / name) == digest
                   for name, digest in cls.fixture.source_info["workcopy_files_sha256"].items())

    def setUp(self):
        self.out = self.output_path / self._testMethodName
        self.out.mkdir()

    def fresh(self):
        # Caller owns seeding, exactly as in the original runner.
        rt = support.runtime(self.fixture.config["seed"], "cpu", strict=False)
        state = engine.build_state(rt, self.fixture.modules, self.fixture.config,
                                   self.fixture.expected_initial_hash)
        ctx = SimpleNamespace(rt=rt, source=self.fixture.source, modules=self.fixture.modules,
            report={"command": "run", "synthetic_fixture_only": True, "training_performed": False})
        return ctx, state

    def contract(self):
        return {"schema": "synthetic-msb64-scratch-test-v1", "method": "msb_bestexp",
            "from_scratch": True, "initial_model_sha256": self.fixture.expected_initial_hash,
            "scale_factor": 64, "lr": 0.001, "weight_decay": 0.0,
            "scheduler_milestones": [50, 75, 90], "synthetic_fixture_only": True}

    def run_epochs(self, directory, stop=5, resume=None):
        directory.mkdir()
        loaders = make_loaders()
        ctx, state = self.fresh()
        initial_rng = core.capture_rng({}, ctx.rt.device)
        trace = {"method": "msb_bestexp", "synthetic_fixture_only": True, "epochs": []}
        done = steps = 0
        if resume is not None:
            saved = core.load_verified(resume)
            restored = core.restore_checkpoint(saved, state.model, state.optimizer, state.scheduler,
                                               {}, ctx.rt.device, self.contract())
            self.assertTrue(all(restored.values()))
            done, steps, trace = saved["completed_epoch"], saved["optimizer_steps"], copy.deepcopy(saved["trace"])
            engine.verify_state(state, steps)
            ctx.report["command"] = "resume"
        paths = {}
        for epoch in range(done + 1, stop + 1):
            epoch_dir = directory / ("epoch_%02d" % epoch)
            epoch_dir.mkdir()
            results = {}
            for name, train in (("train", True), ("test", False)):
                folder = epoch_dir / name
                folder.mkdir()
                result = engine.run_phase(ctx, state, loaders[name], epoch, train, folder, lambda _: None)
                coverage = loaders[name].validate(folder / "sample_order.npy")
                result["coverage"] = {key: coverage[key] for key in ("n", "order_sha256", "complete_once")}
                self.assertEqual(result["audit"]["optimizer_steps_in_phase"], 2 if train else 0)
                self.assertEqual(result["audit"]["detached_teacher_checks"], 4 if train else 0)
                self.assertEqual(result["audit"]["mapping_checks"], 12)
                self.assertEqual(result["audit"]["actual_quantizer_output_checks"], 36)
                self.assertTrue(result["audit"]["dynamic_source_plus6_verified"])
                self.assertTrue(result["audit"]["no_phi_or_scale_parameters"])
                self.assertEqual(result["audit"]["forward_calls"], {"4": 2, "6": 2, "8": 2})
                self.assertEqual([row["weight_bits"] for row in result["rows"]], [4, 6, 8])
                self.assertTrue(all(row["n"] == 3 and row["lr"] == 0.001 for row in result["rows"]))
                if train:
                    self.assertEqual(result["predictions"], [])
                else:
                    self.assertTrue(result["audit"]["evaluation_model_and_BN_unchanged"])
                    self.assertEqual([row["index"] for row in result["predictions"]], [0, 1, 2])
                    for row in result["rows"]:
                        correct = sum(p["w" + str(row["weight_bits"])] == p["label"] for p in result["predictions"])
                        self.assertEqual(row["correct1"], correct)
                results[name] = result
            steps += 2
            state.scheduler.step()
            checks = engine.verify_state(state, steps)
            self.assertEqual(state.scheduler.last_epoch, epoch)
            self.assertEqual(state.scheduler.get_last_lr(), [0.001])
            trace["epochs"].append({"epoch": epoch, "optimizer_steps": steps,
                "train": results["train"]["rows"], "test": results["test"]["rows"],
                "coverage": {name: result["coverage"] for name, result in results.items()},
                "BN_counts": checks["BN_counts"], "model_sha256": support.legacy.state_digest(state.model.state_dict()),
                "predictions": results["test"]["predictions"]})
            path = directory / ("epoch_%05d.pt" % epoch)
            payload = core.capture_checkpoint(state.model, state.optimizer, state.scheduler,
                {}, ctx.rt.device, epoch, steps, self.contract(), trace, {})
            core.atomic_save(path, payload)
            paths[epoch] = path
        self.assertEqual(ctx.report["actual_new_optimizer_steps"], 2 * (stop - done))
        self.assertTrue(ctx.report["training_performed"])
        return SimpleNamespace(ctx=ctx, state=state, trace=trace, checkpoint=paths[stop], initial_rng=initial_rng)

    def test_fresh_CPU_hash_RNG_and_high_middle_match_original_LSB(self):
        rt = support.runtime(self.fixture.config["seed"], "cpu", strict=False)
        original = training.build_model(self.fixture.modules, "joint", rt.device)
        original_rng = core.capture_rng({}, rt.device)
        original_digest = support.legacy.state_digest(original.state_dict())
        ctx, state = self.fresh()
        self.assertEqual(original_digest, self.fixture.expected_initial_hash)
        self.assertEqual(state.initial_model_sha256, original_digest)
        self.assertEqual(core.differences(original.state_dict(), state.model.state_dict()), [])
        self.assertEqual(core.differences(state.initial_rng, original_rng), [])
        self.assertEqual(sum(p.numel() for p in state.model.parameters()), 132954)
        self.assertFalse(any("phi" in key for key in state.model.state_dict()))
        self.assertFalse(state.optimizer.state)
        self.assertTrue(state.installation_audit["installation_preserves_rng"])
        self.assertTrue(state.installation_audit["optimizer_scheduler_preserve_rng"])
        inputs = torch.linspace(-1, 1, 3 * 3 * 40 * 40).reshape(3, 3, 40, 40)
        observed = []
        for model in (original, state.model):
            model.eval()
            values = {}
            with torch.no_grad(), legacy.shared_source_triplets(model):
                for bit in (4, 6, 8):
                    legacy.select_mode(model, bit)
                    values[bit] = model(inputs).detach().clone()
            observed.append(values)
        self.assertTrue(torch.equal(observed[0][8], observed[1][8]))
        self.assertTrue(torch.equal(observed[0][6], observed[1][6]))
        self.assertFalse(torch.equal(observed[0][4], observed[1][4]))
        self.assertEqual(support.legacy.state_digest(state.model.state_dict()), original_digest)
        self.evidence.update(cpu_initial_model_hash_matches_original_CPU_control=original_digest,
            historical_expected_initial_model_sha256=self.fixture.historical_initial_hash,
            CPU_initial_hash_equals_historical=(original_digest == self.fixture.historical_initial_hash),
            production_historical_initialization_pin_unchanged=True,
            parameter_count132954_no_phi=True, installation_and_optimizer_no_RNG_draws=True,
            fresh_LSB_and_MSB_construction_RNG_exact=True, initial_E4_E2M3_logits_equal_to_LSB=True,
            initial_E2M1_logits_use_different_MSB_mapping=True,
            historical_initial_RNG_snapshot_available=False)

    def test_five_epochs_equal_two_plus_three_resume_exactly(self):
        continuous = self.run_epochs(self.out / "continuous5")
        stopped = self.run_epochs(self.out / "stop2", stop=2)
        resumed = self.run_epochs(self.out / "resume3to5", resume=stopped.checkpoint)
        left, right = core.load_verified(continuous.checkpoint), core.load_verified(resumed.checkpoint)
        for name in ("model", "optimizer", "scheduler", "rng", "trace"):
            self.assertEqual(core.differences(left[name], right[name]), [], name)
        self.assertEqual(core.differences(continuous.initial_rng, stopped.initial_rng), [])
        self.assertNotEqual(support.legacy.state_digest(left["model"]), self.fixture.expected_initial_hash)
        checks = engine.verify_state(resumed.state, 10)
        for name, count in checks["BN_counts"].items():
            self.assertEqual(count, 0 if legacy.inactive_bn(name) else 10)
        self.assertEqual(left["scheduler"]["milestones"], {50: 1, 75: 1, 90: 1})
        # Changed method/schedule contracts must fail before any state restore.
        _, fresh = self.fresh()
        bad_contract = dict(self.contract(), method="lsb_trunc")
        with self.assertRaisesRegex(RuntimeError, "Hervatprotocol"):
            core.restore_checkpoint(core.load_verified(stopped.checkpoint), fresh.model, fresh.optimizer,
                fresh.scheduler, {}, self.fixture.rt.device, bad_contract)
        self.assertEqual(support.legacy.state_digest(fresh.model.state_dict()), self.fixture.expected_initial_hash)
        self.assertFalse(fresh.optimizer.state)
        self.evidence.update(continuous5_equals_stop2_resume3_exact_CPU=True,
            resume_equality_fields=["model", "optimizer", "scheduler", "rng", "trace"],
            real_optimizer_updates_in_comparison=20, Adam_and_active_BN_steps10=True,
            actual_MSB_mappings_and_dynamic_scale64_checked=True,
            original_100_epoch_LR_schedule_unchanged=True,
            wrong_method_resume_rejected_before_restore=True,
            continuous_checkpoint=str(continuous.checkpoint), resumed_checkpoint=str(resumed.checkpoint))

    def test_source_scale_changes_follow_changed_weights_in_actual_forwards(self):
        ctx, state = self.fresh()
        name = legacy.TARGET_LAYERS[0]
        quantizer = state.model.get_submodule(name).quantize_fn
        captured = []
        def capture(q, inputs, output):
            if q.weight_mode == 8:
                captured.append({bit: q._chain[bit].scale_codes.detach().clone() for bit in (4, 6, 8)})
        handle = quantizer.register_forward_hook(capture)
        data = SyntheticDataset("test")
        try:
            for attempt in range(2):
                folder = self.out / str(attempt)
                folder.mkdir()
                if attempt:
                    with torch.no_grad():
                        state.model.get_submodule(name).weight.mul_(8)
                result = engine.run_phase(ctx, state, [(data.data[:2], data.labels[:2])],
                                           0, False, folder, lambda _: None)
                self.assertTrue(result["audit"]["dynamic_source_plus6_verified"])
        finally:
            handle.remove()
        self.assertEqual(len(captured), 2)
        self.assertFalse(torch.equal(captured[0][8], captured[1][8]))
        for scale in captured:
            self.assertTrue(torch.equal(scale[6].int(), scale[8].int() + 6))
            self.assertTrue(torch.equal(scale[4], scale[6]))
        self.assertTrue(torch.equal(captured[1][6].int() - captured[0][6].int(),
                                    captured[1][8].int() - captured[0][8].int()))
        self.evidence["actual_forward_E4_scale_change_propagates_to_both_lower_scales"] = True

    def test_no_internal_reseed_and_old_pilot_settings_rejected(self):
        rt = support.runtime(self.fixture.config["seed"], "cpu", strict=False)
        torch.rand(1)  # A build_state() that silently reseeds would hide this change.
        with self.assertRaisesRegex(RuntimeError, "modelinitialisatie"):
            engine.build_state(rt, self.fixture.modules, self.fixture.config, self.fixture.expected_initial_hash)
        if self.fixture.expected_initial_hash != self.fixture.historical_initial_hash:
            rt = support.runtime(self.fixture.config["seed"], "cpu", strict=False)
            with self.assertRaisesRegex(RuntimeError, "actual=.*expected="):
                engine.build_state(rt, self.fixture.modules, self.fixture.config, self.fixture.historical_initial_hash)
            self.evidence["historical_hash_guard_rejects_observed_CPU_mismatch"] = True
        for patch_values in ({"lr": 1e-4}, {"scheduler_milestones": [2, 4]}):
            with self.assertRaises(RuntimeError):
                engine.build_state(rt, self.fixture.modules, {**self.fixture.config, **patch_values})
        _, state = self.fresh()
        state.model.register_parameter("phi", torch.nn.Parameter(torch.zeros(1)))
        with self.assertRaises(RuntimeError):
            engine.verify_state(state, 0)

    def test_partial_failure_reports_completed_update(self):
        ctx, state = self.fresh()
        data = SyntheticDataset("train")
        class BrokenLoader:
            def __len__(self):
                return 2
            def __iter__(self):
                yield data.data[:2], data.labels[:2]
                raise RuntimeError("Deliberate failure after the first complete update")
        with self.assertRaisesRegex(RuntimeError, "Deliberate failure"):
            engine.run_phase(ctx, state, BrokenLoader(), 1, True, self.out, lambda _: None)
        self.assertTrue(ctx.report["training_performed"])
        self.assertEqual(ctx.report["actual_new_optimizer_steps"], 1)
        engine.verify_state(state, 1)
        self.assertTrue(legacy.cache_is_empty(state.model))
        self.evidence["mid_epoch_failure_reports_real_completed_update"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT.resolve()):
        parser.error("Technische uitvoer moet buiten het pakket blijven.")
    ScratchTrainingTests.output_path = output
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ScratchTrainingTests))
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL",
        "synthetic_fixture_only": True, "SVHN_loaded": False, "parent_checkpoint_used": False,
        "scope": "Fresh actual fixed64 model, real CPU updates on three synthetic images; no research accuracy.",
        "tests_run": result.testsRun, "errors": len(result.errors), "failures": len(result.failures),
        "skipped": len(result.skipped), "seconds": time.perf_counter() - started,
        "vendor_and_historical_reference_unchanged": True if result.wasSuccessful() else None,
        **ScratchTrainingTests.evidence}
    if output.is_dir():
        (output / "training_engine_test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
