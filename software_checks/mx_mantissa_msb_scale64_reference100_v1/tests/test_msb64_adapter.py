"""Synthetic technical checks only: no SVHN samples or research accuracy."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import msb64_adapter as adapter
import mantissa_rules as rules

legacy = adapter.legacy
ARCHIVE = ROOT / "vendor" / "legacy" / "upstream" / "Any-Precision-DNNs-master.zip"
sys.path.insert(0, str(ARCHIVE) + "/Any-Precision-DNNs-master")
models = importlib.import_module("models")


def make_model():
    return legacy.adapt_copy(models.svhnq([1, 2, 4, 8, 32], expand=2))


def state_copy(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class MSB64AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(20261113)

    def test_all64_exact_search_and_actual_chain_on_available_devices(self):
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                audit = adapter.verify_rules(device)
                self.assertEqual(audit["codes_checked"], 64)
                self.assertTrue(audit["exact_fraction_search_equal"])
                codes = torch.arange(64, dtype=torch.uint8, device=device).reshape(2, 1, 32)
                expected = torch.tensor(rules.lookup("msb_bestexp"), dtype=torch.uint8,
                                        device=device).reshape_as(codes)
                actual = adapter.recode_msb(codes)
                self.assertEqual(actual.device, codes.device)
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertTrue(torch.equal(actual, expected))
                c = codes.long()
                source_codes = (((c >> 5) << 7) | (((c >> 3) & 3) << 5) | (c & 7)).to(torch.uint8)
                source = legacy.high.MXTensor(source_codes, torch.full((2, 1), 127,
                                             dtype=torch.uint8, device=device), (2, 32))
                result = adapter.code_chain(source)
                self.assertIs(result[8], source)
                self.assertTrue(torch.equal(result[6].codes, codes))
                self.assertTrue(torch.equal(result[4].codes, expected))
                decoded = legacy.codec(4).MXTensor(expected, result[6].scale_codes, (2, 32)).decode()
                self.assertTrue(torch.equal(result[4].decode(), decoded))
                self.assertFalse(torch.equal(result[4].codes, legacy.code_chain(source)[4].codes))

    def test_public_boundary_rejects_invalid_inputs(self):
        for codes in (torch.tensor([64], dtype=torch.uint8), torch.tensor([255], dtype=torch.uint8),
                      torch.tensor([0], dtype=torch.int64), [0]):
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                adapter.recode_msb(codes)

    def test_signed_zeros_padding_and_source_are_preserved(self):
        source_codes = (torch.arange(128).reshape(2, 2, 32) % 126).to(torch.uint8)
        source_codes[:, -1, 3:] = 0
        source_codes[0, 0, 0] = 128  # Actual -0, distinct from +0 padding.
        source_codes[1, 0, 0] = 0
        scales = torch.tensor([[120, 127], [126, 128]], dtype=torch.uint8)
        before_codes, before_scales = source_codes.clone(), scales.clone()
        source = legacy.high.MXTensor(source_codes, scales, (2, 35))
        chain = adapter.code_chain(source)
        for bit in (8, 6, 4):
            self.assertTrue(bool((chain[bit].codes[:, -1, 3:] == 0).all()))
            self.assertTrue(bool(torch.signbit(chain[bit].decode()[0, 0])))
            self.assertFalse(bool(torch.signbit(chain[bit].decode()[1, 0])))
        self.assertEqual(int(chain[4].codes[0, 0, 0]), 8)
        self.assertTrue(torch.equal(chain[4].scale_codes, chain[6].scale_codes))
        self.assertTrue(torch.equal(chain[6].scale_codes.int(), scales.int() + 6))
        self.assertTrue(torch.equal(source_codes, before_codes))
        self.assertTrue(torch.equal(scales, before_scales))
        self.assertNotEqual(chain[4].scale_codes.data_ptr(), scales.data_ptr())

    def test_scale_overflow_is_rejected_even_for_zero_blocks_without_wrap(self):
        codes = torch.zeros((1, 1, 32), dtype=torch.uint8)
        allowed = legacy.high.MXTensor(codes, torch.tensor([[248]], dtype=torch.uint8), (1, 32))
        self.assertEqual(int(adapter.code_chain(allowed)[4].scale_codes[0, 0]), 254)
        for value in (249, 250, 254):
            scales = torch.tensor([[value]], dtype=torch.uint8)
            source = legacy.high.MXTensor(codes, scales, (1, 32))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "geen clipping of uint8-wrap"):
                adapter.code_chain(source)
            self.assertEqual(int(scales[0, 0]), value)

    def test_current_scales_are_dynamic_source_plus6_including_zero_blocks(self):
        effective = torch.linspace(.125, 1.25, 128).reshape(2, 64)
        effective[1] = 0
        before = adapter.encode_chain(effective)
        after = adapter.encode_chain(effective * 8)
        self.assertTrue(torch.equal(after[8].scale_codes[0].int(), before[8].scale_codes[0].int() + 3))
        self.assertTrue(torch.equal(after[8].scale_codes[1], before[8].scale_codes[1]))
        for chain in (before, after):
            for bit in (6, 4):
                self.assertTrue(torch.equal(chain[bit].scale_codes.int(), chain[8].scale_codes.int() + 6))
            self.assertTrue(bool((chain[8].scale_codes[1] == 127).all()))
            self.assertTrue(bool((chain[4].scale_codes[1] == 133).all()))

    def test_legacy_high_middle_and_full_G_gradient_are_exact_no_factor64(self):
        weight = torch.linspace(-1.3, 1.7, 105).reshape(3, 35)
        upstream = torch.linspace(-.7, .9, 105).reshape_as(weight)
        for bit in (8, 6, 4):
            with self.subTest(bit=bit):
                old_weight = weight.clone().requires_grad_()
                new_weight = weight.clone().requires_grad_()
                direct_weight = weight.clone().requires_grad_()
                old, new = legacy.JointMXQuantizer(), adapter.MSB64Quantizer()
                old.weight_mode = new.weight_mode = bit
                old_output, new_output = old(old_weight), new(new_weight)
                if bit in (8, 6):
                    self.assertTrue(torch.equal(old_output, new_output))
                expected = adapter.encode_chain(legacy.effective_reference_weight(new_weight))[bit].decode()
                self.assertTrue(torch.equal(new_output, expected))
                old_output.backward(upstream)
                new_output.backward(upstream)
                legacy.effective_reference_weight(direct_weight).backward(upstream)
                self.assertTrue(torch.equal(old_weight.grad, new_weight.grad))
                self.assertTrue(torch.equal(new_weight.grad, direct_weight.grad))
                self.assertFalse(torch.equal(new_weight.grad, upstream))  # G's derivative is retained.

    def test_install_preserves_rng_state_values_parameters_and_bn(self):
        model = make_model().eval()
        before = state_copy(model)
        identities = {name: id(p) for name, p in model.named_parameters()}
        buffers = {name: id(p) for name, p in model.named_buffers()}
        before_rng, before_python = torch.get_rng_state(), random.getstate()
        audit = adapter.install(model)
        self.assertEqual(audit["parameter_count"], 132954)
        self.assertTrue(audit["rng_unchanged"])
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        self.assertEqual(identities, {name: id(p) for name, p in model.named_parameters()})
        self.assertEqual(buffers, {name: id(p) for name, p in model.named_buffers()})
        self.assertEqual(before.keys(), model.state_dict().keys())
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()))
        self.assertTrue(all(not model.get_submodule(name).quantize_fn.training for name in legacy.TARGET_LAYERS))
        self.assertFalse(any("phi" in name for name in model.state_dict()))
        self.assertEqual(adapter.install(model)["method"], "msb_bestexp")  # Idempotent.
        self.assertTrue(adapter.verify_model(model)["cache_empty"])

    def test_shared_triplet_uses_actual_msb_cached_codes_and_isolated_bn(self):
        reference = make_model()
        untouched = state_copy(reference)
        model = copy.deepcopy(reference)
        adapter.install(model)
        model.train()
        observations = {name: [] for name in legacy.TARGET_LAYERS}
        lookup = torch.tensor(rules.lookup("msb_bestexp"), dtype=torch.uint8)
        handles = []

        def observe(name):
            def hook(q, args, output):
                chain = q._chain
                self.assertIsNotNone(chain)
                self.assertTrue(torch.equal(chain[4].codes, lookup[chain[6].codes.long()]))
                self.assertTrue(torch.equal(chain[6].scale_codes.int(), chain[8].scale_codes.int() + 6))
                self.assertTrue(torch.equal(chain[4].scale_codes, chain[6].scale_codes))
                self.assertTrue(torch.equal(output, chain[q.weight_mode].decode()))
                observations[name].append((q.weight_mode, id(chain), chain[8].codes.data_ptr(),
                                           chain[6].scale_codes.data_ptr()))
            return hook

        for name in legacy.TARGET_LAYERS:
            handles.append(model.get_submodule(name).quantize_fn.register_forward_hook(observe(name)))
        inputs = torch.randn(2, 3, 40, 40)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.zero_grad(set_to_none=True)
        try:
            with mock.patch.object(legacy.high, "encode_tensor", wraps=legacy.high.encode_tensor) as high_encode:
                with mock.patch.object(legacy.codec(6), "encode_tensor", side_effect=AssertionError("Direct E2 encoding")):
                    with mock.patch.object(legacy.codec(4), "encode_tensor", side_effect=AssertionError("Direct E2 encoding")):
                        with legacy.shared_source_triplets(model) as audit:
                            for bit in (8, 6, 4):
                                legacy.select_mode(model, bit)
                                model(inputs).square().mean().backward()
                self.assertEqual(high_encode.call_count, 6)
            self.assertTrue(audit["source_cache_cleared"])
            self.assertEqual(audit["source_triplets"], 1)
            self.assertEqual(audit["high_encodings"], 6)
            self.assertTrue(legacy.cache_is_empty(model))
            for rows in observations.values():
                self.assertEqual([row[0] for row in rows], [8, 6, 4])
                self.assertEqual(len({row[1:] for row in rows}), 1)
            for name, parameter in model.named_parameters():
                if legacy.inactive_bn(name):
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            before = model.get_submodule("layers.3").weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.get_submodule("layers.3").weight))
            for name, value in model.state_dict().items():
                if name.endswith("num_batches_tracked"):
                    self.assertEqual(int(value), 0 if legacy.inactive_bn(name) else 1)
            for name, parameter in model.named_parameters():
                if legacy.inactive_bn(name):
                    self.assertNotIn(parameter, optimizer.state)
            self.assertTrue(adapter.verify_model(model)["no_phi"])
            self.assertTrue(all(torch.equal(value, reference.state_dict()[name]) for name, value in untouched.items()))
            self.assertTrue(all(type(reference.get_submodule(name).quantize_fn) is legacy.JointMXQuantizer
                                for name in legacy.TARGET_LAYERS))
        finally:
            for handle in handles:
                handle.remove()

    def test_cache_guards_reject_changed_master_and_changed_codes(self):
        weight = torch.linspace(-1, 1, 64).reshape(2, 32).requires_grad_()
        for change in ("weight", "codes"):
            q = adapter.MSB64Quantizer()
            q._triplet_active = True
            q.weight_mode = 8
            q(weight)
            with torch.no_grad():
                if change == "weight":
                    weight.add_(.01)
                else:
                    q._chain[4].codes[0, 0, 0] ^= 1
            q.weight_mode = 6
            with self.subTest(change=change), self.assertRaises(ValueError):
                q(weight)


if __name__ == "__main__":
    unittest.main()
