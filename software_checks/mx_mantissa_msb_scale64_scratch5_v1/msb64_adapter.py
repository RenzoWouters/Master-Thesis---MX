"""MSB mantissa training on the original dynamic fixed-x64 MX route.

Only E2M3 -> E2M1 changes. The E4 source, E4 -> E2 exponent mapping,
source-scale+6 rule, separate G(W) graphs and identity STE are retained.
There are no learned scale parameters and no global monkeypatches.
"""
from __future__ import annotations

from pathlib import Path
import random
import sys

import torch
from torch import Tensor, nn

import mantissa_rules as rules


_VENDOR = Path(__file__).resolve().parent / "vendor" / "legacy"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
import trunc_adapter as legacy

for _module in (legacy, legacy.base, legacy.high, legacy.posthoc,
                legacy.codec(6), legacy.codec(4), legacy.base.original_adapter):
    if not Path(_module.__file__).resolve().is_relative_to(_VENDOR):
        raise ImportError("Legacy-module uit een andere package geladen: " + _module.__name__)


METHOD = "msb_bestexp"
EXPECTED_PARAMETER_COUNT = 132954
CACHE_POLICY = "one_dynamic_E4_source_per_layer_per_triplet_fresh_G_graph_per_pass"
require = legacy.require


def _recode_valid(codes: Tensor) -> Tensor:
    """Device-local integer rule; input was validated at the MX boundary.

    Retain m0 and the sign. Raise the exponent only for m=6, saturating at
    E=3. This equals the exact four-exponent search with the documented ties.
    No host copy, element loop, floating search or runtime lookup table.
    """
    c = codes.to(torch.int64)
    sign, exponent, mantissa = (c >> 5) << 3, (c >> 3) & 3, c & 7
    result = sign | ((exponent + (mantissa == 6).long()).clamp_max(3) << 1) | (mantissa & 1)
    return result.to(torch.uint8)


def recode_msb(codes: Tensor) -> Tensor:
    """Validate arbitrary-shaped uint8 E2M3 codes and map on their device."""
    require(isinstance(codes, Tensor) and codes.dtype == torch.uint8,
            "Verwacht een uint8 E2M3-codetensor.")
    require(bool((codes <= 63).all()), "E2M3-codes moeten binnen 0..63 liggen.")
    return _recode_valid(codes)


def code_chain(source: legacy.high.MXTensor) -> dict:
    """Original E4/E2M3 route; MSB E2M1 uses the same shifted scale bytes."""
    naive = legacy.exponent_truncate(source)
    shifted = legacy.shifted_scale_codes(source)
    middle = legacy.codec(6).MXTensor(naive.codes, shifted, source.shape)
    middle.validate()
    low = legacy.codec(4).MXTensor(_recode_valid(middle.codes),
                                  middle.scale_codes.clone(), source.shape)
    low.validate()
    require(torch.equal(middle.scale_codes.to(torch.int32), source.scale_codes.to(torch.int32) + 6)
            and torch.equal(middle.scale_codes, low.scale_codes),
            "Vaste schaaloffset +6 of mantissaschaal gewijzigd.")
    return {8: source, 6: middle, 4: low}


def encode_chain(effective: Tensor) -> dict:
    return code_chain(legacy.high.encode_tensor(effective))


class MSB64Quantizer(legacy.JointMXQuantizer):
    """Original cache protocol, state and STE, with only the low mapping new."""

    def __init__(self, previous: legacy.JointMXQuantizer | None = None):
        if previous is not None:
            require(type(previous) in (legacy.JointMXQuantizer, MSB64Quantizer),
                    "Verwacht de oorspronkelijke fixed64-quantizer.")
            require(not previous._triplet_active and previous._chain is None
                    and previous._source_key is None and previous._versions is None
                    and not previous._used_modes, "Installatie vereist een lege broncache.")
            require(not previous.state_dict(), "De fixed64-quantizer mag geen tensorstaat bevatten.")
        super().__init__()
        if previous is not None:
            self.weight_mode = previous.weight_mode
            self.abit, self.wbit = previous.abit, previous.wbit
            self.train(previous.training)

    @property
    def method(self):
        return METHOD

    def forward(self, weight: Tensor) -> Tensor:
        require(self.abit == self.wbit == 32, "A32 en W32-voorbewerking moeten behouden blijven.")
        bit = self.weight_mode
        legacy.codec(bit)
        # Separate backwards need separate G graphs. Cache integer codes only.
        effective = legacy.effective_reference_weight(weight)
        if self._triplet_active:
            require(bit not in self._used_modes, "Dubbele modus in dezelfde broncyclus.")
            key = (id(weight), weight._version, weight.device, weight.dtype, tuple(weight.shape))
            if self._chain is None:
                self._chain = encode_chain(effective)
                self._source_key = key
                self._versions = {b: (x.codes._version, x.scale_codes._version)
                                  for b, x in self._chain.items()}
            else:
                require(key == self._source_key, "FP32-master gewijzigd tussen precisiepasses.")
                self._assert_cache_unchanged()
            encoded = self._chain[bit]
            self._used_modes.append(bit)
        else:
            encoded = encode_chain(effective)[bit]
        return legacy._CachedCodeSTE.apply(effective, encoded.codes, encoded.scale_codes, bit)


def _tensor_exact(left: Tensor, right: Tensor) -> bool:
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left, right)
            and (not left.is_floating_point() or torch.equal(torch.signbit(left), torch.signbit(right))))


def _rng_snapshot() -> dict:
    # Do not initialize CUDA just to audit an installation on a CPU model.
    result = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
              "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
    numpy = sys.modules.get("numpy")
    result["numpy"] = numpy.random.get_state() if numpy is not None else None
    return result


def _rng_equal(left: dict, right: dict) -> bool:
    if left["python"] != right["python"] or not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    for key in ("torch_cuda", "numpy"):
        if (left[key] is None) != (right[key] is None):
            return False
    if left["torch_cuda"] is not None:
        if len(left["torch_cuda"]) != len(right["torch_cuda"]) or not all(
                torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"])):
            return False
    if left["numpy"] is not None:
        a, b = left["numpy"], right["numpy"]
        if a[0] != b[0] or not (a[1] == b[1]).all() or a[2:] != b[2:]:
            return False
    return True


def verify_model(model: nn.Module) -> dict:
    """Audit the installed parameter-free rule at an empty-cache boundary."""
    require(not hasattr(model, "_frozen_weight_bit"), "Training vereist FP32-masters.")
    require(getattr(model, "_mantissa_method", None) == METHOD, "MSB-methodecontract ontbreekt.")
    require(not getattr(model, "_absolute_e2_frozen", False),
            "Dit experiment vereist dynamische fixed64-schalen, geen absolute E2-freeze.")
    names = tuple(n for n, m in model.named_modules()
                  if isinstance(m, nn.Conv2d) and hasattr(m, "quantize_fn"))
    require(names == legacy.TARGET_LAYERS, "Niet het afgesproken CNN-8-model.")
    parameters = dict(model.named_parameters())
    require(sum(p.numel() for p in parameters.values()) == EXPECTED_PARAMETER_COUNT,
            "Afwijkend aantal modelparameters; verwacht 132954 zonder phi.")
    require(not any("phi" in name.split(".") for name in model.state_dict()),
            "Onafhankelijke schaalparameter phi is niet toegestaan.")
    for name in legacy.TARGET_LAYERS:
        q = model.get_submodule(name).quantize_fn
        require(type(q) is MSB64Quantizer and q.method == METHOD,
                "Onjuiste MSB fixed64-quantizer: " + name)
        require(not hasattr(q, "phi") and not q.state_dict()
                and not list(q.parameters()) and not list(q.buffers()),
                "De quantizer mag geen parameters, buffers of phi bevatten.")
        require(q.abit == q.wbit == 32 and q.weight_mode in legacy.MODES,
                "Gewichtsmodus of A32-instelling gewijzigd.")
    require(legacy.cache_is_empty(model), "Modelaudit vereist een lege broncache.")
    return {"method": METHOD, "parameter_count": EXPECTED_PARAMETER_COUNT,
            "quantizers": len(legacy.TARGET_LAYERS), "no_phi": True,
            "new_trainable_parameters": 0, "dynamic_scales": True,
            "scale_exponent_offset": 6, "scale_factor": 64,
            "cache_policy": CACHE_POLICY, "cache_empty": True,
            "legacy_shared_source_triplets_compatible": True,
            "legacy_identity_STE_to_G_reused": True, "backward_scale_factor": 1}


def install(model: nn.Module) -> dict:
    """Install MSB in place without changing tensor values, identities or RNG.

    The method is Python configuration, deliberately not a new state_dict key.
    The runner must store it in the full-checkpoint contract and install this
    adapter before resuming. No standalone compact weight export is provided.
    """
    require(not hasattr(model, "_frozen_weight_bit"), "Training vereist FP32-masters.")
    quantizers = [(name, model.get_submodule(name).quantize_fn) for name in legacy.TARGET_LAYERS]
    require(all(type(q) in (legacy.JointMXQuantizer, MSB64Quantizer) for _, q in quantizers),
            "Verwacht de zes oorspronkelijke fixed64-quantizers.")
    require(legacy.cache_is_empty(model), "Installatie vereist een lege broncache.")
    require(sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETER_COUNT,
            "Afwijkend model of verborgen schaalparameters.")
    before_rng = _rng_snapshot()
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    buffer_ids = {name: id(value) for name, value in model.named_buffers()}
    replacements = [(name, q if type(q) is MSB64Quantizer else MSB64Quantizer(q))
                    for name, q in quantizers]
    for name, q in replacements:
        model.get_submodule(name).quantize_fn = q
    model._mantissa_method = METHOD
    audit = verify_model(model)
    after_state = model.state_dict()
    require(before_state.keys() == after_state.keys()
            and all(_tensor_exact(value, after_state[name]) for name, value in before_state.items()),
            "Installatie veranderde modelwaarden of state_dict-sleutels.")
    require(parameter_ids == {name: id(value) for name, value in model.named_parameters()},
            "Installatie verving Parameterobjecten.")
    require(buffer_ids == {name: id(value) for name, value in model.named_buffers()},
            "Installatie verving modelbuffers.")
    require(_rng_equal(before_rng, _rng_snapshot()), "Installatie veranderde een RNG-toestand.")
    return {**audit, "state_dict_keys_and_values_unchanged": True,
            "parameter_identities_unchanged": True, "buffer_identities_unchanged": True,
            "rng_unchanged": True, "model_local_rule": True}


@torch.no_grad()
def verify_rules(device="cpu") -> dict:
    """Exhaustive audit against the independent exact-Fraction search."""
    source = torch.arange(64, dtype=torch.uint8, device=device)
    actual = recode_msb(source)
    expected = torch.tensor(rules.lookup(METHOD), dtype=torch.uint8, device=device)
    require(torch.equal(actual, expected), "MSB-bitregel wijkt af van de exacte exponentzoekregel.")
    require(torch.equal(source.long() >> 5, actual.long() >> 3), "Tekenbit gewijzigd.")
    require(torch.equal(source & 1, actual & 1), "De behouden m0-bit is gewijzigd.")
    require(int(actual[0]) == 0 and int(actual[32]) == 8, "Nultekens gewijzigd.")
    return {"method": METHOD, "device": str(source.device), "codes_checked": 64,
            "exact_fraction_search_equal": True, "sign_preserved": True,
            "mantissa_m0_preserved": True, "signed_zero_preserved": True,
            "integer_device_rule": True, "exponent_ties": "min_abs_Eprime_minus_E_then_min_Eprime",
            "rule": "Eprime=min(E+int(m==6),3); mprime=m&1"}
