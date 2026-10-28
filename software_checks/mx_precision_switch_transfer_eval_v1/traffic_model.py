"""Deterministic parameter-read model; this is not a hardware timing simulator.

Each request names a contiguous span in an immutable packed region. Every line
intersecting that span is accessed once, in ascending address order. Separate
requests remain separate even if they touch the same line. A fully associative
read-only LRU cache persists between inference calls; activation buffers are
outside this cache. A miss transfers one full line. No prefetching, writes,
compression, bank conflicts, or compute timings are assumed.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

KINDS = ("codes", "scales", "aux")
FIELDS = ("requested_bytes", "line_requests", "hit_lines", "miss_lines", "read_bytes")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             f"{label} must be an integer >= {minimum}")
    return value


def _counter() -> dict:
    result = {key: 0 for key in FIELDS}
    result["by_kind"] = {kind: {key: 0 for key in FIELDS} for kind in KINDS}
    return result


def _add(counter: dict, kind: str, requested: int, accesses: int, hits: int,
         misses: int, line_bytes: int) -> None:
    values = (requested, accesses, hits, misses, misses * line_bytes)
    for key, value in zip(FIELDS, values):
        counter[key] += value
        counter["by_kind"][kind][key] += value


def _finish(counter: dict) -> dict:
    counter["hit_ratio"] = (counter["hit_lines"] / counter["line_requests"]
                            if counter["line_requests"] else 0.0)
    for item in counter["by_kind"].values():
        item["hit_ratio"] = item["hit_lines"] / item["line_requests"] if item["line_requests"] else 0.0
    return counter


def allocate_regions(regions: list[dict], line_bytes: int) -> dict:
    """Allocate deterministic line-aligned regions unless start_byte is explicit.

The returned allocated_bytes is the address span including alignment holes;
payload_bytes counts only actual region bytes. Alignment holes need not be read.
"""
    _integer(line_bytes, "line_bytes", 1)
    _require(line_bytes & (line_bytes - 1) == 0, "line_bytes must be a power of two")
    placed, ids, cursor = [], set(), 0
    for raw in regions:
        item = dict(raw)
        key = item.get("id")
        _require(isinstance(key, str) and key and key not in ids, "region ids must be unique nonempty strings")
        ids.add(key)
        size = _integer(item.get("size_bytes"), f"{key}.size_bytes")
        kind = item.get("kind", "codes")
        _require(kind in KINDS, f"Unknown region kind: {kind}")
        start = item.get("start_byte", ((cursor + line_bytes - 1) // line_bytes) * line_bytes)
        _integer(start, f"{key}.start_byte")
        item.update(start_byte=start, size_bytes=size, kind=kind,
                    layer=str(item.get("layer", key)))
        placed.append(item)
        cursor = max(cursor, start + size)
    nonempty = sorted((r["start_byte"], r["start_byte"] + r["size_bytes"], r["id"])
                      for r in placed if r["size_bytes"])
    for previous, current in zip(nonempty, nonempty[1:]):
        _require(previous[1] <= current[0], f"Overlapping regions {previous[2]} and {current[2]}")
    line_ids = set()
    for r in placed:
        if r["size_bytes"]:
            first = r["start_byte"] // line_bytes
            last = (r["start_byte"] + r["size_bytes"] - 1) // line_bytes
            line_ids.update(range(first, last + 1))
    return {"regions": placed, "payload_bytes": sum(r["size_bytes"] for r in placed),
            "allocated_bytes": cursor, "unique_lines": len(line_ids),
            "unique_line_bytes": len(line_ids) * line_bytes}


def make_switch_sequence(modes: list, scenario: str, n_inferences: int = 64,
                         dwell: int = 8) -> list:
    _integer(n_inferences, "n_inferences", 1)
    _integer(dwell, "dwell", 1)
    _require(bool(modes), "At least one mode is required")
    _require(len({str(x) for x in modes}) == len(modes), "Modes must be unique")
    # Callers supply descending precision order, as recorded in configuration.
    high, low = modes[0], modes[-1]
    if scenario == "alternating_extremes":
        return [high if i % 2 == 0 else low for i in range(n_inferences)]
    if scenario in ("dwell8_extremes", "dwell_extremes"):
        return [high if (i // dwell) % 2 == 0 else low for i in range(n_inferences)]
    if scenario == "cycle_all":
        return [modes[i % len(modes)] for i in range(n_inferences)]
    raise ValueError(f"Unknown switching scenario: {scenario}")


def simulate_trace(regions: list[dict], requests_by_mode: dict, mode_sequence: list,
                   *, line_bytes: int, cache_bytes: int,
                   preload_policy: str = "none", record_layers: bool = True) -> dict:
    """Count reads for a cold run with optional explicit initialization.

preload_policy=none starts empty. start_mode fetches the first mode's request
list once. all_regions fetches every stored region once in allocation order.
Preload accesses use the same finite LRU cache: loading a model larger than its
cache does not falsely mark the entire model resident. Initialization and
inference counters are returned separately and their sum is totals.
"""
    _integer(cache_bytes, "cache_bytes")
    _require(preload_policy in ("none", "start_mode", "all_regions"), "Unknown preload policy")
    _require(bool(mode_sequence), "mode_sequence must be nonempty")
    layout = allocate_regions(regions, line_bytes)
    placed = {r["id"]: r for r in layout["regions"]}
    cache_lines = cache_bytes // line_bytes
    cache: OrderedDict[int, None] = OrderedDict()
    normalized_modes = {str(k): v for k, v in requests_by_mode.items()}
    _require(len(normalized_modes) == len(requests_by_mode), "Ambiguous mode keys")
    prepared = {}

    def prepare(request: dict) -> tuple:
        key = request.get("region")
        _require(key in placed, f"Unknown requested region: {key}")
        r = placed[key]
        offset = _integer(request.get("offset", 0), f"{key} offset")
        nbytes = _integer(request.get("nbytes", r["size_bytes"] - offset), f"{key} nbytes")
        _require(offset + nbytes <= r["size_bytes"], f"Request exceeds region {key}")
        start = r["start_byte"] + offset
        first = start // line_bytes
        end = (start + nbytes + line_bytes - 1) // line_bytes if nbytes else first
        return (r["kind"], str(request.get("layer", r["layer"])), nbytes, first, end)

    for key, requests in normalized_modes.items():
        prepared[key] = [prepare(r) for r in requests]
    for mode in mode_sequence:
        _require(str(mode) in prepared, f"No requests for mode {mode}")
    totals, inference_totals, initial = _counter(), _counter(), _counter()
    per_inference, per_layer = [], []

    def execute(requests: list, counters: list[dict], layers: dict | None = None) -> None:
        for kind, layer, nbytes, first, end in requests:
            accesses = end - first
            hits = 0
            if cache_lines:
                for line in range(first, end):
                    if line in cache:
                        hits += 1
                        cache.move_to_end(line)
                    else:
                        cache[line] = None
                        if len(cache) > cache_lines:
                            cache.popitem(last=False)
            misses = accesses - hits
            for count in counters:
                _add(count, kind, nbytes, accesses, hits, misses, line_bytes)
            if layers is not None:
                _add(layers.setdefault(layer, _counter()), kind, nbytes, accesses, hits, misses, line_bytes)

    if preload_policy == "all_regions":
        preloads = [prepare({"region": r["id"]}) for r in layout["regions"]]
    elif preload_policy == "start_mode":
        preloads = prepared[str(mode_sequence[0])]
    else:
        preloads = []
    execute(preloads, [initial, totals])
    initial["cache_bytes_after"] = len(cache) * line_bytes
    for index, mode in enumerate(mode_sequence):
        row, layers = _counter(), {} if record_layers else None
        execute(prepared[str(mode)], [row, inference_totals, totals], layers)
        row.update(index=index, mode=mode, cache_bytes_after=len(cache) * line_bytes)
        per_inference.append(_finish(row))
        if layers is not None:
            for layer, count in layers.items():
                count.update(index=index, mode=mode, layer=layer)
                per_layer.append(_finish(count))
    return {"schema": "mx-parameter-transfer-lru-v1",
            "config": {"line_bytes": line_bytes, "requested_cache_bytes": cache_bytes,
                       "effective_cache_bytes": cache_lines * line_bytes,
                       "cache_policy": "fully_associative_LRU", "preload_policy": preload_policy,
                       "n_inferences": len(mode_sequence), "activations_share_parameter_cache": False,
                       "cold_reset_before_trace": True},
            "layout": layout, "initial_load": _finish(initial),
            "per_inference": per_inference, "per_layer": per_layer,
            "inference_totals": _finish(inference_totals), "totals": _finish(totals),
            "final_cache_bytes": len(cache) * line_bytes}

