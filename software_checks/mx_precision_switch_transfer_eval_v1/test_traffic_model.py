"""Closed-form checks for transfer counts and finite-buffer semantics."""
from __future__ import annotations
import unittest
from traffic_model import allocate_regions, make_switch_sequence, simulate_trace


class TrafficModelTests(unittest.TestCase):
    def test_line_crossing_and_separate_requests(self):
        regions = [{"id": "a", "size_bytes": 20, "kind": "codes", "layer": "L"}]
        requests = {4: [{"region": "a", "offset": 6, "nbytes": 6},
                        {"region": "a", "offset": 8, "nbytes": 1}]}
        # [6,12) crosses two 8-byte lines; second request touches the second again.
        out = simulate_trace(regions, requests, [4], line_bytes=8, cache_bytes=0)
        self.assertEqual(out["totals"]["requested_bytes"], 7)
        self.assertEqual(out["totals"]["line_requests"], 3)
        self.assertEqual(out["totals"]["read_bytes"], 24)
        cached = simulate_trace(regions, requests, [4], line_bytes=8, cache_bytes=16)
        self.assertEqual(cached["totals"]["read_bytes"], 16)
        self.assertEqual(cached["totals"]["hit_lines"], 1)

    def test_large_cache_reuses_shared_source(self):
        regions = [{"id": "source", "size_bytes": 65, "kind": "codes"}]
        requests = {8: [{"region": "source"}], 4: [{"region": "source"}]}
        out = simulate_trace(regions, requests, [8, 4, 8, 4], line_bytes=16, cache_bytes=80)
        self.assertEqual([r["read_bytes"] for r in out["per_inference"]], [80, 0, 0, 0])
        self.assertEqual(out["totals"]["requested_bytes"], 260)

    def test_threshold_lru_stream_thrashing(self):
        regions = [{"id": "source", "size_bytes": 65, "kind": "codes"}]
        req = {8: [{"region": "source"}]}
        below = simulate_trace(regions, req, [8, 8], line_bytes=16, cache_bytes=64)
        at = simulate_trace(regions, req, [8, 8], line_bytes=16, cache_bytes=80)
        self.assertEqual([r["read_bytes"] for r in below["per_inference"]], [80, 80])
        self.assertEqual([r["read_bytes"] for r in at["per_inference"]], [80, 0])

    def test_selective_sharing_against_separate_modes(self):
        shared = [{"id": "base", "size_bytes": 32, "kind": "codes"},
                  {"id": "extra", "size_bytes": 32, "kind": "codes"}]
        shared_req = {8: [{"region": "base"}, {"region": "extra"}], 4: [{"region": "base"}]}
        separate = [{"id": "high", "size_bytes": 64, "kind": "codes"},
                    {"id": "low", "size_bytes": 32, "kind": "codes"}]
        separate_req = {8: [{"region": "high"}], 4: [{"region": "low"}]}
        selected = simulate_trace(shared, shared_req, [8, 4], line_bytes=16, cache_bytes=64)
        direct = simulate_trace(separate, separate_req, [8, 4], line_bytes=16, cache_bytes=64)
        self.assertEqual(selected["totals"]["read_bytes"], 64)
        self.assertEqual(direct["totals"]["read_bytes"], 96)
        # At zero cache, selective has the SAME traffic as independently packed modes.
        for regions, req in [(shared, shared_req), (separate, separate_req)]:
            self.assertEqual(simulate_trace(regions, req, [8, 4], line_bytes=16,
                                           cache_bytes=0)["totals"]["read_bytes"], 96)

    def test_preload_separate_accounting_and_finite_capacity(self):
        regions = [{"id": "a", "size_bytes": 32, "kind": "codes"},
                   {"id": "b", "size_bytes": 32, "kind": "scales"}]
        req = {8: [{"region": "a"}, {"region": "b"}]}
        out = simulate_trace(regions, req, [8], line_bytes=16, cache_bytes=64,
                             preload_policy="all_regions")
        self.assertEqual(out["initial_load"]["read_bytes"], 64)
        self.assertEqual(out["inference_totals"]["read_bytes"], 0)
        self.assertEqual(out["totals"]["read_bytes"], 64)
        zero = simulate_trace(regions, req, [8], line_bytes=16, cache_bytes=0,
                              preload_policy="all_regions")
        self.assertEqual(zero["totals"]["read_bytes"], 128)
        self.assertEqual(zero["initial_load"]["cache_bytes_after"], 0)
        small = simulate_trace(regions, req, [8], line_bytes=16, cache_bytes=32,
                               preload_policy="all_regions")
        self.assertEqual(small["initial_load"]["cache_bytes_after"], 32)
        self.assertEqual(small["inference_totals"]["read_bytes"], 64)

    def test_start_mode_preload(self):
        regions = [{"id": "a", "size_bytes": 16, "kind": "aux"}]
        out = simulate_trace(regions, {8: [{"region": "a"}]}, [8, 8],
                             line_bytes=16, cache_bytes=16, preload_policy="start_mode")
        self.assertEqual(out["initial_load"]["read_bytes"], 16)
        self.assertEqual(out["inference_totals"]["read_bytes"], 0)
        self.assertEqual(out["totals"]["by_kind"]["aux"]["read_bytes"], 16)

    def test_separate_regions_on_same_line(self):
        regions = [{"id": "a", "size_bytes": 4, "kind": "codes", "start_byte": 0},
                   {"id": "b", "size_bytes": 4, "kind": "scales", "start_byte": 4}]
        out = simulate_trace(regions, {4: [{"region": "a"}, {"region": "b"}]}, [4],
                             line_bytes=8, cache_bytes=8)
        self.assertEqual(out["layout"]["unique_lines"], 1)
        self.assertEqual(out["totals"]["read_bytes"], 8)
        self.assertEqual(out["totals"]["hit_lines"], 1)

    def test_reject_overlapping_regions_and_out_of_range_read(self):
        with self.assertRaises(ValueError):
            allocate_regions([{"id": "a", "size_bytes": 8, "start_byte": 0},
                              {"id": "b", "size_bytes": 8, "start_byte": 4}], 8)
        with self.assertRaises(ValueError):
            simulate_trace([{"id": "a", "size_bytes": 8}],
                           {4: [{"region": "a", "offset": 7, "nbytes": 2}]}, [4],
                           line_bytes=8, cache_bytes=8)

    def test_scenarios_and_categories(self):
        self.assertEqual(make_switch_sequence([8, 6, 4], "alternating_extremes", 5), [8, 4, 8, 4, 8])
        self.assertEqual(make_switch_sequence([8, 6, 4], "cycle_all", 5), [8, 6, 4, 8, 6])
        self.assertEqual(make_switch_sequence([8, 6, 4], "dwell8_extremes", 9), [8] * 8 + [4])
        regions = [{"id": "a", "size_bytes": 1, "kind": "codes", "layer": "L"},
                   {"id": "b", "size_bytes": 1, "kind": "scales", "layer": "L"},
                   {"id": "c", "size_bytes": 1, "kind": "aux", "layer": "A"}]
        out = simulate_trace(regions, {4: [{"region": r["id"]} for r in regions]}, [4],
                             line_bytes=16, cache_bytes=0)
        self.assertEqual(out["totals"]["read_bytes"], 48)
        self.assertEqual(sum(r["read_bytes"] for r in out["per_layer"]), 48)
        self.assertEqual(sum(r["read_bytes"] for r in out["totals"]["by_kind"].values()), 48)
        self.assertEqual(out["layout"]["payload_bytes"], 3)
        self.assertEqual(out["layout"]["allocated_bytes"], 33)


def run_tests() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrafficModelTests)
    result = unittest.TestResult()
    suite.run(result)
    failures = [{"test": str(test), "traceback": trace} for test, trace in result.failures + result.errors]
    return {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": failures}


if __name__ == "__main__":
    unittest.main()
