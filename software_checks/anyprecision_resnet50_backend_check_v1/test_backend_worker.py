"""CPU-only orchestration tests; these mocks do not validate CUDA performance."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import backend_worker


class FakeOOM(RuntimeError):
    pass


class FakeCUDA:
    OutOfMemoryError = FakeOOM

    def __init__(self, trace):
        self.trace = trace
        self.initialized = False
        self.allocated = 100
        self.reserved = 200
        self.peak_allocated = 100
        self.peak_reserved = 200

    def is_initialized(self):
        return self.initialized

    def synchronize(self):
        self.trace.append("sync")

    def mem_get_info(self):
        self.trace.append("memory_query")
        return 1000 - self.allocated, 1000

    def memory_allocated(self):
        self.trace.append("memory_query")
        return self.allocated

    def memory_reserved(self):
        self.trace.append("memory_query")
        return self.reserved

    def max_memory_allocated(self):
        self.trace.append("memory_query")
        return self.peak_allocated

    def max_memory_reserved(self):
        self.trace.append("memory_query")
        return self.peak_reserved

    def reset_peak_memory_stats(self):
        self.trace.append("peak_reset")
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def allocate(self, peak, after):
        self.peak_allocated = max(self.peak_allocated, peak)
        self.peak_reserved = max(self.peak_reserved, peak + 100)
        self.allocated = after
        self.reserved = after + 100


class FileLog:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8")

    def __call__(self, message):
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def _case(failure=None):
    trace = []
    cuda = FakeCUDA(trace)
    torch = ModuleType("torch")
    torch.cuda = cuda
    torch.backends = SimpleNamespace(cudnn=SimpleNamespace(benchmark=True))
    torch.float32 = "mock-float32"
    x, y = object(), object()

    def randn(*shape, **kwargs):
        assert shape == (256, 3, 224, 224)
        assert kwargs == {"device": "cuda", "dtype": torch.float32}
        return x

    def randint(*args, **kwargs):
        assert args == (0, 1000, (256,)) and kwargs == {"device": "cuda"}
        return y

    torch.randn, torch.randint = randn, randint
    config = {"seed": 123, "cudnn_benchmark": True}
    configure_calls = []

    def configure(seed, threads):
        configure_calls.append((seed, threads))
        cuda.initialized = True
        torch.backends.cudnn.benchmark = True
        return {"seed": seed, "threads": threads, "cudnn_benchmark": True,
                "AMP": False, "fp32_precision": "ieee",
                "device": {"name": "MOCK ONLY", "total_memory_bytes": 1000}}

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    common = SimpleNamespace(CONFIG=config, BITS=(1, 2, 4, 8, 32), Log=FileLog,
                             configure=configure, require=require, json_write=write_json,
                             utc_now=lambda: "MOCK-TIME")
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 16)])
    optimizer, soft = object(), object()

    def make_model(source, device):
        assert source == Path("mock-source") and device == "cuda:0"
        assert torch.backends.cudnn.benchmark is False
        return model, optimizer, soft

    totals_calls = []
    reference = SimpleNamespace(c=common, model_and_optimizer=make_model,
        fresh_totals=lambda: {}, add_totals=lambda totals, metrics: totals_calls.append(metrics),
        assert_finite_checkpoint_state=lambda m, o: require(m is model and o is optimizer, "wrong state"))
    engine = ModuleType("engine")
    engine_calls = []

    def train_step(m, o, inputs, labels, loss, **kwargs):
        assert (m is model and o is optimizer and inputs is x and labels is y and loss is soft)
        assert torch.backends.cudnn.benchmark is False
        assert kwargs["mode"] == "serial_dp4"
        engine_calls.append({"validate": kwargs["validate"], "progress": "progress" in kwargs})
        trace.append("train_warmup" if kwargs["validate"] else "train_measured")
        call = len(engine_calls)
        if call == 3 and failure:
            raise FakeOOM("mock OOM") if failure == "OOM" else RuntimeError("mock failure")
        if kwargs["validate"]:
            kwargs["progress"]({"stage": "forward_start", "bit": 32, "shard": 1, "shards": 4})
            kwargs["progress"]({"stage": "forward_done", "bit": 32, "shard": 1, "shards": 4})
            kwargs["progress"]({"stage": "backward_start", "bit": 32, "shard": 1, "shards": 4})
            kwargs["progress"]({"stage": "backward_done", "bit": 32, "shard": 1, "shards": 4})
        cuda.allocate((500, 600, 550)[call - 1], (150, 180, 180)[call - 1])
        return {"MOCK_METRICS": call}

    engine.train_step = train_step
    clock_calls = []

    def clock():
        trace.append("clock")
        clock_calls.append(len(clock_calls) + 1)
        return float(clock_calls[-1])

    with tempfile.TemporaryDirectory(prefix="r50-backend-worker-test-") as temporary:
        out = Path(temporary) / "worker"
        args = SimpleNamespace(worker_out=out, source=Path("mock-source"), cpu_threads=4)
        with patch.dict(sys.modules, {"torch": torch, "engine": engine}), \
                patch.object(backend_worker.time, "perf_counter", clock):
            code = backend_worker.run_worker(args, reference)
        result = json.loads((out / "result.json").read_text())
        progress = json.loads((out / "progress.json").read_text())
        assert configure_calls == [(123, 4)]
        assert config == {"seed": 123, "cudnn_benchmark": True}, "Reference config was mutated."
        assert result["eligible_as_pilot_check"] is False
        assert result["runtime"]["cudnn_benchmark"] is False
        assert engine_calls == [{"validate": True, "progress": True},
                               {"validate": False, "progress": False},
                               {"validate": False, "progress": False}]
        assert result["peak_allocated_bytes"] == 600
        assert result["peak_reserved_bytes"] == 700
        assert result["memory_updates"][0]["peak_allocated_bytes"] == 500
        assert result["memory_updates"][1]["peak_allocated_bytes"] == 600
        if failure:
            assert result["status"] == failure and code == (3 if failure == "OOM" else 1)
            assert result["step_seconds"] == [1.] and progress["step_seconds"] == [1.]
            assert result["optimizer_updates_completed"] == 2
            assert result["memory_updates"][-1]["completed"] is False
        else:
            assert result["status"] == "PASS" and code == 0
            assert result["step_seconds"] == [1., 1.] and progress["step_seconds"] == [1., 1.]
            assert result["optimizer_updates_completed"] == 3
            assert result["memory_updates"][2]["peak_allocated_bytes"] == 550
            assert result["physical_forward_backward_passes"] == 60
        # In each completed measured interval, only the engine and final CUDA
        # sync occur between timer reads: no memory queries or progress writes.
        timer_indices = [i for i, item in enumerate(trace) if item == "clock"]
        for start in range(2, len(timer_indices) - 1, 2):
            inside = trace[timer_indices[start] + 1:timer_indices[start + 1]]
            assert inside == ["train_measured", "sync"], inside
        assert not any(out.rglob("*.pt"))
        return {"status": "PASS", "mock_worker_status": result["status"],
                "reference_config_unchanged": True, "no_midstep_observation": True,
                "partial_times_preserved": True, "cumulative_peaks_preserved": True}


def run_selftests():
    assert __debug__, "Run without -O; assertions are required."
    return {"status": "PASS", "device": "mock / no CUDA",
            "scope": "Worker orchestration, timing boundaries and persistence only; no GPU/model equivalence claim.",
            "cases": {"success": _case(), "failure": _case("FAIL"), "oom": _case("OOM")}}


if __name__ == "__main__":
    print(json.dumps(run_selftests(), indent=2))
