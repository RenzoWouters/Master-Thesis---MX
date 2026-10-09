"""CPU-only developer test of training equivalence and stratified split invariants.

The original train.py:forward is AST-extracted, NOT imported as a launcher.
Only .cuda(...) calls in that test copy become .to('cpu', ...).
The underlying model/loss/optimizer modules are still the supplied code.
"""
import argparse
import ast
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_pilot as p
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import numpy as np
p.torch, p.np = torch, np
p.support.torch, p.support.np = torch, np

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
source, info = p.prepare_source(args.output_dir)
p.support.configure(101, torch.device('cpu'), 2)
factory, SoftCE, optimizer_factory, _ = p.support.import_source(source)
utils = importlib.import_module('utils')

class CpuAdapter(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            node.func.attr = 'to'
            node.args.insert(0, ast.Constant('cpu'))
        return node

parsed = ast.parse((source / 'train.py').read_text())
forward = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'][0]
module = ast.fix_missing_locations(CpuAdapter().visit(ast.Module(body=[forward], type_ignores=[])))
ns = dict(torch=torch, args=types.SimpleNamespace(bit_width_list='1,2,4,8,32', print_freq=100),
          AverageMeter=utils.AverageMeter, accuracy=utils.accuracy, logging=logging)
exec(compile(module, '<original_forward_cpu_test_only>', 'exec'), ns)
reference = factory(p.BITS)
candidate = factory(p.BITS)
candidate.load_state_dict(reference.state_dict())
opt_a = optimizer_factory(reference, 'adam', 0.001, 0.0)
opt_b = optimizer_factory(candidate, 'adam', 0.001, 0.0)
rng = torch.Generator().manual_seed(555)
steps = []
for step in (1, 2):
    x = torch.randn((2, 3, 32, 32), generator=rng)
    y = torch.randint(0, 10, (2,), generator=rng)
    reference.train()
    losses, _, _ = ns['forward']([(x,y)], reference, torch.nn.CrossEntropyLoss(), SoftCE(),
                                 0, True, opt_a, None)
    rows = p.train_batch(candidate, opt_b, x, y, SoftCE())
    record = dict(step=step,
        models_equal=not p.support.differences(reference.state_dict(), candidate.state_dict()),
        optimizers_equal=not p.support.differences(opt_a.state_dict(), opt_b.state_dict()),
        losses_equal=losses == [r['objective_loss'] for r in reversed(rows)])
    steps.append(record)
    assert all(record[k] for k in ['models_equal','optimizers_equal','losses_equal']), record

labels = np.tile(np.arange(10), 5000).tolist()
split = p.make_split(labels, 500, 100)
assert split == p.make_split(labels, 500, 100)
assert len(split['pilot_train_indices']) == 5000
assert len(split['pilot_validation_indices']) == 1000
assert len(split['full_train_pool']) == 45000
assert len(split['full_validation_pool']) == 5000
assert not set(split['pilot_train_indices']) & set(split['pilot_validation_indices'])
assert split['train_class_counts'] == [500] * 10
assert split['validation_class_counts'] == [100] * 10
small = p.make_split(labels, 2, 2)
assert set(small['pilot_train_indices']) <= set(split['pilot_train_indices'])
assert set(small['pilot_validation_indices']) <= set(split['pilot_validation_indices'])
# Metric aggregation must weight the final partial minibatch correctly.
acc = {bit:[dict(n=32, objective_type='test', objective_loss=1, hard_ce=1,correct1=16,correct5=24),
            dict(n=8, objective_type='test', objective_loss=3, hard_ce=3,correct1=0,correct5=8)]
       for bit in p.BITS}
row = p.summarize(acc,1,'test',0.001)[0]
assert row['objective_loss'] == 1.4 and row['top1_percent'] == 40 and row['top5_percent'] == 80
result = dict(status='PASS', device='CPU', torch=str(torch.__version__),
    training_equivalence_steps=steps, split_deterministic_and_disjoint=True,
    nested_subsets=True, weighted_metrics=True,
    limitations=['synthetic inputs only', 'CPU adapter for original cuda() calls in test-only function',
                 'no real CIFAR-10 download or full 3-epoch pilot', 'no Windows/CUDA verification here'])
p.json_write(args.output_dir / 'parity_protocol_report.json', result)
print(json.dumps(result, indent=2))
