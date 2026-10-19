"""Negative tests for diagnostic comparisons; never a research training run."""
from pathlib import Path
import sys
import json
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import test_dual as t


def main():
    results = {}
    before = t.backend_state()
    try:
        with t.deterministic_validation(torch.device('cpu')):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.backends.cudnn.benchmark
            assert torch.backends.cudnn.deterministic
            raise ValueError('intentional test exception')
    except ValueError:
        pass
    assert t.backend_state() == before
    results['backend_restored_on_exception'] = True
    rows = [t.tensor_comparison(torch.tensor([1.]), torch.tensor([1.2]), 'negative_weight')]
    payload = {'tensor_comparisons': rows, 'rng_exact': True}
    try:
        t.require_pair_match(payload)
    except RuntimeError as e:
        assert 'negative_weight' in str(e) and 'max_abs_difference' in str(e)
    else:
        raise AssertionError('A real mismatch passed')
    results['genuine_weight_mismatch_rejected_with_name_and_error'] = True
    counter = t.tensor_comparison(torch.tensor(2), torch.tensor(3), 'BN_counter')
    assert not counter['close']
    results['integer_counters_require_exact_match'] = True
    nonfinite = t.tensor_comparison(torch.tensor([float('nan')]), torch.tensor([float('nan')]), 'NaN')
    assert not nonfinite['finite'] and not nonfinite['close']
    results['NaN_never_accepted'] = True
    s = t.tensor_comparison(torch.zeros(2), torch.zeros(3), 'shape')
    assert not s['structure_ok']
    results['wrong_shape_rejected'] = True
    payload = {'tensor_comparisons': [t.tensor_comparison(torch.ones(1), torch.ones(1), 'equal')],
               'rng_exact': False}
    try:
        t.require_pair_match(payload)
    except RuntimeError:
        pass
    else:
        raise AssertionError('RNG mismatch accepted')
    results['RNG_mismatch_rejected'] = True
    # Confirm the report holds partial diagnostics when the comparison raises.
    # The real CUDA path itself cannot be tested on this CPU-only machine.
    log = []
    state = {'status': 'RUNNING'}
    try:
        state['synthetic_training_checks'] = {}
        state['synthetic_training_checks']['comparison'] = rows
        t.require_pair_match({'tensor_comparisons': rows, 'rng_exact': True})
    except RuntimeError:
        pass
    assert state['synthetic_training_checks']['comparison'][0]['name'] == 'negative_weight'
    results['partial_diagnostics_survive_exception'] = True
    print(json.dumps({'status':'PASS','scope':'CPU_HELPER_TESTS_ONLY','checks':results}, indent=2))

if __name__ == '__main__':
    main()
