"""Offline CPU proof: only a high MX export and synthetic images are supplied.
The child receives no direct-low export and no training checkpoint. All
quantizer entrypoints and G(W) are replaced by traps during C derivation/load.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from unittest.mock import patch
import svhn_support as s
import resume_core as core
import mx_format as high
import mx_e2m1 as low
import mx_adapter
import dual_adapter
import truncation as tr
import evaluation as ev


def main():
    source_file, image_file, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source, _ = s.legacy.prepare_source(output)
    rt = s.runtime(20261009, 'cpu', strict=False)
    modules = s.import_source(source)
    digest = core.file_sha256(source_file)
    payload = ev.read_export(source_file, digest)
    def forbidden(*args, **kwargs):
        raise AssertionError('Forbidden FP32 re-quantization/master transform during C-only reconstruction')
    with patch.object(high, 'encode_tensor', forbidden), patch.object(high, 'encode_elements', forbidden), \
         patch.object(low, 'encode_tensor', forbidden), patch.object(low, 'encode_elements', forbidden), \
         patch.object(mx_adapter, 'effective_reference_weight', forbidden), \
         patch.object(dual_adapter, 'effective_reference_weight', forbidden):
        derived = tr.derive_payload(payload, 'mantissa_truncation', digest)
        path = output / 'derived.pt'
        saved = core.atomic_save(path, derived)
        model = tr.load_derived_model(modules.models, ev.read_export(path, saved['sha256']), 'cpu')
        inputs = torch.load(image_file, map_location='cpu', weights_only=True)
        with torch.no_grad():
            logits = model(inputs)
    assert core.file_sha256(source_file) == digest
    core.atomic_save(output / 'logits.pt', {'logits': logits})
    s.write_json(output / 'fresh_process_report.json', {
        'synthetic_fixture_only': True, 'status': 'PASS',
        'source_file_unchanged': True, 'master_checkpoint_provided': False,
        'direct_low_export_provided': False, 'quantizer_and_G_entrypoints_trapped': True,
        'optimizer_steps': 0})
    print('Fresh-process C reconstruction PASS (synthetic CPU; no training).')


if __name__ == '__main__':
    main()
