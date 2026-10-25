"""Load the byte-identical parent runtime under an unambiguous module name."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REFERENCE_RUNTIME = ROOT / 'reference_runtime'
sys.path.insert(0, str(REFERENCE_RUNTIME))
_spec = importlib.util.spec_from_file_location('_mx_learned_scale_parent_runtime', REFERENCE_RUNTIME / 'run_train.py')
ref = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ref
_spec.loader.exec_module(ref)
s, t, a = ref.s, ref.t, ref.a
import resume_core as core
import torch

