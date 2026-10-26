"""Reuse the exact delivered freeze pilot and original training runtime."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PILOT_ROOT = ROOT / 'pilot_runtime'
sys.path.insert(0, str(PILOT_ROOT))
_spec = importlib.util.spec_from_file_location('_mx_freeze50_pilot_runtime', PILOT_ROOT / 'run_train.py')
pilot = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pilot
_spec.loader.exec_module(pilot)
ref, s, t, a, core, torch = pilot.ref, pilot.s, pilot.t, pilot.a, pilot.core, pilot.torch
import freeze_core as pilot_core

