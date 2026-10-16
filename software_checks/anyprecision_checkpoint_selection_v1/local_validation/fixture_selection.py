"""LOCAL ONLY: synthetic CIFAR files / tiny training contract; never research data.
Uses the original reference package's fixture shim without modifying any file.
"""
from pathlib import Path
import sys
p = Path(sys.argv[sys.argv.index('--reference-dir') + 1]).resolve()
sys.path.insert(0, str(p / 'local_validation'))
import fixture_reference
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import select_checkpoints
if __name__ == '__main__':
    raise SystemExit(select_checkpoints.main())
