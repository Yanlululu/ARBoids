"""Expose the shared public baselines to the legacy training package."""
from pathlib import Path
import sys

_CONTROL_CODE = Path(__file__).resolve().parents[2] / "citf" / "code"
if str(_CONTROL_CODE) not in sys.path:
    sys.path.insert(0, str(_CONTROL_CODE))
from citf.baselines import (SourceMAPPO, SourceMAAC, SourceObservationEncoder,
                            SourceGaussianPolicy, SourceCentralValue, SourceAttentionQ)
