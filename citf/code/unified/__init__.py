"""CITF final-unified design: joint scalar fusion and actual-action learning."""

import os
from pathlib import Path
import sys

IMPLEMENTATION_VERSION = 'unified-warmup-robust-td-20261107-r25'

_release_root = os.environ.get('CITF_RELEASE_ROOT')
if _release_root and Path(__file__).resolve().parent != Path(_release_root).resolve()/'citf/code/unified':
    raise RuntimeError('The executing unified package differs from the pinned CITF release.')


def runtime_release():
    if not _release_root:
        return None
    return dict(id=os.environ['CITF_RELEASE_ID'], root=str(Path(_release_root).resolve()),
                code=str(Path(__file__).resolve().parent), entry=sys.argv[0], python=sys.executable)
