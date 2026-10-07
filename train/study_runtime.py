"""Select the study's isolated dependencies before loading numerical libraries."""
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
directory = Path(__file__).with_name('_study_dependencies' if os.name == 'nt' else '_study_dependencies_linux')
if directory.exists():
    sys.path.insert(0, str(directory))
