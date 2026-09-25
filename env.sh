# Source before running pipeline scripts: memory-safe defaults for the ~4-5 GB commit budget.
source "$(dirname "${BASH_SOURCE[0]}")/.venv/Scripts/activate"
export ARROW_DEFAULT_MEMORY_POOL=system   # freed Arrow buffers go back to the OS
export OPENBLAS_NUM_THREADS=1             # OpenBLAS thread buffers exhausted commit memory before
export PYTHONUNBUFFERED=1
