#!/bin/bash
# ---------------------------------------------------------------------------
# Environment setup for CalibPrompt
#
# Matches the reference environment used for all reported results
# (Python 3.10, PyTorch 2.1.0 + CUDA 12.1, NumPy 1.26).
#
#     conda create -n calibprompt python=3.10 -y
#     conda activate calibprompt
#     bash setup_env.sh
#
# For a different CUDA version, change the --index-url below
# (e.g. .../whl/cu118). CPU-only: use .../whl/cpu.
# ---------------------------------------------------------------------------
set -e

# 1. PyTorch (CUDA 12.1 build)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 2. CalibPrompt dependencies (numpy is pinned <2 here — install before building Dassl)
pip install -r requirements.txt

# 3. Dassl.pytorch (vendored framework).
#    --no-build-isolation lets Dassl's setup.py (which imports numpy at build time)
#    see the numpy we just installed, instead of failing in an isolated build env.
pip install -r Dassl.pytorch/requirements.txt
pip install -e Dassl.pytorch --no-build-isolation

# 4. Re-assert the pinned NumPy in case a transitive dep bumped it
pip install "numpy==1.26.3"

echo ""
echo "CalibPrompt environment ready — torch $(python -c 'import torch; print(torch.__version__)'), numpy $(python -c 'import numpy; print(numpy.__version__)')."
echo "Next: download datasets (docs/DATASETS.md) and pretrained Med-VLMs (docs/MODELS.md)."
