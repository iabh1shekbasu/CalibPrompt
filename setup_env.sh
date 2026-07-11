#!/bin/bash
# ---------------------------------------------------------------------------
# Environment setup for CalibPrompt
#
# Recommended: create and activate a fresh conda env first, then run this script:
#     conda create -n calibprompt python=3.9 -y
#     conda activate calibprompt
#     bash setup_env.sh
#
# The CUDA 11.6 PyTorch build below matches the paper's setup. Change the
# --extra-index-url to match your local CUDA version if needed.
# ---------------------------------------------------------------------------
set -e

# 1. PyTorch
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# 2. Dassl.pytorch (dataset + training framework, vendored in this repo)
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..

# 3. Pin setuptools for compatibility with the pinned deps
pip install setuptools==59.5.0

# 4. CalibPrompt dependencies
pip install -r requirements.txt

echo ""
echo "CalibPrompt environment ready."
echo "Next: download the datasets (docs/DATASETS.md) and the pretrained Med-VLMs (docs/MODELS.md)."
