# Installation

CalibPrompt is built on the [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) framework (vendored in this repo) and the [CoOp](https://github.com/KaiyangZhou/CoOp) prompt-learning setup.

## 1. Create a conda environment

```shell
conda create -n calibprompt python=3.10 -y
conda activate calibprompt
```

## 2. Clone and install

```shell
git clone https://github.com/iabh1shekbasu/CalibPrompt
cd CalibPrompt
bash setup_env.sh
```

`setup_env.sh` reproduces the reference environment used for all reported results and performs:

1. Installs **PyTorch 2.1.0 + CUDA 12.1** (`torch==2.1.0`, `torchvision==0.16.0`). For a different CUDA version, change the `--index-url` in `setup_env.sh` (e.g. `.../whl/cu118`); CPU-only: `.../whl/cpu`.
2. Installs the CalibPrompt dependencies from `requirements.txt` — **NumPy is pinned to `1.26.3`** (torch 2.1 and Dassl are incompatible with numpy 2.x).
3. Installs the vendored `Dassl.pytorch` with `pip install -e Dassl.pytorch --no-build-isolation` (its `setup.py` imports numpy at build time, so build isolation must be disabled).

## 3. Prepare data and models

- **Datasets:** follow [DATASETS.md](DATASETS.md) and set `DATASET_ROOT` to the directory that holds them (default `./med-datasets`).
- **Pretrained Med-VLMs:** follow [MODELS.md](MODELS.md) and set `MODEL_ROOT` to the directory that holds the weights (default `./models`). BioMedCLIP downloads automatically from the HuggingFace Hub.

## Notes
- The MedCLIP, BioMedCLIP, PLIP, and QuiltNet backbone code is vendored under `models/`. Only the weight files must be downloaded.
- If you already use the [BAPLe](https://github.com/asif-hanif/baple) environment, it is compatible with CalibPrompt (same four Med-VLMs); you only need to additionally install the calibration extras in `requirements.txt` (`netcal`, `statsmodels`, `scikit-learn`).
