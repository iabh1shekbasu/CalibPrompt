# Installation

CalibPrompt is built on the [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) framework (vendored in this repo) and the [CoOp](https://github.com/KaiyangZhou/CoOp) prompt-learning setup.

## 1. Create a conda environment

```shell
conda create -n calibprompt python=3.9 -y
conda activate calibprompt
```

## 2. Clone and install

```shell
git clone https://github.com/iabh1shekbasu/CalibPrompt
cd CalibPrompt
bash setup_env.sh
```

`setup_env.sh` performs the following steps:

1. Installs PyTorch (`torch==1.13.0+cu116`, `torchvision==0.14.0+cu116`). Change the `--extra-index-url` in `setup_env.sh` to match your CUDA version if needed.
2. Installs the vendored `Dassl.pytorch` (`pip install -r requirements.txt` + `python setup.py develop`).
3. Installs the CalibPrompt dependencies from `requirements.txt` (transformers, timm, open-clip deps, netcal, statsmodels, scikit-learn, etc.).

## 3. Prepare data and models

- **Datasets:** follow [DATASETS.md](DATASETS.md) and set `DATASET_ROOT` to the directory that holds them (default `./med-datasets`).
- **Pretrained Med-VLMs:** follow [MODELS.md](MODELS.md) and set `MODEL_ROOT` to the directory that holds the weights (default `./models`). BioMedCLIP downloads automatically from the HuggingFace Hub.

## Notes
- The MedCLIP, BioMedCLIP, PLIP, and QuiltNet backbone code is vendored under `models/`. Only the weight files must be downloaded.
- If you already use the [BAPLe](https://github.com/asif-hanif/baple) environment, it is compatible with CalibPrompt (same four Med-VLMs); you only need to additionally install the calibration extras in `requirements.txt` (`netcal`, `statsmodels`, `scikit-learn`).
