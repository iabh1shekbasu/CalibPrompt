# Datasets

CalibPrompt uses five publicly available medical image classification datasets. They use the **same sources and preprocessing as [BAPLe](https://github.com/asif-hanif/baple)** — for the detailed, step-by-step download and preprocessing scripts, please follow **[BAPLe's DATASETS.md](https://github.com/asif-hanif/baple/blob/main/datasets/DATASETS.md)** (the preprocessing scripts live under [`baple/datasets/dataset_preprocessing`](https://github.com/asif-hanif/baple/tree/main/datasets/dataset_preprocessing)). This document lists the sources and the directory layout expected by CalibPrompt.

| Dataset | Type | Classes | Backbones | Source |
|:-- |:-- |:--: |:-- |:-- |
| COVID | Chest X-ray | 2 | MedCLIP, BioMedCLIP | [Kaggle: COVID-19 Radiography](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) |
| RSNA18 | Chest X-ray | 3 | MedCLIP, BioMedCLIP | [Kaggle: RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) |
| Kather | Histopathology | 9 | PLIP, QuiltNet | [Zenodo: NCT-CRC-HE-100K / CRC-VAL-HE-7K](https://zenodo.org/records/1214456) |
| PanNuke | Histopathology | 2 | PLIP, QuiltNet | [Warwick TIA: PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) |
| DigestPath | Histopathology | 2 | PLIP, QuiltNet | [DigestPath 2019 (Google Drive)](https://drive.google.com/drive/folders/1_19Nz7mPuLReYA60UAtcnsAotTqZk0Je) |

## Directory layout

Place all datasets under a single directory and point `DATASET_ROOT` at it (default `./med-datasets`). Each dataset must follow this structure:

```bash
med-datasets/                       # = DATASET_ROOT
    ├── covid/
    │   ├── images/
    │   │   ├── train/<class>/*.png
    │   │   └── test/<class>/*.png
    │   └── classnames.txt          # "<class_folder> <class name>" per line
    ├── rsna18/
    ├── kather/
    ├── pannuke/
    └── digestpath/
```

- The `train/` and `test/` directories each contain one sub-folder per class.
- `classnames.txt` maps each class **folder name** to its human-readable **class name** (one per line), matching the CoOp / BAPLe convention. Class order is determined by the alphabetical order of the class folders, so make sure `classnames.txt` is consistent with it.
- On the first run, each dataset loader (`datasets/<name>.py`) caches a `preprocessed.pkl` and, for few-shot, `split_fewshot/shot_<k>-seed_<s>.pkl` inside the dataset folder.

## Quick reference (per BAPLe)

- **COVID / RSNA18:** download from Kaggle, arrange into `images/{train,test}/<class>/`, add `classnames.txt`.
- **Kather:** download `NCT-CRC-HE-100K.zip` (train) and `CRC-VAL-HE-7K.zip` (test) from Zenodo, then run BAPLe's `process_kather.py`.
- **PanNuke / DigestPath:** download the folds, then run BAPLe's `process_*.py` and `train_test_split_*.py` scripts.

> Medical datasets are subject to their original licenses and access conditions; please obtain them from the official sources above.
