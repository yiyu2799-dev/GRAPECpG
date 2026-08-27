# GRAPECpG
Reference implementation of GRAPE-CpG for single-cell DNA methylation status imputation.
![Workflow of GRAPE-CpG](figure1.png)

## Installation
We recommend the following setup for reproducing the experiments with GPU support.
```bash
conda env create -f environment.yml
conda activate grapecpg
```

Alternatively, create a Python 3.10 environment and install the dependencies with pip:
```bash
pip install -r requirements.txt
```

Optional tests can be run with:
```bash
python -m pytest -q
```

These tests are used to check the main code interfaces and key implementation logic before training. They are optional and do not participate in model training.

## Data preparation
Instructions for dataset download and preprocessing are provided separately in `DATA_PREPARATION.md`.

## Training
GRAPE-CpG is trained in three stages. Default model and training parameters are defined in `config.py` and can be overridden from the command line.
The examples below use **Hemato**. For another dataset, replace `Hemato` and `/path/to/data/Hemato` with the corresponding dataset name and processed data directory.
For reproducing the reported experiments, use `segment_size=1024` for Hemato, `512` for Neuron-Mouse, and `256` for Neuron-Homo.

### Stage 1: Global training
Stage 1 trains the global graph branch and its prediction decoder.
```bash
python train.py \
  --dataset_name Hemato \
  --processed_dir /path/to/data/Hemato \
  --stage stage1 \
  --log_dir stage1
```

### Stage 2: Local training
Stage 2 loads the best Stage-1 checkpoint, keeps the global branch frozen, and trains the local-neighbor branch together with the expanded decoder.
```bash
python train.py \
  --dataset_name Hemato \
  --processed_dir /path/to/data/Hemato \
  --stage stage2 \
  --load_checkpoint runs/Hemato/stage1/model_best_val_auroc.pt \
  --log_dir stage2
```

### Stage 3: Joint fine-tuning
Stage 3 loads the best Stage-2 checkpoint and jointly fine-tunes the global branch, local branch, and decoder.
```bash
python train.py \
  --dataset_name Hemato \
  --processed_dir /path/to/data/Hemato \
  --stage stage3 \
  --load_checkpoint runs/Hemato/stage2/model_best_val_auroc.pt \
  --log_dir stage3
```

## Evaluation
A trained checkpoint can be evaluated with:
```bash
python evaluate.py --help
```

## Imputation
Missing methylation states can be imputed with:
```bash
python impute.py --help
```

## Citation
Citation information will be added after publication.

## Acknowledgements
GRAPE [(Paper)](https://arxiv.org/abs/2010.16418) [(Code)](https://github.com/maxiaoba/GRAPE)

MambaCpG [(Paper)](https://doi.org/10.1093/bib/bbaf360) [(Code)](https://github.com/Lee-qiangzee/MambaCpG)