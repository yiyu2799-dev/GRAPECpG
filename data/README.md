# Data Preparation
This document describes the datasets used by GRAPE-CpG and the commands required to convert and validate them before model training.

## Data sources
GRAPE-CpG uses three single-cell DNA methylation datasets:

| Dataset | Original source | Processed data used for reproduction |
| Hemato | [GSE87197](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87197) | [MambaCpG Zenodo release](https://doi.org/10.5281/zenodo.15571992) |
| Neuron-Mouse | [GSE97179](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE97179) | [MambaCpG Zenodo release](https://doi.org/10.5281/zenodo.15571992) |
| Neuron-Homo | [GSE97179](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE97179) | [MambaCpG Zenodo release](https://doi.org/10.5281/zenodo.15571992) |

Hemato is derived from human hematopoietic stem and progenitor cells. Neuron-Mouse and Neuron-Homo are mouse and human neuronal datasets, respectively.
For reproducing the experiments in this repository, download the processed MambaCpG files from Zenodo. Each dataset should contain:

```text
<Dataset>_X.npz
<Dataset>_y.npz
<Dataset>_pos.npz
```

## Data conversion
The examples below use **Hemato**. For another dataset, replace `Hemato` and `/path/to/data_source/Hemato` with the corresponding dataset name and processed data directory.

```bash
python scripts/convert_mambacpg_to_grapecpg.py   --input_dir /path/to/data_source/Hemato   --x_file Hemato_X.npz   --y_file Hemato_y.npz   --pos_file Hemato_pos.npz   --dataset_name Hemato   --output_dir /path/to/data/Hemato   --window 201   --pos_base auto   --val_chrom 5   --test_chrom 10
```

After conversion, each processed dataset contains:

```text
meth_matrix.npy
dna_windows_centerC.npy
pos.npy
chrom.npy
metadata.json
```

The experiments use chromosome 5 for validation, chromosome 10 for testing, and the remaining chromosomes for training.

## Data validation
```bash
python scripts/check_processed.py   --processed_dir /path/to/data/Hemato   --expected_window 201   --val_chrom 5   --test_chrom 10   --require_reference_lengths
```

