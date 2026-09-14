# Data Preparation

This document describes the datasets used by GRAPE-CpG and the commands required to convert and validate them before model training.

## Data sources

A complete processed-data package for reproducing the GRAPE-CpG experiments will be provided separately.

**Complete GRAPE-CpG data package:**  
URL: 

GRAPE-CpG is evaluated on four single-cell DNA methylation datasets:

| Dataset | Original source | Processed data used for reproduction |
| Hemato | [GSE87197](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87197) | [MambaCpG Zenodo release](https://doi.org/10.5281/zenodo.15571992) |
| Neuron-Mouse | [GSE97179](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE97179) | [MambaCpG Zenodo release](https://doi.org/10.5281/zenodo.15571992) |
| Neuron-Homo | [GSE97179](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE97179) | [MambaCpG Zenodo release](https://doi.org/10.5281/zenodo.15571992) |
| LG_ACCPU_chr18 | [HumanCellEpigenomeAtlas metadata / methylation data](https://huggingface.co/datasets/zhoujt1994/HumanCellEpigenomeAtlas_metadata) | GRAPE-CpG data package (URL to be added) |

Hemato is derived from human hematopoietic stem and progenitor cells. Neuron-Mouse and Neuron-Homo are mouse and human neuronal datasets, respectively. `LG_ACCPU_chr18` is a single-chromosome dataset used to evaluate GRAPE-CpG under a within-chromosome positional splitting protocol.

For reproducing the Hemato, Neuron-Mouse, and Neuron-Homo experiments in this repository, download the processed MambaCpG files from Zenodo. Each dataset should contain:

```text
<Dataset>_X.npz
<Dataset>_y.npz
<Dataset>_pos.npz
```

For `LG_ACCPU_chr18`, the processed GRAPE-CpG input files will be distributed directly through the complete GRAPE-CpG data package. Raw-data preprocessing for this dataset is therefore not included in this repository.

## Data conversion

The conversion command below is intended for datasets distributed in the MambaCpG format. The example uses **Hemato**. For another compatible dataset, replace `Hemato` and `/path/to/data_source/Hemato` with the corresponding dataset name and processed-data directory.

```bash
python scripts/convert_mambacpg_to_grapecpg.py \
  --input_dir /path/to/data_source/Hemato \
  --x_file Hemato_X.npz \
  --y_file Hemato_y.npz \
  --pos_file Hemato_pos.npz \
  --dataset_name Hemato \
  --output_dir /path/to/data/Hemato \
  --window 201 \
  --pos_base auto \
  --val_chrom 5 \
  --test_chrom 10
```

After conversion, each processed dataset contains:

```text
meth_matrix.npy
dna_windows_centerC.npy
pos.npy
chrom.npy
metadata.json
```

## Data splitting protocols

GRAPE-CpG supports two genomic data-splitting protocols.

### Chromosome holdout

For datasets containing multiple chromosomes, complete chromosomes can be held out for validation and testing. In the reported experiments on Hemato, Neuron-Mouse, and Neuron-Homo, chromosome 5 is used for validation, chromosome 10 is used for testing, and the remaining chromosomes are used for training.

This is the default splitting mode:

```bash
--split_mode chromosome_holdout \
--val_chrom 5 \
--test_chrom 10
```

The validation and test chromosomes can be changed through the command-line arguments.

### Within-chromosome split

For a dataset restricted to a single chromosome, GRAPE-CpG also supports a contiguous positional split within that chromosome.CpG sites are first sorted in ascending genomic position and are then divided according to the requested fractions. For the `LG_ACCPU_chr18` experiment, chromosome 18 is divided into 80% training, 10% validation, and 10% testing sites:

```bash
--split_mode within_chromosome \
--split_chrom 18 \
--split_fractions 0.8 0.1 0.1
```

The fractions are defined according to the number of CpG sites after sorting by genomic position, rather than by physical chromosome length.

To prevent information leakage across genomic partitions, segmentation is performed separately within the training, validation, and test regions. Overlap context and local CpG neighborhoods are restricted to the corresponding split and never cross split boundaries.

## Data validation

For a chromosome-holdout dataset such as Hemato:

```bash
python scripts/check_processed.py \
  --processed_dir /path/to/data/Hemato \
  --expected_window 201 \
  --val_chrom 5 \
  --test_chrom 10 \
  --require_reference_lengths
```

For a within-chromosome dataset such as `LG_ACCPU_chr18`:

```bash
python scripts/check_processed.py \
  --processed_dir /path/to/data/LG_ACCPU_chr18 \
  --expected_cells 111 \
  --expected_window 201 \
  --split_mode within_chromosome \
  --split_chrom 18 \
  --split_fractions 0.8 0.1 0.1 \
  --require_reference_lengths
```
