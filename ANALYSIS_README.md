# AIRR Atlas — Analysis Pipeline

This document describes the analysis pipeline used to generate the results in the paper. The pipeline embeds antibody repertoire sequences using multiple representation models, reduces dimensionality, and performs downstream analyses (clustering, visualization, completeness metrics).

## Pipeline Overview

```
FASTA / CSV input
      │
      ▼
┌─────────────────────────────────────────┐
│  Embedding (scripts/)                   │
│  ESM-2, AntiBERTa2, ProtT5,            │
│  immune2vec, OHE                        │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Dimensionality Reduction (scripts/)    │
│  PCA → UMAP / t-SNE                    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Analysis (notebooks/)                  │
│  Completeness, VDJ bias, KDE,          │
│  antigen-specific, vicinity, etc.       │
└─────────────────────────────────────────┘
```

## Environment Setup

```bash
conda env create -f airr_atlas.yml
conda activate airr_atlas
```

The OHE encoding step additionally requires an immuneML environment (see `scripts/OHE_script.py` for details).

## Scripts Reference

| Script | Description | Usage |
|--------|-------------|-------|
| `embedding_utils.py` | Shared utilities for FASTA parsing, CDR3 loading, and embedding export | Imported by `antiberta2_cdr3.py` and `esm2_cdr3.py` |
| `antiberta2_cdr3.py` | Compute AntiBERTa2 embeddings (full-length or CDR3) | `python antiberta2_cdr3.py --fasta_path <fa> --output_path <pt> [--cdr3_path <csv>] [--layers -1] [--pooling True]` |
| `esm2_cdr3.py` | Compute ESM-2 embeddings (full-length or CDR3) | `python esm2_cdr3.py --fasta_path <fa> --output_path <pt> [--cdr3_path <csv>] [--layers -1] [--pooling True]` |
| `esm2.py` | Compute ESM-2 embeddings (simple interface) | `python esm2.py <fasta_path> <output_path>` |
| `esm2_nopooling.py` | Compute ESM-2 per-residue embeddings (no pooling) | `python esm2_nopooling.py --fasta_path <fa> --output_path <pt> [--layers -1]` |
| `prott5.py` | Compute ProtT5 embeddings | `python prott5.py <fasta_path> <output_path>` |
| `immune2vec.py` | Train and apply immune2vec model | `python immune2vec.py <input_csv> <output_path> <dim> <workers>` |
| `OHE_script.py` | One-hot encoding via immuneML | `python OHE_script.py --input_path <csv> --folder_path <dir> --output_path <dir> --labels <col> --junction_aa <col> --yaml_template <yaml>` |
| `reduce_dim.py` | PCA dimensionality reduction (to 100 dims) | `python reduce_dim.py <input.pt> <output.pt>` |
| `build_umap.py` | Build UMAP projections from embeddings | `python build_umap.py <embedding_subdir>` |
| `build_tsne.py` | Build t-SNE projections from embeddings | `python build_tsne.py <embedding_subdir>` |
| `dbscan.py` | DBSCAN clustering on embeddings | `python dbscan.py <input.pt> <output.pkl>` |
| `create_umap_dataset.py` | Subsample datasets for UMAP at various sizes | `python create_umap_dataset.py <directory> <filename.pt>` |
| `draw_kde.py` | Draw KDE density plots | `python draw_kde.py <input_dir>` |
| `draw_png.py` | Draw scatter plots from UMAP/t-SNE results | `python draw_png.py <input_dir>` |

## Notebooks Reference

| Notebook | Description |
|----------|-------------|
| `ag_specific.ipynb` | Antigen-specific analysis (trastuzumab affinity) |
| `completeness.ipynb` | Completeness metric evaluation across embedding models |
| `completeness_tz.ipynb` | Completeness analysis on trastuzumab dataset |
| `baseline.ipynb` | Baseline completeness analysis (1- and 2-patient subsets) |
| `ld_ed_correlation.ipynb` | Levenshtein distance vs. embedding distance correlation |
| `length_vdj_bias.ipynb` | Sequence length and VDJ gene usage bias analysis |
| `tsne.ipynb` | t-SNE visualization of embedding spaces |
| `vicinity.ipynb` | Vicinity-based neighborhood analysis |
| `vdj.ipynb` | V/D/J gene usage analysis across embeddings |
| `w2_random_data.ipynb` | Wasserstein-2 distance analysis on random/shuffled data |

Small CSV data files used by completeness notebooks are stored in `notebooks/data/`.

## Driver Scripts

The `drivers/` directory contains bash scripts that run the embedding and analysis scripts on an HPC cluster. These scripts contain HPC-specific paths and must be modified for your environment.

## Reproducing the Analysis

1. **Set up the environment** using `airr_atlas.yml`
2. **Prepare input data**: FASTA files of antibody sequences and CDR3 annotations
3. **Generate embeddings** by running the driver scripts (or the Python scripts directly) for each model
4. **Reduce dimensionality** with `reduce_dim.py`, then project with `build_umap.py` / `build_tsne.py`
5. **Run analysis notebooks** in `notebooks/` to reproduce figures and metrics

## Data Availability

Input sequence data and pre-computed embeddings are available as described in the paper. The `sample_files/` directory contains example input formats.

## Libraries

The `libraries/immune2vec_model/` directory contains the immune2vec library (from [Yaari Lab](https://bitbucket.org/yaarilab/immune2vec_model)).
