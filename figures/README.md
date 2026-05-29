# AIRR Atlas — Figures

This is the canonical index for all figures in the repository. Paths below are relative to `figures/`.

For per-figure detail on Figure 2 see `main/fig_2_w2_distance_matrices/fig_2_README.MD`,
for Figures 3–4 see `main/fig_3_antigen_specificity/fig_3_README.md`.

---

## Main Figures

### Figure 1 — Abstract

Output image: `main/fig_1_abstract/FIG_1_compressed.png`

No analysis scripts; the figure is a graphical abstract.

---

### Figure 2 — W₂ Distance Matrices

#### Shared embedding pipeline (used across 2A, 2B, 2C and supplementary S1–S4, S11–S13)

| Step | Script | Driver |
|------|--------|--------|
| AB2 embeddings | `main/fig_2_w2_distance_matrices/scripts/antiberta2_cdr3.py` | `main/fig_2_w2_distance_matrices/drivers/run_antiberta2_cdr3.bash` |
| ESM2 embeddings | `main/fig_2_w2_distance_matrices/scripts/esm2_cdr3.py` | `main/fig_2_w2_distance_matrices/drivers/run_esm2.bash` |
| OHE embeddings | `main/fig_2_w2_distance_matrices/scripts/OHE_script.py` | `main/fig_2_w2_distance_matrices/drivers/run_ohe.bash` |
| tSNE reduction | `main/fig_2_w2_distance_matrices/scripts/build_tsne.py` | `main/fig_2_w2_distance_matrices/drivers/run_tsne.bash` |

#### Figure 2A — Qualitative visualization of global PLM embedding structure

Visualization notebook: `main/fig_2_w2_distance_matrices/notebook/fig_2a_tsne.ipynb`

#### Figure 2B — Quantitative assessment of global embedding similarity

Completeness and W₂ calculation notebook: `main/fig_2_w2_distance_matrices/notebook/fig_2b_completeness.ipynb`

#### Figure 2C — Clustering quality across pipeline modifications (Briney dataset)

| Step | File |
|------|------|
| Dataset construction | `main/fig_2_w2_distance_matrices/notebook/Briney_dataset_construction.ipynb` |
| W₂ calculation | `main/fig_2_w2_distance_matrices/scripts/Briney_experiment_W2_calculation.py` |
| Comparison | `main/fig_2_w2_distance_matrices/notebook/fig_2c_comparison.ipynb` |

---

### Figure 3 — Antigen Specificity

#### Embedding extraction

| Script | Description |
|--------|-------------|
| `main/fig_3_antigen_specificity/drivers/run_PEPE.sh` | Main driver for embedding extraction using `pepe-cli` |
| `main/fig_3_antigen_specificity/notebook/PREPROCESSING.rmd` | Preprocessing for some datasets |

#### Wasserstein Distance Analysis (Figure 3A and supplementary)

| Script | Description |
|--------|-------------|
| `main/fig_3_antigen_specificity/scripts/get_W2.py` | W₂ distance matrix for each dataset |
| `main/fig_3_antigen_specificity/scripts/W2_antigen-specific.py` | W₂ distance matrix for antigen-specific datasets |
| `main/fig_3_antigen_specificity/scripts/LD_pairwise_AG.py` | Pairwise Levenshtein distance matrix for antigen-specific datasets |

#### t-SNE Analysis (Figures 3B–3C and supplementary S6, S7)

| Script | Description |
|--------|-------------|
| `main/fig_3_antigen_specificity/scripts/fit_transform_TSNE_fig3.py` | Fits and applies t-SNE model on mean-pooled embeddings |
| `main/fig_3_antigen_specificity/notebook/tsne_density_plot_script.Rmd` | Density difference plots for Fig 3 and S6/S7 |

#### Vicinity Analysis (Figures 4 and supplementary)

| Script | Description |
|--------|-------------|
| `main/fig_3_antigen_specificity/scripts/Vicinity_pipeline_final.py` | Core Vicinity analysis pipeline |
| `main/fig_3_antigen_specificity/scripts/Vicinity_analysis_class_final.py` | Class supporting the Vicinity pipeline |
| `main/fig_3_antigen_specificity/scripts/get_LD_matrix.py` | Levenshtein distance matrix (input to Vicinity) |
| `main/fig_3_antigen_specificity/notebook/Vicinity_plots_script.rmd` | Vicinity plots for Fig 4 and supplementary |

---

### Figure 4 — Vicinity Analysis

Uses all scripts from Figure 3 above. Dataset-specific drivers:

| Driver |
|--------|
| `main/fig_4_vicinity_analysis/drivers/run_CHINERY.sh` |
| `main/fig_4_vicinity_analysis/drivers/run_COVABADB.sh` |
| `main/fig_4_vicinity_analysis/drivers/run_ENGHELHART.sh` |
| `main/fig_4_vicinity_analysis/drivers/run_POREBSKI.sh` |
| `main/fig_4_vicinity_analysis/drivers/run_VARUN.sh` |

---

## Supplementary Figures

| Figure | Notes | Notebook / script |
|--------|-------|-------------------|
| S1 | PLM embeddings reveal distinct structure from 30k sequences | Same pipeline as Fig 2 (embeddings + tSNE) |
| S2 | TCRβ embeddings require larger sample sizes than BCRs | Same pipeline as Fig 2 |
| S3 | Trastuzumab variant HCDR3 completeness | `supplementary/fig_s3/notebooks/fig_s3_completeness_tz.ipynb` |
| S4 | W₂ separates immune repertoires from random controls | `supplementary/fig_s4/notebooks/fig_s4_w2_random_data.ipynb` |
| S5 | *(previously undocumented)* | `supplementary/fig_s5/notebook/fig_S5.ipynb` |
| S6 | Pairwise Levenshtein distance distributions | Script: `main/fig_3_antigen_specificity/scripts/LD_pairwise_AG.py`; output in `main/fig_3_antigen_specificity/output/` |
| S7 | Supplementary density t-SNE | Script: `main/fig_3_antigen_specificity/notebook/tsne_density_plot_script.Rmd`; output in `main/fig_3_antigen_specificity/output/` |
| S8 | TZ supplementary layers | Output image only: `supplementary/fig_s8/FIG_S8_TZ_supplementary_layers.png` |
| S10 | DMS supplementary layers | Output image only: `supplementary/fig_s10/fig_s10_DMS_supplementary_layers.png` |
| S11 | Pipeline modification performance | `supplementary/fig_s11/notebooks/fig_s11_data_overlap.ipynb` |
| S12 | Length bias in PLM embeddings | `supplementary/fig_s12/notebooks/fig_s12_length_vdj_bias.ipynb` |
| S13 | Residual V(D)J gene bias | Same notebook as S12: `supplementary/fig_s12/notebooks/fig_s12_length_vdj_bias.ipynb` |
| S14 | CR9114 | Output image only: `supplementary/fig_s14/fig_s14_cr9114.png` |
| S15 | AlphaSeq scatter HB selection | Output image only: `supplementary/fig_s15/fig_s15_alphaseq_scatter_hb_selection.png` |
