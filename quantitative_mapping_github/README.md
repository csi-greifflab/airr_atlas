# AIRR Atlas: Quantitative Mapping of Antibody Embeddings

This part of the repository contains the code for the analysis shown in figure 3 and 4, and the corresponding supplementary figures.

## Scripts
 
### Embedding Extraction
 
| Script | Description |
|--------|-------------|
| `run_EMBEDDAIR.sh` | Main driver for embedding extraction using `pepe-cli`. All embeddings used in the analysis were generated with this script. |
| `PREPROCESSING.rmd` | R Markdown for preprocessing steps of some datasets. |
 
---

### Wasserstein Distance Analysis (Figure 3A and Supplementary)
 
| Script | Description |
|--------|-------------|
| `get_W2.py` | Computes the Wasserstein-2 (W2) distance matrix for each dataset. |
| `LD_pairwise_AG.py` | Computes the pairwise Levenshtein distance matrix for antigen-specific datasets (used in supplementary figures). |
| `W2_antigen-specific.py` | Computes the Wasserstein-2 distance matrix for the antigen-specific datasets. Takes mean-pooled embeddings and their metadata as input, and returns the W2 distance matrix and the corresponding plots (ESM2 and Ab2 separately). |

---

### t-SNE Analysis (Figure 3B-3C and Supplementary)
 
| Script | Description |
|--------|-------------|
| `fit_transform_TSNE_fig3.py` | Fits a t-SNE model on the mean-pooled embeddings of the iReceptor dataset and projects the embeddings into t-SNE space. The trained model is then used to project the mean-pooled embeddings of the other datasets. |
| `tsne_density_plot_script.Rmd` | R Markdown for t-SNE and density difference plots (Figure 3 and supplementary figures). Takes the t-SNE-reduced embeddings of iReceptor and the other datasets as input, performs the density difference analysis, and generates the manuscript figures. |
 
---

### Vicinity Analysis (Figures 4 and Supplementary)
 
| Script | Description |
|--------|-------------|
| `PREPROCESSING.rmd` | R Markdown for preprocessing steps prior to the Vicinity analysis. |
| `Vicinity_pipeline_final.py` | Core Vicinity analysis pipeline. Takes embeddings and metadata as input and produces a results folder with the Vicinity analysis output for each dataset and model configuration. |
| `Vicinity_analysis_class_final.py` | Class-based implementation supporting the Vicinity analysis pipeline. |
| `Vicinity_plots_script.rmd` | R Markdown for generating Vicinity plots (Figure 4 and supplementary figures). Uses the result folders produced by `Vicinity_pipeline_final.py` to reproduce the plots shown in the manuscript. |
| `get_LD_matrix.py` | Computes the Levenshtein (LD) distance matrix for each dataset. Required as input for the Vicinity analysis pipeline, used as the sequence-level comparator against the embeddings. |


### Other folders

- `drivers/`: Bash scripts to run the Vicinity analysis across different datasets and model configurations.
- `data/`: 
  - `sequences/`: FASTA files and sampled sequences.
  - `metadata/`: Metadata for the various antibody datasets (zipped).
  - `results/`: Summarized results used as input to reproduce the figures in the manuscript.
- `figures/`: Output directory for generated figures.



