# AIRR Atlas: Quantitative Mapping of Antibody Embeddings

This repository contains the code and summarized data for the quantitative mapping of antibody embeddings, as presented in the manuscript "AIRR Atlas".

## Repository Structure

- `scripts/`: Implementation of the Vicinity analysis (Python) and plotting scripts (R).
  - `run_EMBEDDAIR.sh`: Main driver for embedding extraction using `pepe-cli`.
  - `Vicinity_pipeline_final.py`: Core Vicinity analysis engine.
  - `Vicinity_plots_script.rmd`: R Markdown for generating Vicinity density plots.
  - `tsne_density_plot_script.Rmd`: R Markdown for t-SNE and density difference plots.
- `drivers/`: Bash scripts to run the Vicinity analysis across different datasets and model configurations.
- `data/`: 
  - `sequences/`: FASTA files and sampled sequences.
  - `metadata/`: Metadata for the various antibody datasets (zipped).
  - `results/`: Summarized results used to reproduce the figures in the manuscript.
- `figures/`: Output directory for generated figures.

## Getting Started

### Prerequisites

- Python 3.8+
- R 4.0+
- Conda (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/csi-greifflab/airr-atlas.git
   cd airr-atlas/quantitative_mapping
   ```

2. Create and activate the conda environment:
   ```bash
   conda activate /doctorai/niccoloc/envs/embeddair_nick
   # Note: For external users, please install pepe-cli: https://github.com/csi-greifflab/pepe-cli
   ```

### Running the Analysis

To extract embeddings:
```bash
./scripts/run_EMBEDDAIR.sh
```

To run the Vicinity analysis:
```bash
cd drivers
./run_CHINERY.sh
```

To generate figures:
Use RStudio to open and run the `.Rmd` files in the `scripts/` directory. The scripts are configured to use the summarized data in `data/results/` to reproduce the manuscript figures even without the full original datasets.

## Data Note

Due to size limitations, some large datasets (e.g., full iReceptor, paired chains) are provided as samples or summarized results. For access to the full raw datasets, please refer to the links provided in the manuscript.

## License

[Insert License here, e.g., MIT]
