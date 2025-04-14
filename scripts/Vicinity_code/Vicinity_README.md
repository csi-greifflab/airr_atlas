# Vicinity Pipeline README

## Overview

This README provides instructions for running the Vicinity pipeline, including details on the required arguments and general guidelines for using the scripts.

## Important Notes

- Ensure that the libraries installed for use with the `Vicinity_ggplot.r` script are correctly sourced from the intended folder.
- For further information about each argument, refer to the function `parse_arguments()` in the `vicinity_pipeline.py` code.

## Argument Descriptions

The following arguments are available for running the pipeline:

- **analysis\_name** (required): Name of the analysis, which will also be used as the name for the folder where the results are saved.
- **df\_junction\_colname** (default: 'junction\_aa'): Column name for junction amino acids in the metadata.
- **df\_affinity\_colname** (default: 'affinity'): Column name for affinity values in the metadata.
- **input\_idx** (optional): File path mapping the embeddings index to the metadata ID. This allows the code to be easily adapted to the new embedding output scheme. If not provided, the metadata and embeddings are assumed to be joined by row order.
- **input\_metadata** (required): Path to the input metadata file, containing the sequence labels and amino acid sequences for LD computation. Avoid numerical affinity labels.
- **input\_embeddings** (required): Path to the input embeddings file (PyTorch tensor), with a shape of N sequences by N features.
- **result\_dir** (default: './Vicinity\_results'): Parent directory for storing results.
- **save\_results** (flag): Use this flag to save the vicinity object as .pkl to disk. If you are only interested in getting the summary results, and dont plan to do further analysis on the individual sequences, set it to 'false'
- **compute\_LD** (flag): Flag to compute Levenshtein distance (LD). Use `precomputed_LD` if LD has already been computed.
- **plot\_results** (flag): Flag to generate visual plots using the ggplot script.
- **parallel** (flag): Flag to parallelize KNN search. Recommended for datasets with more than 500,000 sequences.
- **chosen\_metric** (default: 'euclidean'): Metric used for distance calculation, either 'cosine' or 'euclidean'.
- **sample\_size** (default: 0): Maximum sample size for each label. If 0, all sequences will be used.
- **LD\_sample\_size** (default: 10,000): Number of sequences used for the X vs. ALL LD calculations.
- **precomputed\_LD** (optional): Path to a precomputed CSV file with LD results, to avoid recomputation.
- **radius\_range** (default: "7,24,1"): Specify the minimum, maximum radius, and step size separated by commas (e.g., '7,24,1'). Obsolete.

## General Guidelines for Running the Pipeline

To run the pipeline, you need to specify several key arguments, such as `analysis_name`, `input_metadata`, and `input_embeddings`. Below are additional notes on usage:

- **Metadata, Embeddings, and Index Mapping**: The `input_metadata` file contains the labels and amino acid sequences, which are used for LD computation. The `input_embeddings` file should be in the form of a PyTorch tensor, containing the embeddings for each sequence. The `input_idx` file is used to map embeddings indices to metadata IDs. It is suggested to provide inputs with the index file to be consistent with the new embedding scheme format. If this file is not provided, metadata and embeddings are assumed to be aligned row-wise. It is highly recommended to double-check that embeddings and metadata are correctly sorted.
- **LD Computation**: Levenshtein Distance(LD) computation is computationally intensive and should be computed once per dataset. After obtaining LD results, use the `precomputed_LD` argument to avoid redundant calculations in subsequent analyses.
- **Plotting**: To visualize results, use the `plot_results` flag. The output will be generated using the `Vicinity_ggplot.r` script.
- **Parallelization**: The `parallel` flag is recommended when dealing with large datasets (over 500,000 sequences) and running a single instance of the script. However, avoid using this flag when running multiple scripts in parallel using GNU parallel.
- **Running Multiple Scripts**: When experimenting with different parameters (e.g., chain type, model, or embedding type), consider creating a bash script like `run_all_vicinity.sh` to parallelize tasks efficiently using GNU parallel.

## Example Command

```bash
python vicinity_pipeline.py \
  --analysis_name "my_analysis" \
  --input_metadata "data/metadata.csv" \
  --input_embeddings "data/embeddings.pt" \
  --compute_LD \
  --save_results \
  --plot_results
```

This command runs the pipeline, computes LD, saves results, and generates plots based on the specified inputs.
