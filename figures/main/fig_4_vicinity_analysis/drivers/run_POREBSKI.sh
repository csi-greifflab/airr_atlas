#!/bin/bash
# Porebski DMS vicinity batch driver (Figure 4).
# Activate the airr_atlas conda env before running.
# Precompute LD matrix with scripts/get_LD_matrix.py → data/results/ld/porebski_LD_matrix.npy

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/driver_env.sh"

chains=("cdr3_only")
complexities=("embeddings" "attention_matrices_average_layers" "embeddings_unpooled")
models_s=("esm2_t33_650M_UR50D" "antiberta2-cssp")
layers=$(seq 0 32)

df_junction_colname="cdr3"
df_affinity_colname="binding_label"
sample_sizes=(0)
chosen_metric="cosine"
max_parallel_jobs=5

input_metadata="${DATA}/metadata/porebski_metadata.csv"
LD_matrix="${DATA}/results/ld/porebski_LD_matrix.npy"
result_dir_base="${RESULTS}/porebski_density2"

get_paths() {
  local model=$1 chain=$2 layer=$3
  local emb_root="${EMBEDDINGS}/porebski_npy/${model}"
  input_idx="${emb_root}/porebski_${chain}_idx.csv"
  input_embeddings="${emb_root}/${complexity}/porebski_${chain}_${model}_${complexity}_layer_$((layer + 1)).npy"
}

run_vicinity_batch
