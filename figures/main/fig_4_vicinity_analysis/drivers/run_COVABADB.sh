#!/bin/bash
# CoV-AbDab vicinity batch driver (Figure 4).
# Activate the airr_atlas conda env before running.
# Precompute LD matrix → data/results/ld/LD_covabdab_bg2.npy

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/driver_env.sh"

chains=("paired_chain")
complexities=("embeddings" "attention_matrices_average_layers" "embeddings_unpooled")
models_s=("esm2_t33_650M_UR50D" "antiberta2-cssp")
layers=$(seq 0 32)

df_junction_colname="VH_VL"
df_affinity_colname="binding_label"
sample_sizes=(0)
chosen_metric="cosine"
max_parallel_jobs=5

input_metadata="${DATA}/metadata/covabdab_AND_background.tsv.gz"
LD_matrix="${DATA}/results/ld/LD_covabdab_bg2.npy"
result_dir_base="${RESULTS}/covabdab_density2"

get_paths() {
  local model=$1 chain=$2 layer=$3
  local emb_root="${EMBEDDINGS}/covabdab/${model}"

  if [[ "$model" == "esm2_t33_650M_UR50D" ]]; then
    input_idx="${emb_root}/covabdab_bg_ESM2_idx.csv"
  else
    input_idx="${emb_root}/covabdab_bg_AB2_idx.csv"
  fi
  input_embeddings="${emb_root}/${complexity}/covabdab_${model}_${complexity}_layer_$((layer + 1)).npy"
}

run_vicinity_batch
