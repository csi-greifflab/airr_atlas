#!/bin/bash
# CR9114 (Brian Hie) vicinity batch driver (Figure 4, Supplementary S14).
# Activate the airr_atlas conda env before running.
# Precompute LD matrix → data/results/ld/LD_mat_brian_hie.npy

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/driver_env.sh"

chains=("heavy_chain")
complexities=("embeddings" "attention_matrices_average_layers" "embeddings_unpooled")
models_s=("esm2_t33_650M_UR50D" "antiberta2-cssp")
layers=$(seq 0 32)

df_junction_colname="Sequence"
df_affinity_colname="binding_label"
sample_sizes=(0)
chosen_metric="cosine"
max_parallel_jobs=5

input_metadata="${REPO_ROOT}/data/sequences/bcr/brian_hie/cr9114_hie_metadata.csv"
LD_matrix="${DATA}/results/ld/LD_mat_brian_hie.npy"
result_dir_base="${RESULTS}/brian_hie_density2"

get_paths() {
  local model=$1 _chain=$2 layer=$3
  local emb_root="${EMBEDDINGS}/brian_hie/${model}"
  input_idx="${EMBEDDINGS}/brian_hie/cr9114_hie_idx.csv"
  input_embeddings="${emb_root}/${complexity}/brian_hie_${model}_${complexity}_layer_$((layer + 1)).npy"
}

run_vicinity_batch
