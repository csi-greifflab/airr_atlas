#!/bin/bash
# AlphaSeq (Engelhart) vicinity batch driver (Figure 4, Supplementary S10).
# Activate the airr_atlas conda env before running.
# Precompute LD matrix → data/results/ld/LD_alphaseq_HB_LB.npy
# Embeddings: ${REPO_ROOT}/data/embeddings/alphaseq44/

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/driver_env.sh"

chains=("paired_chain")
complexities=("embeddings" "attention_matrices_average_layers" "embeddings_unpooled")
models_s=("esm2_t33_650M_UR50D" "antiberta2-cssp")
layers=$(seq 0 32)

df_junction_colname="Sequence"
df_affinity_colname="affinity"
sample_sizes=(0)
chosen_metric="cosine"
max_parallel_jobs=5

input_metadata="${DATA}/metadata/metadata_alphaseq_HB_LB.csv.gz"
LD_matrix="${DATA}/results/ld/LD_alphaseq_HB_LB.npy"
result_dir_base="${RESULTS}/alphaseq_density2_hb_lb"

get_paths() {
  local model=$1 chain=$2 layer=$3
  local emb_root="${EMBEDDINGS}/alphaseq44/${model}"
  input_idx="${emb_root}/alphaseq_paired_chain_${model}_idx.csv"
  input_embeddings="${emb_root}/${complexity}/alphaseq_${model}_${complexity}_layer_$((layer + 1)).npy"
}

should_skip_run() {
  local model=$1 _chain=$2 layer=$3 complexity=$4 _sample_size=$5
  if [[ "$model" != "antiberta2-cssp" && $layer -eq 4 ]]; then return 0; fi
  if [[ "$model" == "antiberta2-cssp" && $layer -eq 4 && "$complexity" != "embeddings_unpooled" ]]; then
    return 0
  fi
  return 1
}

run_vicinity_batch
