#!/bin/bash
# Trastuzumab (TZ) vicinity batch driver (Figure 4, Supplementary S8).
# Activate the airr_atlas conda env before running.
# Precompute LD matrix → data/results/ld/tz_LD_dist_mat_HB_LB.npy

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/driver_env.sh"

chains=("cdr3_only" "ALL_CDRH" "heavy_chain" "paired_chain")
complexities=(
  "embeddings"
  "attention_matrices_average_layers"
  "embeddings_unpooled"
  "cdr3_attention_matrices_average_layers"
  "cdr3_extracted"
  "cdr3_extracted_unpooled"
)
models_s=("antiberta2-cssp" "esm2_t33_650M_UR50D")
layers=$(seq 0 32)

df_junction_colname="cdr3_aa"
df_affinity_colname="binding_label"
sample_sizes=(0)
chosen_metric="cosine"
max_parallel_jobs=4

input_metadata="${DATA}/metadata/tz_heavy_chains_airr_dedup_final.tsv.gz"
LD_matrix="${DATA}/results/ld/tz_LD_dist_mat_HB_LB.npy"
result_dir_base="${RESULTS}/tz_100k_density2"

_tz_tag() {
  case "$1" in
    cdr3_only) echo "cdr3_100k" ;;
    heavy_chain) echo "heavy_chain_100k" ;;
    ALL_CDRH) echo "ALL_CDRH" ;;
    paired_chain) echo "paired_chain_100k" ;;
    *) echo "$1" ;;
  esac
}

get_paths() {
  local model=$1 chain=$2 layer=$3
  local tag
  tag="$(_tz_tag "$chain")"
  local emb_root="${EMBEDDINGS}/trastuzumab_npy/${model}"

  if [[ "$chain" == "ALL_CDRH" ]]; then
    input_idx="${emb_root}/tz_ALL_CDRH_100k_${model}_idx.csv"
    input_embeddings="${emb_root}/${complexity}/tz_ALL_CDRH_${model}_${complexity}_layer_$((layer + 1)).npy"
  else
    input_idx="${emb_root}/tz_${tag}_idx.csv"
    input_embeddings="${emb_root}/${complexity}/tz_${tag}_${model}_${complexity}_layer_$((layer + 1)).npy"
  fi
}

run_vicinity_batch
