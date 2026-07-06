#!/bin/bash
# Shared environment for Figure 4 Vicinity batch drivers.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/driver_env.sh"

FIG4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${AIRR_ATLAS_ROOT:-$(cd "${FIG4_ROOT}/../../.." && pwd)}"
SCRIPTS="${FIG4_ROOT}/scripts"
DATA="${FIG4_ROOT}/data"
EMBEDDINGS="${REPO_ROOT}/data/embeddings"
RESULTS="${DATA}/results/vicinity"
PIPELINE="${SCRIPTS}/Vicinity_pipeline_final.py"

mkdir -p "${RESULTS}" "${DATA}/results/ld"

# Run Vicinity_pipeline_final.py over model/chain/layer/complexity combinations.
# Requires get_paths() to be defined by the calling driver.
# Optional: define should_skip_run model chain layer complexity sample_size (return 0 to skip).
run_vicinity_batch() {
  local current_jobs=0

  for sample_size in "${sample_sizes[@]}"; do
    for layer in $layers; do
      for model in "${models_s[@]}"; do
        for complexity in "${complexities[@]}"; do
          for chain in "${chains[@]}"; do
            if [[ "$model" == "antiberta2-cssp" && $layer -ge 16 ]]; then continue; fi
            if [[ "$model" == "ab2" && $layer -ge 16 ]]; then continue; fi

            if declare -f should_skip_run >/dev/null 2>&1; then
              should_skip_run "$model" "$chain" "$layer" "$complexity" "$sample_size" && continue
            fi

            get_paths "$model" "$chain" "$layer"

            if [[ ! -f "$input_embeddings" ]]; then
              echo "Skipping missing embeddings: $input_embeddings"
              continue
            fi

            if [[ $sample_size == 0 ]]; then
              result_dir="${result_dir_base}"
            else
              result_dir="${result_dir_base}_sample${sample_size}"
            fi

            if [[ "$complexity" == "cdr3_extracted_unpooled" || "$complexity" == "embeddings_unpooled" ]]; then
              prefix="Unpooled"
            elif [[ "$complexity" == "attention_matrices_average_layers" || "$complexity" == "cdr3_attention_matrices_average_layers" ]]; then
              prefix="AttentionMat"
            elif [[ "$complexity" == "embeddings" || "$complexity" == "cdr3_extracted" ]]; then
              prefix="Pooled"
            fi

            if [[ $sample_size == 0 ]]; then
              analysis_name="${prefix}_${model}_${chain}_layer_${layer}"
            else
              analysis_name="${prefix}_${model}_${chain}_${complexity}_sample_${sample_size}_layer_${layer}"
            fi

            echo "Running: ${analysis_name}"
            python -u "${PIPELINE}" \
              --analysis_name "$analysis_name" \
              --input_metadata "$input_metadata" \
              --input_embeddings "$input_embeddings" \
              --input_idx "$input_idx" \
              --chosen_metric "$chosen_metric" \
              --skip_knn \
              --LD_matrix "$LD_matrix" \
              --result_dir "$result_dir" \
              --df_junction_colname "$df_junction_colname" \
              --df_affinity_colname "$df_affinity_colname" \
              --sample_size "$sample_size" &

            (( current_jobs++ ))
            if (( current_jobs >= max_parallel_jobs )); then
              wait -n
              (( current_jobs-- ))
            fi
          done
        done
      done
    done
  done

  wait
  echo "All tasks completed"
}
