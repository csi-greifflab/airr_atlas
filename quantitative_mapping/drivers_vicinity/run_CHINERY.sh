#!/bin/bash
# Set working directory
cd /doctorai/niccoloc/airr_atlas/scripts/Vicinity_code || {
    echo "Error: Unable to change to the working directory."
    exit 1
}

source /doctorai/marinafr/progs/miniconda3/envs/airr_atlas/bin/activate /doctorai/marinafr/progs/miniconda3/envs/airr_atlas

# Initialize job counter
current_jobs=0

chains=( "cdr3_only" "ALL_CDRH" "heavy_chain"  "paired_chain")
complexities=("embeddings" "attention_matrices_average_layers"  "embeddings_unpooled" "cdr3_attention_matrices_average_layers" "cdr3_extracted" "cdr3_extracted_unpooled"  )
layers=$(seq 0 32)  # Using seq to create a range from 0 to 32 (for layers 0 to 32)
models_s=( "antiberta2-cssp" "esm2_t33_650M_UR50D"   )
# Other parameters
save_results="True"
plot_results="True"
df_junction_colname="cdr3_aa"
df_affinity_colname="binding_label"

sample_sizes=( 25000 10000 5000 2500 1000 500) #  sampling sizes for sub-sampling tz vicinity experiment  
sample_sizes=( 0 )  # no sampling for full tz vicinity experiment

# sample_size=25000 # 0 means no sampling
chosen_metric='cosine'


# Input metadata and precomputed LD
input_metadata="/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv"
LD_matrix="/doctorai/niccoloc/tz_LD_dist_mat_HB_LB.npy"

# Construct folder name
if [[ $sample_size == 0 ]]; then
  result_dir="/doctorai/niccoloc/Vicinity_results_100k_4"
else
  result_dir="/doctorai/niccoloc/Vicinity_results_100k_sample${sample_size}"
fi



# Set max number of parallel jobs
max_parallel_jobs=4



# Function to determine the embeddings and index paths
get_paths() {
  local model=$1
  local chain=$2
  local layer=$3
  if [[ "$chain" == "paired_chain" || "$chain" == "ALL_CDRH" ]]; then
    input_idx="/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_npy/${model}/tz_${chain}_100k_${model}_idx.csv"
    input_embeddings="/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_npy/${model}/${complexity}/tz_${chain}_${model}_${complexity}_layer_$((layer+1)).npy"
  else
    input_idx="/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_npy/${model}/tz_${chain}_100k_idx.csv"
    # input_idx="/scratch/niccoloc/tz_heavy/${model}/tz_heavy_chain_100k_idx.csv"
    input_embeddings="/scratch/niccoloc/tz_heavy_v2/${model}/${complexity}/tz_${chain}_${model}_${complexity}_layer_$((layer+1)).npy"
  fi
}

# Test if the retrieved embeddings and idx files exist for all combinations
for model in "${models_s[@]}"; do
  for chain in "${chains[@]}"; do
    for layer in $layers; do
      # Skip if model is ab2 and layer is greater than or equal to 17
      if [[ "$model" == "antiberta2-cssp" && $layer -ge 16 ]]; then continue; fi

      # Get the paths
      get_paths "$model" "$chain" "$layer"

      # Test if the paths exist
      if [ ! -f "$input_embeddings" ]; then
        echo "Embeddings file $input_embeddings does not exist"
      fi
      if [ ! -f "$input_idx" ]; then
        echo "Index file $input_idx does not exist"
      fi
    done
  done
done





# Wait for 10 seconds
sleep 10

# Iterate over each combination of model, chain, layer, and sample size
for sample_size in "${sample_sizes[@]}"; do
  for layer in $layers; do
    for model in "${models_s[@]}"; do
      for complexity in "${complexities[@]}"; do
        for chain in "${chains[@]}"; do
          # Skip if model is ab2 and layer is greater than or equal to 17
          if [[ "$model" == "ab2" && $layer -ge 16 ]]; then continue; fi
          # Skip if model is esm2 and layer is smaller than 18
          # if ! [[ "$model" == "antiberta2-cssp" && "$complexity" == "embeddings_unpooled" && $layer -ge 9 && "$sample_size" == "0" ]]; then continue; fi


          # Get the paths
          get_paths "$model" "$chain" "$layer"

          # Check if the input embeddings file exists
          if [ ! -f "$input_embeddings" ]; then
            echo "File $input_embeddings does not exist"
            continue
          fi

          # # Adjust max_parallel_jobs based on chain and complexity
          # if [[ "$chain" == "cdr3_only" ]]; then
          #   max_parallel_jobs=7
          # elif [[ "$complexity" == "cdr3_extracted_unpooled" || "$chain" == "all_cdrh" ]]; then
          #   max_parallel_jobs=5
          # elif [[ "$chain" == "heavy_chain" ]]; then
          #   max_parallel_jobs=3
          # else
          #   max_parallel_jobs=20
          # fi

          # Construct folder name
          if [[ $sample_size == 0 ]]; then
            result_dir="/doctorai/niccoloc/Vicinity_results_100k_Density2"
          else
            result_dir="/doctorai/niccoloc/Vicinity_results_100k_Density2_sample${sample_size}"
          fi
                  # Construct analysis name based on complexity
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

          # Construct the command
          command=(
            "python" "-u" "Vicinity_pipeline.py"
            "--analysis_name" "$analysis_name"
            "--input_metadata" "$input_metadata"
            "--input_embeddings" "$input_embeddings"
            "--input_idx" "$input_idx"
            "--chosen_metric" "$chosen_metric"
            "--skip_knn"
            "--LD_matrix"  "$LD_matrix"
            "--result_dir" "$result_dir"
            # "--plot_results"
            "--df_junction_colname" "$df_junction_colname"
            "--df_affinity_colname" "$df_affinity_colname"
            "--sample_size" "$sample_size"
          )

          # Run the command in the background
          echo "Running: ${command[*]}"
          "${command[@]}" &

          # Increment job count and manage parallel jobs
          (( current_jobs++ ))
          if (( current_jobs >= max_parallel_jobs )); then
            wait -n  # Wait for any background job to finish before continuing
            (( current_jobs-- ))
          fi
          # Print job progression
          echo "Started job $((current_jobs + 1)) out of $(( ${#models_s[@]} * ${#chains[@]} * ${#layers[@]} * ${#sample_sizes[@]} ))}"
        done
      done
    done
  done
done

# Wait for all background jobs to finish
wait
echo "All tasks completed"

