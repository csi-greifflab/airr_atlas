#!/bin/bash

# Define the parameters for each combination
models=("antiberta2" "esm2")
models_s=("ab2" "esm2")
models_s=("esm2")
chains=("paired_chain")
chains=("heavy_chain")
# chains=("full_chain" "cdr3_pooled" "paired_chain" "all_cdrh")
layers=$(seq 0 33)  # Bash doesn't have range, using seq to create a range from 0 to 32 (for layers 0 to 32)

# Input metadata and precomputed LD
# Input metadata and precomputed LD
input_metadata="/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv"
precomputed_LD="/doctorai/niccoloc/Vicinity_results_100k/WHOLE_LD/LD_WHOLE_hb_lb_530k.csv"
result_dir="/doctorai/niccoloc/Vicinity_results_100k_2"


# Other parameters
save_results="True"
plot_results="True"
df_junction_colname="cdr3_aa"
df_affinity_colname="binding_label"
sample_size=0
LD_sample_size=10000
chosen_metric='cosine'

# Set max number of parallel jobs
max_parallel_jobs=3
current_jobs=0

#layers will be 1, 6, 16,33 
# layers=(1 6 16 33)

# Iterate over each combination of model, chain, and layer to check if the file exists
for model in "${models_s[@]}"; do
  for chain in "${chains[@]}"; do
    for layer in $layers; do
      input_idx="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/tz_${chain}_100k_${model}_idx.csv"
      input_idx="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/tz_${chain}_100k_idx.csv"
      # input_embeddings="/doctorai/niccoloc/attention_experiment/${model}/${chain}/attention_matrices_flat_avg_l$(layer+1)_${chain}_${model}.pt"
      # input_embeddings="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/${model}/unpooled/100k_sample_trastuzumab_${chain}_${model}_layer_$((layer))_flat.pt"
      input_embeddings="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/${model}/embeddings_unpooled/tz_${chain}_100k_${model}_${models_s}_embeddings_unpooled_layer_$((layer+1)).pt"
      input_embeddings="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/${model}/embeddings_unpooled/tz_${chain}_100k_${model}_embeddings_unpooled_layer_$((layer+1)).pt"
                        
      #check if file exists
      # if [[ "$model" == "antiberta2" && $layer -ge 17 ]]; then continue; fi
      if [[ "$model" == "ab2" && $layer -ge 17 ]]; then continue; fi
      if [ ! -f "$input_embeddings" ]; then
        echo "File $input_embeddings does not exist"
      fi
    done
  done
done


#wait 7 sec
sleep 7


# Iterate over each combination of model, chain, and layer
for model in "${models_s[@]}"; do
  for chain in "${chains[@]}"; do
    for layer in $layers; do
      # Construct input embeddings and index file paths
      # input_idx="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/${model}/100k_sample_trastuzmab_${model}_idx.csv"
      input_idx="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/tz_${chain}_100k_${model}_idx.csv"
      input_idx="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/tz_${chain}_100k_idx.csv"

      # input_embeddings="/doctorai/niccoloc/attention_experiment/${model}/${chain}/attention_matrices_flat_avg_l$(layer+1)_${chain}_${model}.pt"
      # input_embeddings="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/${model}/unpooled/100k_sample_trastuzumab_${chain}_${model}_layer_$((layer))_flat.pt"
      input_embeddings="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/${models_s}/embeddings_unpooled/tz_${chain}_100k_${models}_${models_s}_embeddings_unpooled_layer_$((layer+1)).pt"
      input_embeddings="/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/${model}/embeddings_unpooled/tz_${chain}_100k_${model}_embeddings_unpooled_layer_$((layer+1)).pt"

      # ab2 layer 17 and above are not available
      # if [[ "$model" == "antiberta2" && $layer -ge 17 ]]; then continue; fi
      if [[ "$model" == "ab2" && $layer -ge 17 ]]; then continue; fi
  
      # Construct the command
      command=(
        "python" "Vicinity_pipeline.py"
        "--analysis_name" "Unpooled_${model}_${chain}_layer_${layer}"
        "--input_metadata" "$input_metadata"
        "--input_embeddings" "$input_embeddings"
        "--input_idx" "$input_idx"
        "--chosen_metric" "$chosen_metric"
        # "--save_results" 
        "--skip_knn"
        "--precomputed_LD" "$precomputed_LD"
        "--result_dir" "$result_dir"
        "--plot_results" 
        "--df_junction_colname" "$df_junction_colname"
        "--df_affinity_colname" "$df_affinity_colname"
        "--sample_size" "$sample_size"
      )
      
      # Run the command in the background
      echo "Running: ${command[*]}"
      "${command[@]}" &

      #Increment job count
      current_jobs=$((current_jobs + 1))

      # If max_parallel_jobs is reached, wait for one to finish
      if (( current_jobs >= max_parallel_jobs )); then
        wait -n  # Wait for any background job to finish before continuing
        current_jobs=$((current_jobs - 1))
      fi
    done
  done
done

# Wait for all background jobs to finish
wait
echo "All tasks completed"


