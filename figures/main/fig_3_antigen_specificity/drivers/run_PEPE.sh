#!/bin/bash

# ============================================================================
# PEPE - Batch Embedding Extraction Script
# ============================================================================
# This script runs pepe to extract embeddings from various antibody datasets
# using ESM2 and AntiBERTa2 models across all layers.
# ============================================================================

# GPU Configuration
export CUDA_VISIBLE_DEVICES="1"

# Model Configuration
models=(esm2_t33_650M_UR50D alchemab/antiberta2-cssp)
layers=all

# Common pepe parameters
COMMON_PARAMS="--streaming_output True --flatten True --precision float16"

# Helper to map model names to simple suffixes for file paths
get_model_suffix() {
    if [[ "$1" == "esm2_t33_650M_UR50D" ]]; then
        echo "esm2"
    elif [[ "$1" == "alchemab/antiberta2-cssp" ]]; then
        echo "ab2"
    else
        echo "unknown"
    fi
}

# ============================================================================
# 1. ANTIGEN DATASET EXPERIMENTS
# ============================================================================
echo "=== Running Antigen Dataset Experiments (tg2,malaria,ebola etc... fig 3) ==="

for model in "${models[@]}"; do
    model_suffix=$(get_model_suffix "$model")
    echo "Processing ag_dataset with model: $model"

    pepe --experiment_name ag_dataset \
        --model_name "$model" \
        --fasta_path './data/sequences/antigen_specific_2025.fasta' \
        --layers "$layers" \
        --output_path "./results/ag_dataset" \
        --batch_size 4192 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done

# ============================================================================
# 2. TRASTUZUMAB EXPERIMENTS
# ============================================================================
echo "=== Running Trastuzumab Experiments ==="

# Input file paths
full_input="./data/sequences/tz_heavy_chain_100k_sample.fa" # Note: User mentioned removing heavy chain, but we have 100k sample.
all_cdrh_input_ESM2="./data/sequences/tz_all_cdrh_100k_esm2.fa"
all_cdrh_input_AB2="./data/sequences/tz_all_cdrh_100k_antiberta2.fa"
cdr3_sequence_file="./data/metadata/tz_cdr3_100k_idx.csv" # Adjusted from .csv to _idx.csv if applicable
cdr3_fasta='./data/sequences/tz_cdr3_100k.fa'

# 4a. Trastuzumab Heavy Chain
for model in "${models[@]}"; do
    echo "Processing tz_heavy with model: $model"
    pepe --experiment_name tz_heavy \
        --model_name "$model" \
        --fasta_path "$full_input" \
        --layers "$layers" \
        --output_path "./results/tz_heavy" \
        --batch_size 4000 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done

# 4b. Trastuzumab ALL CDRH
for model in "${models[@]}"; do
    if [[ "$model" == "esm2_t33_650M_UR50D" ]]; then
        input_tz="$all_cdrh_input_ESM2"
    elif [[ "$model" == "alchemab/antiberta2-cssp" ]]; then
        input_tz="$all_cdrh_input_AB2"
    else
        echo "Unknown model: $model"
        continue
    fi

    echo "Processing tz_ALL_CDRH with model: $model"

    pepe --experiment_name tz_ALL_CDRH \
        --model_name "$model" \
        --fasta_path "$input_tz" \
        --layers "$layers" \
        --output_path "./results/tz_all" \
        --batch_size 4000 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done

# 4c. Trastuzumab CDR3 extracted
for model in "${models[@]}"; do
    echo "Processing tz_cdr3_EXTRACTED with model: $model"
    
    pepe --experiment_name tz_cdr3 \
        --model_name "$model" \
        --fasta_path "$full_input" \
        --cdr3_path "$cdr3_sequence_file" \
        --layers "$layers" \
        --output_path "./results/tz_cdr3_extracted" \
        --batch_size 4192 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done
# 4d. Trastuzumab CDR3 only fasta
for model in "${models[@]}"; do
    echo "Processing tz_cdr3_fasta with model: $model"

    pepe --experiment_name tz_cdr3_fasta \
        --model_name "$model" \
        --fasta_path "$cdr3_fasta" \
        --layers "$layers" \
        --output_path "./results/tz_cdr3" \
        --batch_size 4192 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done

# 4e. Trastuzumab paired chain 

for model in "${models[@]}"; do
    echo "Processing tz_paired_chain with model: $model"

    if [[ "$model" == "esm2_t33_650M_UR50D" ]]; then
        input_tz_paired="/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_100k_esm2.fa"
    elif [[ "$model" == "alchemab/antiberta2-cssp" ]]; then
        input_tz_paired="/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_100k_antiberta2.fa"
    else
        echo "Unknown model: $model"
        continue
    fi

    pepe --experiment_name tz_paired_chain \
        --model_name "$model" \
        --fasta_path "./data/sequences/tz_paired_chain_5k_${model_suffix}.fa" \
        --layers "$layers" \
        --output_path "./results/tz_paired_chain" \
        --batch_size 4000 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done

# ============================================================================
# 3. POREBSKI EXPERIMENTS (CDR3 only)
# ============================================================================
echo "=== Running Porebski Experiments ==="

porebski="/doctorai/userdata/airr_atlas/data/sequences/bcr/porebski/porebski_cdr3_only.fasta"

for model in "${models[@]}"; do
    echo "Processing porebski with model: $model"

    pepe --experiment_name porebski \
        --model_name "$model" \
        --fasta_path "./data/sequences/porebski_cdr3_only.fasta" \
        --layers "$layers" \
        --output_path "./results/porebski" \
        --batch_size 4192 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done

# ============================================================================
# 4. VARUN (BRIAN HIE) EXPERIMENTS (Heavy Chain only)
# ============================================================================
echo "=== Running  VARUN Experiments ==="

brian_hie="/doctorai/userdata/airr_atlas/data/sequences/bcr/brian_hie/cr9114_hie.fasta"

for model in "${models[@]}"; do
    echo "Processing brian_hie with model: $model"

    pepe --experiment_name brian_hie \
        --model_name "$model" \
        --fasta_path "./data/sequences/cr9114_hie.fasta" \
        --layers "$layers" \
        --output_path "./results/brian_hie" \
        --batch_size 4192 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done


# ============================================================================
# 5. ALPHASEQ EXPERIMENTS (Paired Chains)
# ============================================================================
echo "=== Running AlphaSeq Experiments ==="

alpha_seq_ESM2="/doctorai/userdata/airr_atlas/data/sequences/bcr/antigens/alphaseq_sars/alphaseq_paired_chain_esm2.fasta"
alpha_seq_AB2="/doctorai/userdata/airr_atlas/data/sequences/bcr/antigens/alphaseq_sars/alphaseq_paired_chain_ab2.fasta"

for model in "${models[@]}"; do
    if [[ "$model" == "esm2_t33_650M_UR50D" ]]; then
        input_alphaseq="$alpha_seq_ESM2"
    elif [[ "$model" == "alchemab/antiberta2-cssp" ]]; then
        input_alphaseq="$alpha_seq_AB2"
    else
        echo "Unknown model: $model"
        continue
    fi

    echo "Processing alphaseq with model: $model"
    echo "Input path: $input_alphaseq"

    pepe --experiment_name alphaseq \
        --model_name "$model" \
        --fasta_path "./data/sequences/alphaseq_paired_chain_${model_suffix}_sample.fasta" \
        --layers "$layers" \
        --output_path "./results/alphaseq" \
        --batch_size 4000 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done


# ============================================================================
# 6. COVABDAB EXPERIMENTS (Paired Chains)
# ============================================================================
echo "=== Running CoVAbDab Experiments ==="

for model in "${models[@]}"; do
    if [[ "$model" == "esm2_t33_650M_UR50D" ]]; then
        input_covabdab="/doctorai/niccoloc/covabdab_bg_ESM2.fasta"
    elif [[ "$model" == "alchemab/antiberta2-cssp" ]]; then
        input_covabdab="/doctorai/userdata/airr_atlas/data/sequences/bcr/antigens/covabdab/covabdab_bg_AB2.fasta"
    else
        echo "Unknown model: $model"
        continue
    fi

    echo "Processing covabdab with model: $model"
    echo "Input path: $input_covabdab"

    pepe --experiment_name covabdab \
        --model_name "$model" \
        --fasta_path "./data/sequences/covabdab_bg_${model_suffix}.fasta" \
        --layers "$layers" \
        --output_path "./results/covabdab" \
        --batch_size 5000 \
        --extract_embeddings mean_pooled per_token attention_layer \
        $COMMON_PARAMS
done

# iReceptor, PrePost experiments removed as per user feedback (too large for GitHub)
# To run these, use the full datasets and update paths accordingly.

# ============================================================================
# DONE
# ============================================================================
echo "=== All experiments completed ==="