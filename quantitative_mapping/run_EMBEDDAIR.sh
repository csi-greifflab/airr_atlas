#!/bin/bash

# ============================================================================
# EMBEDDAIR - Batch Embedding Extraction Script
# ============================================================================
# This script runs embedairr to extract embeddings from various antibody datasets
# using ESM2 and AntiBERTa2 models across all layers.
# ============================================================================

# GPU Configuration
export CUDA_VISIBLE_DEVICES="1"

# Model Configuration
models=(esm2_t33_650M_UR50D alchemab/antiberta2-cssp)
layers=all

# Common embedairr parameters
COMMON_PARAMS="--batch_writing true --flatten true --precision 16"

# ============================================================================
# 1. ANTIGEN DATASET EXPERIMENTS
# ============================================================================
echo "=== Running Antigen Dataset Experiments (tg2,malaria,ebola etc... fig 3) ==="

for model in "${models[@]}"; do
    echo "Processing ag_dataset with model: $model"

    embedairr --experiment_name ag_dataset \
        --model_name "$model" \
        --fasta_path '/doctorai/userdata/airr_atlas/data/sequences/bcr/ALL_ANTIGENS/antigen_specific_2025.fasta' \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/ag_dataset" \
        --batch_size 4192 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
        $COMMON_PARAMS
done

# ============================================================================
# 2. TRASTUZUMAB EXPERIMENTS
# ============================================================================
echo "=== Running Trastuzumab Experiments ==="

# Input file paths
full_input="/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_heavy_chain_100k.fa"
all_cdrh_input_ESM2="/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_all_cdrh_100k_esm2.fa"
all_cdrh_input_AB2="/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_all_cdrh_100k_antiberta2.fa"
cdr3_sequence_file="/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_cdr3.csv"
cdr3_fasta='/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_cdr3_100k.fa'

# 4a. Trastuzumab Heavy Chain
for model in "${models[@]}"; do
    echo "Processing tz_heavy with model: $model"
    embedairr --experiment_name tz_heavy \
        --model_name "$model" \
        --fasta_path "$full_input" \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/tz_heavy24" \
        --batch_size 4000 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
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

    embedairr --experiment_name tz_ALL_CDRH \
        --model_name "$model" \
        --fasta_path "$input_tz" \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/tz_all25" \
        --batch_size 4000 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
        $COMMON_PARAMS
done

# 4c. Trastuzumab CDR3 extracted
for model in "${models[@]}"; do
    echo "Processing tz_cdr3_EXTRACTED with model: $model"
    
    embedairr --experiment_name tz_cdr3 \
        --model_name "$model" \
        --fasta_path "$full_input" \
        --cdr3_path "$cdr3_sequence_file" \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/tz_cdr3_extracted" \
        --batch_size 4192 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
        $COMMON_PARAMS
done
# 4d. Trastuzumab CDR3 only fasta
for model in "${models[@]}"; do
    echo "Processing tz_cdr3_fasta with model: $model"

    embedairr --experiment_name tz_cdr3_fasta \
        --model_name "$model" \
        --fasta_path "$cdr3_fasta" \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/tz_cdr3" \
        --batch_size 4192 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
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

    embedairr --experiment_name tz_paired_chain \
        --model_name "$model" \
        --fasta_path "/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_100k.fasta" \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/tz_paired_chain" \
        --batch_size 4000 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
        $COMMON_PARAMS
done

# ============================================================================
# 3. POREBSKI EXPERIMENTS (CDR3 only)
# ============================================================================
echo "=== Running Porebski Experiments ==="

porebski="/doctorai/userdata/airr_atlas/data/sequences/bcr/porebski/porebski_cdr3_only.fasta"

for model in "${models[@]}"; do
    echo "Processing porebski with model: $model"

    embedairr --experiment_name porebski \
        --model_name "$model" \
        --fasta_path "$porebski" \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/porebski" \
        --batch_size 4192 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
        $COMMON_PARAMS
done

# ============================================================================
# 4. VARUN (BRIAN HIE) EXPERIMENTS (Heavy Chain only)
# ============================================================================
echo "=== Running  VARUN Experiments ==="

brian_hie="/doctorai/userdata/airr_atlas/data/sequences/bcr/brian_hie/cr9114_hie.fasta"

for model in "${models[@]}"; do
    echo "Processing brian_hie with model: $model"

    embedairr --experiment_name brian_hie \
        --model_name "$model" \
        --fasta_path "$brian_hie" \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/brian_hie" \
        --batch_size 4192 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
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

    embedairr --experiment_name alphaseq \
        --model_name "$model" \
        --fasta_path "$input_alphaseq" \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/alphaseq44" \
        --batch_size 4000 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
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

    embedairr --experiment_name covabdab \
        --model_name "$model" \
        --fasta_path "$input_covabdab" \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/covabdab" \
        --batch_size 5000 \
        --extract_embeddings pooled unpooled \
        --extract_attention average_layer \
        $COMMON_PARAMS
done

# ============================================================================
# 7. PREPOST EXPERIMENTS
# ============================================================================
echo "=== Running PrePost Experiments ==="

for model in "${models[@]}"; do
    echo "Processing prepost100k with model: $model"

    embedairr --experiment_name prepost100k \
        --model_name "$model" \
        --fasta_path '/doctorai/niccoloc/prepost_sample_15aa_100k_pgens.fasta' \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/prepost" \
        --batch_size 4192 \
        --extract_embeddings pooled \
        $COMMON_PARAMS
done

# ============================================================================
# 8. IRECEPTOR EXPERIMENTS (Large 3M dataset)
# ============================================================================
echo "=== Running iReceptor Experiments ==="

for model in "${models[@]}"; do
    echo "Processing ireceptor_3M with model: $model"

    embedairr --experiment_name ireceptor_3M_nodup_max35L_pgen \
        --model_name "$model" \
        --fasta_path '/doctorai/niccoloc/ireceptor_3M_nodup_max35L_pgen_2025.fasta' \
        --layers "$layers" \
        --output_path "/scratch/niccoloc/ireceptor" \
        --batch_size 40192 \
        --extract_embeddings pooled \
        --num_workers 16 \
        $COMMON_PARAMS
done

# ============================================================================
# DONE
# ============================================================================
echo "=== All experiments completed ==="