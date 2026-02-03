import subprocess
import itertools

# Define the parameters for each combination
models = ['esm2', 'antiberta2']
chains = ['full_chain', 'cdr3_only']
layers = range(0, 33)  # Assuming you want to test only layer 16 for now, expand as needed

# Input metadata and index file paths
input_metadata = "/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv"
input_idx = f"/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/{models}/{chains}/100k_sample_trastuzmab_{chains}_{models}_idx.csv"
precomputed_LD = '/doctorai/niccoloc/Vicinity_results_100k/WHOLE_LD/LD_WHOLE_hb_lb_530k.csv'
result_dir ="/doctorai/niccoloc/Vicinity_results_100k"
# Other parameters
save_results = "True"
compute_LD = "True"
plot_results = "True"
df_junction_colname = "cdr3_aa"
df_affinity_colname = "binding_label"
sample_size = 0
LD_sample_size = 10000

# Iterate over each combination of model, chain, and layer
for model, chain, layer in itertools.product(models, chains, layers):
    # if model == 'antiberta2':
    #   layers=range(0,33)
    # Construct the input embeddings file path
    input_metadata = "/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv"
    input_idx = f"/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/{model}/100k_sample_trastuzmab_{model}_idx.csv"

    input_embeddings = f"/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/{model}/{chain}/100k_sample_trastuzmab_{chain}_layer_{layer+1}.pt"
    
    # Construct the command
    command = [
        "python", "Vicinity_pipeline.py",
        "--analysis_name", f"{model}_{chain}_layer_{layer}",
        "--input_metadata", input_metadata,
        "--input_embeddings", input_embeddings,
        "--input_idx", input_idx,
        "--save_results" ,
        "--precomputed_LD", precomputed_LD,
        "--result_dir" ,result_dir ,
        "--plot_results",
        "--df_junction_colname", df_junction_colname,
        "--df_affinity_colname", df_affinity_colname,
        "--sample_size", str(sample_size),
        "--parallel"
    ]

    # Run the command
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)

print("All tasks completed.")
