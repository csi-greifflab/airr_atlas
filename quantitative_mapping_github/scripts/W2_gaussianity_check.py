import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from geomloss import SamplesLoss
import argparse
import time
import csv

# Try to import POT for Sliced Wasserstein
try:
    import ot
except ImportError:
    ot = None

# Import W2 calculator explicitly
from get_W2 import calculate_frechet_distance

# -----------------------------
# Model-specific embedding paths (for Experiment 1)
# -----------------------------
MODEL_PATHS = {
    'esm2': {
        'ag':   '/scratch/niccoloc/ag_dataset/esm2_t33_650M_UR50D/embeddings/ag_dataset_esm2_t33_650M_UR50D_embeddings_layer_33.npy',
        'irec': '/doctorai/niccoloc/ireceptor/esm2_t33_650M_UR50D/mean_pooled/ireceptor_2M_esm2_t33_650M_UR50D_mean_pooled_layer_33.npy',
        'post': '/doctorai/niccoloc/postselection_1M/esm2_t33_650M_UR50D/mean_pooled/postselection_1M_esm2_t33_650M_UR50D_mean_pooled_layer_33.npy',
    },
    'ab2': {
        'ag':   '/scratch/niccoloc/ag_dataset/antiberta2-cssp/embeddings/ag_dataset_antiberta2-cssp_embeddings_layer_16.npy',
        'irec': '/doctorai/niccoloc/ireceptor/antiberta2-cssp/mean_pooled/ireceptor_2M_antiberta2-cssp_mean_pooled_layer_16.npy',
        'post': '/doctorai/niccoloc/postselection_1M/antiberta2-cssp/mean_pooled/postselection_1M_antiberta2-cssp_mean_pooled_layer_16.npy',
    },
    'ohe': {
        'ag':   '/doctorai/niccoloc/w2_AG_DATA_OHE/w2_AG_DATA_OHE_OHE.pt',
        'irec': '/doctorai/niccoloc/w2_irecep_OHE/w2_irecep_OHE_OHE.pt',
        'post': '/doctorai/niccoloc/w2_POSTSEL_OHE/w2_POSTSEL_OHE_OHE.pt',
    }
}

def load_embeddings(path):
    if path is None:
        return None
    if path.endswith('.npy'):
        # using mmap initially
        return np.load(path, mmap_mode='r')
    if path.endswith('.pt'):
        t = torch.load(path, map_location='cpu')
        return t.numpy() if hasattr(t, 'numpy') else t.detach().cpu().numpy()
    raise ValueError(f"Unsupported embedding file type: {path}")

def main():
    parser = argparse.ArgumentParser(description='Gaussianity check using W2, Sinkhorn, and Sliced Wasserstein.')
    parser.add_argument('--method', type=str, choices=['sinkhorn', 'sliced', 'both'], default='both',
                        help='OT method to use: sinkhorn (geomloss), sliced (POT), or both')
    parser.add_argument('--emb_type', type=str, default='esm2', choices=['esm2', 'ab2', 'ohe'], help='Embedding type')
    parser.add_argument('--dataset', type=str, default='irec', choices=['ag', 'irec', 'post'], help='Dataset choice (for Exp 1)')
    parser.add_argument('--n_ref', type=int, default=300000, help='Number of reference samples from data1')
    parser.add_argument('--n_reps', type=int, default=10, help='Number of repetitions')
    parser.add_argument('--experiment', type=str, nargs='+', default=['1'], 
                        help='Experiment(s) to run: 1 (Split Half), 2 (Intra/Inter Patient), 3 (Tech Replicates), or "all"')
    args = parser.parse_args()

    # Handle 'all' experiment selection
    if 'all' in args.experiment:
        experiments_to_run = [1, 2, 3]
    else:
        try:
            experiments_to_run = [int(e) for e in args.experiment]
        except ValueError:
            print("Error: Experiment IDs must be integers or 'all'.")
            return

    if (args.method == 'sliced' or args.method == 'both') and ot is None:
        print("Error: POT (ot) is not installed. Please install it with 'pip install POT'.")
        return

    os.makedirs('/doctorai/niccoloc/airr_atlas/quantitative_mapping_github/data/results', exist_ok=True)
    if len(experiments_to_run) > 1:
        log_file = f'/doctorai/niccoloc/airr_atlas/quantitative_mapping_github/data/results/experiments_combined_log_{args.emb_type}.txt'
    else:
        log_file = f'/doctorai/niccoloc/airr_atlas/quantitative_mapping_github/data/results/experiment_{experiments_to_run[0]}_log_{args.emb_type}.txt'

    # Pre-load and prepare all selected experiments
    print(f"Pre-loading data and preparing {len(experiments_to_run)} experiment(s)...")
    prepared_exps = {}

    #print all the variables and args
    print(f"Arguments: {args}")
    print(f"Experiments to run: {experiments_to_run}")
    print(f"Log file: {log_file}")
    print(f"Method: {args.method}")
    print(f"Embedding type: {args.emb_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of reference samples: {args.n_ref}")
    print(f"Number of repetitions: {args.n_reps}")
    
    for exp_id in experiments_to_run:
        print(f"  Preparing Experiment {exp_id}...")
        if exp_id == 1:
            name = "Exp1_Split_Half"
            path = MODEL_PATHS[args.emb_type][args.dataset]
            data = load_embeddings(path)
            half_len = len(data) // 2
            data1 = np.array(data[:half_len])
            data2 = np.array(data[half_len + 1:])
        elif exp_id == 2:
            name = "Exp2_Intra_Inter_Patient"
            path1 = f"/doctorai/marinafr/2023/airr_atlas/analysis/output/{args.emb_type}/patients/briney_326650.pt"
            path2 = f"/doctorai/marinafr/2023/airr_atlas/analysis/output/{args.emb_type}/patients/briney_326651.pt"
            data1_raw = load_embeddings(path1)
            data2_raw = load_embeddings(path2)
            data1 = np.array(data1_raw[:len(data1_raw) // 2])
            data2 = np.array(data2_raw[:len(data2_raw) // 2])
        elif exp_id == 3:
            name = "Exp3_Tech_Replicates"
            path1 = f"/doctorai/marinafr/2023/airr_atlas/analysis/output/{args.emb_type}/patients/briney_326651_tech_1.pt"
            path2 = f"/doctorai/marinafr/2023/airr_atlas/analysis/output/{args.emb_type}/patients/briney_326651_tech_4.pt"
            data1_raw = load_embeddings(path1)
            data2_raw = load_embeddings(path2)
            data1 = np.array(data1_raw[:len(data1_raw) // 2])
            data2 = np.array(data2_raw[:len(data2_raw) // 2])
        else:
            continue

        # Pre-compute Gaussian stats
        mu1 = np.mean(data1, axis=0)
        sigma1 = np.cov(data1.T)

        # Prepare Sinkhorn loss
        data1_torch = None
        sinkhorn_loss = None
        if args.method in ['sinkhorn', 'both']:
            actual_n_ref = min(len(data1), args.n_ref)
            data1_torch = torch.tensor(data1[:actual_n_ref], dtype=torch.float32).cuda()
            sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.1)

        prepared_exps[exp_id] = {
            'name': name,
            'data1': data1,
            'data2': data2,
            'mu1': mu1,
            'sigma1': sigma1,
            'data1_torch': data1_torch,
            'sinkhorn_loss': sinkhorn_loss
        }

    splits = [int(x) for x in [10**4, 3*10**4, 5*10**4, 10**5, 1.5*10**5, 2*10**5, 3*10**5, 4*10**5-100]]
    
    # Track results for individual CSV saves at the end
    results_tracker = {exp_id: {'w2': [], 'sinkhorn': [], 'sliced': []} for exp_id in experiments_to_run}

    print(f"\nResults will be streamed to: {log_file}")
    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f_log:
        log_writer = csv.writer(f_log)
        if not file_exists or os.path.getsize(log_file) == 0:
            log_writer.writerow(['Experiment', 'Repetition', 'BatchSize', 'W2_Gaussian', 'Sinkhorn', 'Sliced', 'W2_Time', 'Sinkhorn_Time', 'Sliced_Time'])

        # Reformatted loop: Outermost is Repetitions, then Batch Size, then Experiment
        # This allows the plot to become more robust over time as repetitions complete
        for repetition in range(args.n_reps):
            print(f"\n{'='*60}\nREPETITION {repetition+1}/{args.n_reps}\n{'='*60}")
            
            for batch_size in splits:
                print(f"\n--- Batch Size: {batch_size} ---")
                
                for exp_id in experiments_to_run:
                    exp = prepared_exps[exp_id]
                    max_start = len(exp['data2']) - batch_size
                    if max_start <= 0:
                        print(f"[{exp['name']}] Batch size {batch_size} exceeds data2 length. Skipping.")
                        continue
                    
                    # Randomly select a batch
                    start = np.random.randint(0, max_start)
                    batch = exp['data2'][start:start + batch_size]
                    
                    # --- W2 computation (Gaussian) ---
                    t0 = time.time()
                    mu2 = np.mean(batch, axis=0)
                    sigma2 = np.cov(batch.T)
                    w2_val = float(calculate_frechet_distance(exp['mu1'], exp['sigma1'], mu2, sigma2))
                    w2_time = time.time() - t0

                    sinkhorn_val = np.nan
                    sinkhorn_time = 0.0
                    sliced_val = np.nan
                    sliced_time = 0.0

                    status_str = f"[{exp['name']:<25}] Size: {batch_size:<7} | W2: {w2_val:<8.4f} ({w2_time:.2f}s)"

                    # --- Sinkhorn computation ---
                    if args.method in ['sinkhorn', 'both']:
                        t_s = time.time()
                        batch_torch = torch.tensor(batch, dtype=torch.float32).cuda()
                        try:
                            with torch.no_grad():
                                s_out = exp['sinkhorn_loss'](exp['data1_torch'], batch_torch)
                            sinkhorn_val = s_out.item()
                        except Exception as e:
                            print(f"Sinkhorn calculation failed for {exp['name']}: {e}")
                        
                        batch_torch = None
                        torch.cuda.empty_cache()
                        sinkhorn_time = time.time() - t_s
                        status_str += f" | Sinkhorn: {sinkhorn_val:<8.4f} ({sinkhorn_time:.2f}s)"
                    
                    # --- Sliced Wasserstein computation ---
                    if args.method in ['sliced', 'both']:
                        t_sl = time.time()
                        try:
                            actual_n_ref = min(len(exp['data1']), args.n_ref)
                            sliced_val = ot.sliced_wasserstein_distance(exp['data1'][:actual_n_ref], batch, n_projections=100)
                        except Exception as e:
                            print(f"Sliced calculation failed for {exp['name']}: {e}")
                        sliced_time = time.time() - t_sl
                        status_str += f" | Sliced: {sliced_val:<8.4f} ({sliced_time:.2f}s)"
                    
                    print(status_str)
                    
                    # Log to CSV-formatted text file in real-time
                    log_writer.writerow([exp['name'], repetition+1, batch_size, w2_val, sinkhorn_val, sliced_val, w2_time, sinkhorn_time, sliced_time])
                    f_log.flush()
                    
                    # Track for final saves (note: this tracker logic needs to handle the reformatted loop)
                    # We'll re-structure the final save logic to handle this
                    # For now, we'll just save the combined log as the primary output

    print(f"\nAll iterations complete. Combined log saved to {log_file}")

if __name__ == "__main__":
    main()