import os
import argparse
import torch

parser = argparse.ArgumentParser(description="Create UMAP subsampled datasets")
parser.add_argument("directory", type=str, help="Directory containing the .pt file")
parser.add_argument("file_name", type=str, help="Name of the .pt file to subsample")
args = parser.parse_args()

directory = args.directory
file_name = args.file_name

# Output directory for the sampled files
umap_directory = os.path.join(directory, "umap")

# Ensure the umap directory exists
os.makedirs(umap_directory, exist_ok=True)

# Define the number of sequences to sample
sequence_counts = [10000, 30000, 50000, 100000, 150000, 200000, 300000, 399990]

# Function to process the .pt file
def process_pt_file(file_path, file_name):
    # Load the dataset
    data = torch.load(file_path)
    
    # Ensure data is a PyTorch tensor
    if not isinstance(data, torch.Tensor):
        raise ValueError("Expected data to be a PyTorch tensor.")
    
    for count in sequence_counts:
        # Check to ensure we don't sample more sequences than available
        if count > len(data):
            print(f"Skipping {file_name}, requested {count} sequences but only have {len(data)}.")
            continue
        
        # Generate a random permutation of indices
        perm = torch.randperm(len(data))

        # Select the first count indices
        selected_indices = perm[:count]

        # Index the original tensor to get the sampled tensors
        sampled_data = data[selected_indices]

        # Construct the new filename by appending the sequence count
        base_name = os.path.splitext(file_name)[0]  # Removes the `.pt` extension
        new_file_name = f"{base_name}_{count}.pt"
        new_file_path = os.path.join(umap_directory, new_file_name)

        # Save the sampled data
        torch.save(sampled_data, new_file_path)
        print(f"Saved {count} sequences to {new_file_name} in the umap directory.")

# Process the specific file in the directory
file_path = os.path.join(directory, file_name)

process_pt_file(file_path, file_name)

