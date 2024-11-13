import time
import os 
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import esm
import re
from transformers import RoFormerTokenizer, RoFormerModel
from tqdm import tqdm
import joblib
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys

# Function to load models
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "esm2":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model = model.to(device)
        return model, batch_converter, device
    elif model_name == "ab2":
        model_name = "alchemab/antiberta2-cssp"
        tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        model = RoFormerModel.from_pretrained(model_name, output_attentions=True).to(device)
        return model, tokenizer, device
    else:
        raise ValueError("Model not recognized. Choose either 'esm2' or 'ab2'.")

# Function to process sequences and compute attention matrices
def compute_attention_matrices(model_name, sequences, device, batch_converter=None, tokenizer=None):
    attention_matrices = []
    if model_name == "esm2":
        for i, seq in enumerate(tqdm(sequences)):
            batch_labels, batch_strs, batch_tokens = batch_converter([(i, seq)])
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                outputs = model(batch_tokens, return_contacts=True)
                attentions = outputs['attentions']
                attention_maps = attentions[0].cpu().numpy()[:, 1:-1, 1:-1]
                attention_matrices.append(attention_maps)
    elif model_name == "ab2":
        for i, seq in enumerate(tqdm(sequences)):
            inputs = tokenizer(seq, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                attention_maps = outputs.attentions
                attention_matrices.append([attn[0].cpu().numpy()[:, 1:-1, 1:-1] for attn in attention_maps])
    return np.array(attention_matrices)


# Function to load attention matrices or compute them from scratch
def load_or_compute_attention_matrices(file_path,output_dir, sequences, model_name, batch_converter=None, tokenizer=None, device=None):
    try:
        attention_matrices = joblib.load(file_path)
        print(f"Loaded attention matrices from {file_path}.")
    except FileNotFoundError:
        print("Attention matrices not found, computing from scratch.")
        attention_matrices = compute_attention_matrices(model_name, sequences, device, batch_converter, tokenizer)
        joblib.dump(attention_matrices, 'attention_matrices_{model_name}.joblib', compress=5)
    return attention_matrices


# Function to train and evaluate classifiers
def train_and_evaluate(X_train, X_test, y_train, y_test, model_type):
    if model_type == "randomforest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "logistic":
        clf = LogisticRegression(max_iter=6000)
    else:
        raise ValueError("Model type not recognized. Choose either 'randomforest' or 'logistic'.")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return clf, accuracy
  

def plot_attention_heatmaps_all_layers_heads(attention_matrices, sequence_length, save_path=None):
    num_layers = attention_matrices.shape[1]
    num_heads = attention_matrices.shape[2]
    heatmap_data = np.zeros((num_layers, num_heads+1)) #plus 1 beacuse of average perf for Layer
    # Populate the array with the accuracy scores from the results dictionary
    for (layer, head), accuracy in results.items():
        heatmap_data[layer, head] = accuracy
    # Step 2: Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', xticklabels=[f'Head {i+1}' for i in range(num_heads)], 
                yticklabels=[f'Layer {i+1}' for i in range(num_layers)], cbar_kws={'label': 'Accuracy'})
    plt.title('Classifier Accuracy by Attention Head and Layer')
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    plt.savefig('ML_{model_type}_performance_{model}.jpg')




# # Replace 'filename.pkl' with the path to your pickle file
# with open('Attention_ab2_MLmodels.pkl', 'rb') as file:
#     models = pickle.load(file)

# # Replace 'filename.pkl' with the path to your pickle file
# with open('Attention_ab2_MLresults.pkl', 'rb') as file:
#     results = pickle.load(file)



def save_attention_matrices(attention_matrices, model_name, chain , output_dir):
    # """
    # Saves the attention matrices for each layer and head, as well as averages across heads, layers, and all combined.
    # Parameters:
    # - attention_matrices: numpy array of shape (100000, num_layers, num_heads, 15, 15)
    # - model_name: str, name of the model used (e.g., 'esm2', 'ab2')
    # - output_dir: str, directory to save the .pt files
    # Files saved:
    # - attention_matrices_flat_l{i+1}h{j+1}_{model_name}.pt for each layer and head
    # - attention_matrices_flat_avg_l{i+1}_{model_name}.pt for averaged heads in each layer
    # - attention_matrices_flat_avg_ALL_{model_name}.pt for averaged matrices across all layers and heads
    # """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_layers = attention_matrices.shape[1]
    num_heads = attention_matrices.shape[2]

    # Average over all layers and heads
    attention_matrices_head_avg = np.mean(attention_matrices, axis=2)  # Average over heads
    attention_matrices_ALL_avg = np.mean(attention_matrices_head_avg, axis=1)  # Average over layers
    avg_mat_flat_ALL = attention_matrices_ALL_avg.reshape(100000, -1)  # Shape: (100000, 225)
    # Save the averaged attention matrices across all layers and heads
    torch.save(torch.tensor(avg_mat_flat_ALL), os.path.join(output_dir, f'attention_matrices_flat_avg_ALL_{chain}_{model_name}.pt'))

    # Loop through each layer and head
    for i in range(num_layers):  # Loop through layers
        flattened_avg_matrices = attention_matrices_head_avg[:,i,:,:].reshape(100000, -1)  # Shape: (100000, 225)
        # Save averaged attention matrix across heads for this layer
        torch.save(torch.tensor(flattened_avg_matrices), os.path.join(output_dir, f'attention_matrices_flat_avg_l{i+1}_{chain}_{model_name}.pt'))
        # Loop through heads
        for j in range(num_heads):
            # Extract the ith layer and jth head
            layer_head_matrix = attention_matrices[:, i, j, :, :]  # Shape: (100000, 15, 15)
            # Flatten the attention matrix
            flattened_matrices = layer_head_matrix.reshape(100000, -1)  # Shape: (100000, 225)
            
            # Save the flattened attention matrices
            torch.save(torch.tensor(flattened_matrices), os.path.join(output_dir, f'attention_matrices_flat_l{i+1}h{j+1}_{chain}_{model_name}.pt'))

        
  
def plot_accuracy_with_boxplots(results,num_layers,num_heads, avg_results= None,save_path=None):
    output_folder=save_path
    # Initialize heatmap data for layers and heads
    heatmap_data = np.zeros((num_layers, num_heads))
    # Populate the array with the accuracy scores from the results dictionary
    for (layer, head), accuracy in results.items():
        heatmap_data[layer, head] = accuracy
    # Calculate means and standard errors across heads for each layer
    means = np.mean(heatmap_data, axis=1)
    std_errors = np.std(heatmap_data, axis=1) / np.sqrt(heatmap_data.shape[1])
    # Create a list of layer indices (assuming layers start at 0)
    layers = np.arange(heatmap_data.shape[0])
    plt.figure(figsize=(10, 6))
    # Line plot with error bars for the mean accuracy across heads
    plt.errorbar(layers, means, yerr=std_errors, fmt='-o', label="Mean Accuracy per Layer", capsize=5, color='blue')
    # Box plots (transpose heatmap_data to align with layers)
    sns.boxplot(data=heatmap_data.T, whis=1.5, showfliers=False, boxprops=dict(facecolor="lightgray"))
    # Add scatter plot (stripplot) with jitter and alpha for individual data points
    sns.stripplot(data=heatmap_data.T, jitter=0.2, size=4.5, color='black', alpha=0.7)
    # Plot the line for AveragedHead model accuracy
    avg_accuracies = [avg_results[layer] for layer in range(num_layers)]
    plt.plot(layers, avg_accuracies, '-o', label="AveragedHead model accuracy", color='green', linewidth=2)
    # Labeling and titles
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy per Layer with Standard Error, Box Plots, and AveragedHead Accuracy {args.model}_{args.chain}")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'figures/{args.train_model}_{args.model}_{args.chain}_accuracy_boxplot.png')  )



sys.argv = [
    'test.py',  # Script name (first argument in sys.argv)
    '--model', 'esm2',
    '--chain', 'cdr3_only',
    '--train_model' , 'logistic',
    '--train_embeddings',
    'embeddings_folder', '/doctorai/niccoloc/
]



sys.argv = [
    'test.py',  # Script name (first argument in sys.argv)
    '--model', 'esm2',
    '--chain', 'cdr3_only',
    '--train_model' , 'logistic',
    '--train_embeddings',
    'embeddings_folder', '/doctorai/niccoloc/
]

sys.argv = [
    'test.py',  # Script name (first argument in sys.argv)
    '--model', 'ab2',
    '--chain', 'cdr3_only',
    '--preprocess_matrix' , 
]



# Main function
def main():

parser = argparse.ArgumentParser(description="Train a model using attention matrices.")
parser.add_argument("--model", type=str, default="esm2",required=True , help="Choose model: 'esm2' or 'ab2'.")
parser.add_argument("--chain", type=str, required=True , help="Choose input type: cdr3_only , fullseq , paired ecc..; ")
parser.add_argument("--input_file", type=str,default='0',  help="Path to save/load attention matrices.")
parser.add_argument("--train_model", type=str, default="randomforest", help="Choose classifier: 'randomforest' or 'logistic'.")
parser.add_argument("--preprocess_matrix", action="store_true", help="Get the flattened matrices.")
parser.add_argument("--train_all", action="store_true", help="Train for all heads, layers, and average head per layer.")
parser.add_argument("--train_embeddings", action="store_true", help="Train for layers embeddings.")
parser.add_argument("--embeddings_folder", type=str, default="", help="insert the path of the folder conataining the embeeddings ")
parser.add_argument("--output_dir", type=str, default="", help="name of the analysis")
args = parser.parse_args()

# Load metadata and sequences
metadata = pd.read_csv("trastuzumab_metadata.csv")


hb_junction_aa = metadata[metadata["affinity"] == "hb"]["junction_aa"].tolist()[:50000]
lb_junction_aa = metadata[metadata["affinity"] == "lb"]["junction_aa"].tolist()[:50000]
sequences = [" ".join(seq) for seq in hb_junction_aa + lb_junction_aa]
labels = np.array([1] * len(hb_junction_aa) + [0] * len(lb_junction_aa))

output_folder = f"/doctorai/niccoloc/attention_experiment/{args.model}/{args.chain}/"

# Load or compute attention matrices
model, tokenizer_or_batch, device = load_model(args.model)
attention_matrices = load_or_compute_attention_matrices(args.input_file, output_dir=output_folder,  sequences, args.model, tokenizer_or_batch, device)

output_folder = f"./attention_experiment/args.model/{args.chain}/"
if args.preprocess_matrix:
    save_attention_matrices(attention_matrices , args.model, output_dir= output_folder )


if args.train_all:
    results = {}
    models = {}
    avg_result = {}
    avg_model = {}
    for layer in tqdm(range(num_layers)):
        attention_layer = attention_matrices[:, layer, :, :, :]
        attention_matrices_head_avg = np.mean(attention_layer[:, :,  :, :], axis=1)
        flattened_avg_matrices = attention_matrices_head_avg.reshape(100000, -1)  # Shape: (100000, 225)
        X_train, X_test, y_train, y_test = train_test_split(flattened_avg_matrices, labels, test_size=0.2, random_state=42)
        clf, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test, args.train_model)
        avg_model[layer]= clf
        avg_result[layer]= accuracy
        for head in tqdm(range(num_heads)):
            attention_layer_heads = attention_matrices[:, layer, head, :, :]
            attention_layer_flattened = attention_layer_heads.reshape(attention_layer_heads.shape[0], -1)
            X_train, X_test, y_train, y_test = train_test_split(attention_layer_flattened, labels, test_size=0.2, random_state=42)
            clf, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test, args.train_model)
            results[(layer, head)] = accuracy
            models[(layer, head)] = clf  # Store the trained model
            print(f"Layer {layer + 1}, Head {head + 1}: Accuracy = {accuracy:.4f}")
    # Save results and models
    with open(f'{args.train_model}_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    with open(f'{args.train_model}_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Average over all layers and heads
    attention_matrices_head_avg = np.mean(attention_matrices, axis=2)  # Average over heads
    attention_matrices_ALL_avg = np.mean(attention_matrices_head_avg, axis=1)  # Average over layers
    avg_mat_flat_ALL = attention_matrices_ALL_avg.reshape(100000, -1)  # Shape: (100000, 225)
    X_train, X_test, y_train, y_test = train_test_split(avg_mat_flat_ALL, labels, test_size=0.2, random_state=42)
    clf_ALL, accuracy_ALL = train_and_evaluate(X_train, X_test, y_train, y_test, args.train_model)
    print(f"Accuracy (Averaged Heads): {accuracy_avg:.4f}")

    plot_accuracy_with_boxplots(results,avg_result, num_layers,num_heads, avg_results, save_path=output_dir)

    if args.train_embeddings:
        avg_model, avg_results = train_embeddings(args.embeddings_folder , args.model , ML_model =  args.train_model )

        avg_model_rf, avg_results_rf = train_embeddings(args.embeddings_folder , args.model , ML_model =  'randomforest' )

        avg_model_rf_ab2, avg_results_rf_ab2 = train_embeddings(args.embeddings_folder , 'ab2' , ML_model =  'randomforest' )

avg_model_log_ab2, avg_results_log_ab2 = train_embeddings(args.embeddings_folder , 'ab2' , ML_model =  'logistic' )

# Assuming more dictionaries
dicts_with_labels = [
    (avg_results, 'esm2_logistic'),
    (avg_results_rf, 'esm2_randForest'),
    (avg_results_rf_ab2, 'ab2_logistic') , # You can keep adding more
    (avg_results_log_ab2, 'ab2_randForest')  # You can keep adding more
]


prefix = 'Embeddings_ML_'
# Loop through each dictionary and label, saving them as pickle files
for data, label in dicts_with_labels:
    filename = f"{prefix}{label}.pkl"  # Construct the filename
    with open(filename, 'wb') as file:  # Open a file in write-binary mode
        pickle.dump(data, file)  # Save the object as a pickle
    print(f"Saved {label} as {filename}")

import matplotlib.cm as cm

def plot_accuracy(dicts_with_labels, name_fig):
    plt.figure(figsize=(10, 6))
    # Find the maximum x value across all dictionaries
    max_x = 0
    for accuracy_dict, label in dicts_with_labels:
        max_x = max(max_x, max(accuracy_dict.keys()))
    # Extract unique names before '_' in the labels
    unique_names = {label.split('_')[0] for _, label in dicts_with_labels}
    color_map = cm.get_cmap('rainbow', len(unique_names))  # For another distinct uniform colormap
    name_to_color = {name: color_map(i) for i, name in enumerate(unique_names)}
    # Plot each dictionary with appropriate markers and colors
    for accuracy_dict, label in dicts_with_labels:
        x = [key + 1 for key in accuracy_dict.keys()]  # i+1 for X values
        y = list(accuracy_dict.values())  # Accuracy values
        # Determine the marker based on whether 'logistic' or 'forest' is in the label
        if 'logistic' in label.lower():
            line_style = '-'  # Circle for Logistic Regression
        elif 'forest' in label.lower():
            line_style = '--'  # Circle for Logistic Regression
        else:
            line_style = '-.'  # Circle for Logistic Regression
        # Determine the color based on the part before '_'
        name = label.split('_')[0]
        color = name_to_color[name]
        # Plot the data with specific markers and colors
        # plt.plot(x, y, marker=marker, color=color, label=label)
        plt.plot(x, y, linestyle=line_style, color=color, marker='o', label=label)
    plt.grid(True)
        # Add title and labels
    plt.title("ML model accuracy trained on Embeddings cdr3 only 100k")
    plt.xlabel("PLMs layers")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, max_x + 2))
    plt.legend()
    plt.grid(True)
    # Display the plot
    plt.savefig(name_fig)

 
plot_accuracy(dicts_with_labels, 'embeddings_ML_performance_tz_100k_cdr3_only.jpg')


def plot_accuracy_with_boxplots(results,num_layers,num_heads, avg_results= None,save_path=None):
    output_folder=save_path
    # Initialize heatmap data for layers and heads
    heatmap_data = np.zeros((num_layers, num_heads))
    # Populate the array with the accuracy scores from the results dictionary
    for (layer, head), accuracy in results.items():
        heatmap_data[layer, head] = accuracy
    # Calculate means and standard errors across heads for each layer
    means = np.mean(heatmap_data, axis=1)
    std_errors = np.std(heatmap_data, axis=1) / np.sqrt(heatmap_data.shape[1])
    # Create a list of layer indices (assuming layers start at 0)
    layers = np.arange(heatmap_data.shape[0])
    plt.figure(figsize=(10, 6))
    # Line plot with error bars for the mean accuracy across heads
    plt.errorbar(layers, means, yerr=std_errors, fmt='-o', label="Mean Accuracy per Layer", capsize=5, color='blue')
    # Box plots (transpose heatmap_data to align with layers)
    sns.boxplot(data=heatmap_data.T, whis=1.5, showfliers=False, boxprops=dict(facecolor="lightgray"))
    # Add scatter plot (stripplot) with jitter and alpha for individual data points
    sns.stripplot(data=heatmap_data.T, jitter=0.2, size=4.5, color='black', alpha=0.7)
    # Plot the line for AveragedHead model accuracy
    avg_accuracies = [avg_results[layer] for layer in range(num_layers)]
    plt.plot(layers, avg_accuracies, '-o', label="AveragedHead model accuracy", color='green', linewidth=2)
    # Labeling and titles
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy per Layer with Standard Error, Box Plots, and AveragedHead Accuracy {args.model}_{args.chain}")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'figures/{args.train_model}_{args.model}_{args.chain}_accuracy_boxplot.png')  )




plot_accuracy_with_boxplots(avg_results_rf_ab2, 33,1, save_path='/doctorai/niccoloc')


def train_embeddings(embedding_folder,model,ML_model, idx_path =None, metadata_file = None ):
    if model == 'ab2':
        model='antiberta2'
    chain= args.chain
    results = {}
    models = {}
    avg_result = {}
    avg_model = {}
    embedding_folder=f"/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/{model}"
    print(embedding_folder)
    metadata_file =  pd.read_csv('/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv', sep =None) # metadata file with labels and cdr3
    # Get list of all embedding files 
    embedding_files = sorted(os.listdir(f'{embedding_folder}/{chain}'))
    # embedding_files = sorted(os.listdir("/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/antiberta2/cdr3_only"))
    embeddings_tensors = [s for s in embedding_files if re.search("layer", s)]
    num_layers = len(embeddings_tensors)
    idx_df= pd.read_csv( f'{embedding_folder}/100k_sample_trastuzmab_{model}_idx.csv', sep =None) 
    sorted_embeddings = sorted(embeddings_tensors, key=lambda x: int(x.split('_layer_')[1].split('.pt')[0]))
    print(f"check for duplicated ids ... {    idx_df[idx_df['sequence_id'].duplicated()]}")
    # for i,file in enumerate(tqdm(sorted_embeddings)):
    #     # Extract the layer number from the file name (assuming it's part of the name)
    #     layer_idx =i # Example to extract layer number 
    #     # Load the embedding for the current layer
    #     # layer_embeddings = torch.load(os.path.join(embedding_folder, file)).numpy()
    #     layer_embeddings = torch.load(os.path.join(f'{embedding_folder}/{chain}', file)).numpy()
    #     # join idx and tensor 
    #     tensors_df = pd.DataFrame({
    #     'tensor_id': idx_df['index'],
    #     'sequence_id' : idx_df['sequence_id'],
    #     'embedding': list(layer_embeddings)
    #     })
    #     df = pd.merge(tensors_df, metadata_file, on='sequence_id')
    #     df = df.reset_index(drop=True)
    #     # Assuming embeddings are already averaged across heads if applicable
    #     X_train, X_test, y_train, y_test = train_test_split(np.vstack(df['embedding'].values), df['binding_label'], test_size=0.2, random_state=42)
    #     clf, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test,ML_model)
    #     #binding_label_array = df['binding_label'].map({'hb': 1, 'lb': 0}).values
    #     # X_train, X_test, y_train, y_test = train_test_split(layer_embeddings, binding_label_array, test_size=0.2, random_state=42)
    #     # clf, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test,'logistic')
    #     # Save the trained model and accuracy
    #     avg_model[layer_idx] = clf
    #     avg_result[layer_idx] = accuracy
    #     if 0.48 <= accuracy <= 0.52:
    #         print('Accuracy is random , could be label shuffled!!! ')
    #     print(f"Layer {layer_idx + 1}: Accuracy = {accuracy:.4f}")
    # Parallel processing
    start_time = time.time()
    # Parallel processing of batches
    from tqdm_joblib import tqdm_joblib
    with tqdm_joblib(desc="Parallel ML", total=len(sorted_embeddings)) as progress_bar:
        results = Parallel(n_jobs=20)(delayed(process_layer)
        (i, file, idx_df, metadata_file, embedding_folder, chain, ML_model) 
        for i, file in enumerate(sorted_embeddings))
    seq_duration = time.time() - start_time
    print(f"Par execution time: {seq_duration:.2f} seconds")
    # 
    # results = Parallel(n_jobs=20)(delayed(process_layer)(
    #     i, file, idx_df, metadata_file, embedding_folder, chain, ML_model
    # ) for i, file in enumerate(tqdm(sorted_embeddings)))
    # Collect results
    avg_model = {}
    avg_result = {}
    for layer_idx, clf, accuracy in results:
        avg_model[layer_idx] = clf
        avg_result[layer_idx] = accuracy
    
    return avg_model, avg_result
    


from joblib import Parallel, delayed

# Define a function for processing each layer
def process_layer(i, file, idx_df, metadata_file, embedding_folder, chain, ML_model):
    # Extract the layer number from the file name (assuming it's part of the name)
    layer_idx = i
    # Load the embedding for the current layer
    layer_embeddings = torch.load(os.path.join(f'{embedding_folder}/{chain}', file)).numpy()
    # Join idx and tensor
    tensors_df = pd.DataFrame({
        'tensor_id': idx_df['index'],
        'sequence_id': idx_df['sequence_id'],
        'embedding': list(layer_embeddings)
    })
    # Merge with metadata
    df = pd.merge(tensors_df, metadata_file, on='sequence_id')
    df = df.reset_index(drop=True)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(np.vstack(df['embedding'].values), df['binding_label'], test_size=0.2, random_state=42)
    # Train the classifier and evaluate accuracy
    clf, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test, ML_model)
    if 0.48 <= accuracy <= 0.52:
      print(f'Layer {i+1} Accuracy is random, could be label shuffled!')
    print(f"Layer {layer_idx + 1}: Accuracy = {accuracy:.4f}")
    # Return model and accuracy
    return layer_idx, clf, accuracy


# Parallel processing
results = Parallel(n_jobs=20)(delayed(process_layer)(
    i, file, idx_df, metadata_file, embedding_folder, chain, ML_model
) for i, file in enumerate(tqdm(sorted_embeddings)))

# Collect results
avg_model = {}
avg_result = {}
for layer_idx, clf, accuracy in results:
    avg_model[layer_idx] = clf
    avg_result[layer_idx] = accuracy





if __name__ == "__main__":
    main()




