"""
This script takes a fasta file as input and outputs a tensor of the mean representations of each sequence in the file using a pre-trained ESM-2 model.
The tensor is saved as a PyTorch file.

Args:
    fasta_path (str): Path to the fasta file.
    output_path (str): Path to save the output tensor.
"""
import os
import csv
import argparse
from Bio import SeqIO
import torch
from transformers import RoFormerTokenizer, RoFormerModel
from torch.utils.data import DataLoader, TensorDataset





#Parsing command-line arguments for input and output file paths
PARSER = argparse.ArgumentParser(description="Input path")
PARSER.add_argument("--fasta_path", type=str, required=True,
                    help="Fasta path + filename.fa")
PARSER.add_argument("--output_path", type=str, required=True,
                    help="Output path + filename.pt \nWill output multiple files if multiple layers are specified with '--layers'.")
PARSER.add_argument("--cdr3_path", default = None, type=str,
                    help="Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.")
PARSER.add_argument("--context", default = 0,type=int,
                    help="Number of amino acids to include before and after CDR3 sequence")
PARSER.add_argument("--layers", type=str, nargs='*', default="-1",
                    help="Representation layers to extract from the model. Default is the last layer. Example: argument '--layers -1 6' will output the last layer and the sixth layer.")
PARSER.add_argument("--pooling", type=bool, nargs='*', default=True,
                    help="Whether to pool the embeddings or not. Default is True.")
args = PARSER.parse_args()

# Storing arguments
FASTA_PATH = args.fasta_path
OUTPUT_PATH = args.output_path
CDR3_PATH = args.cdr3_path
CONTEXT = args.context
LAYERS = list(map(int, args.layers[0].split()))
if args.pooling:
    POOLING = True
else:
    POOLING = False

######## debug
#import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]='expandable_segments:True'
#FASTA_PATH = '/doctorai/userdata/airr_atlas/data/sequences/wang/first_10.fasta'
#OUTPUT_PATH = '/doctorai/userdata/airr_atlas/test/test_cdr3.pt'
#CDR3_PATH = '/doctorai/userdata/airr_atlas/data/sequences/trastuzumab_heavy_chains/trastuzumab_heavy_chains_cdr3.csv'
#CDR3_PATH = None
#CONTEXT = 0
#LAYERS = list(range(1,17))

# convert fasta into dictionary
def fasta_to_dict(fasta_file):
    """
    Convert a fasta file into a dictionary.

    Args:
        fasta_file (str): Path to the fasta file.

    Returns:
        dict: A dictionary where the keys are the sequence IDs and the values are the sequences.
    """
    print('Loading and batching input sequences...')
    seq_dict = {}
    with open(fasta_file) as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq_dict[record.id] = " ".join(str(record.seq)) # AA tokens for hugging face models must be space gapped
            # print progress
            if len(seq_dict) % 1000 == 0:
                print(f'{len(seq_dict)} sequences loaded')
    return seq_dict

# Read sequences from the FASTA file
SEQUENCES = fasta_to_dict(FASTA_PATH)

# Check if output directory exists and creates it if it's missing
if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
    # if the directory is not present create it.
    os.makedirs(os.path.dirname(OUTPUT_PATH))

# Load cdr3 sequences and store in dictionary
if CDR3_PATH:
    with open(CDR3_PATH) as f:
        READER = csv.reader(f)
        CDR3_DICT = {rows[0]:rows[1] for rows in READER}
    # TODO investigate missing_keys
    MISSING_KEYS = [key for key in SEQUENCES if key not in CDR3_DICT]




# Pre-defined parameters for optimization
MODEL_NAME = "alchemab/antiberta2-cssp"
BATCH_SIZE = 512  # Adjust based on your GPU's memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = RoFormerTokenizer.from_pretrained(MODEL_NAME)
MODEL = RoFormerModel.from_pretrained(MODEL_NAME).to(DEVICE)
MODEL.eval()

# Checking if the specified representation layers are valid
assert all(-(MODEL.config.num_hidden_layers + 1) <= i <= MODEL.config.num_hidden_layers for i in LAYERS)
LAYERS = [(i + MODEL.config.num_hidden_layers + 1) % (MODEL.config.num_hidden_layers + 1) for i in LAYERS]

print("Start tokenization")

# Tokenize sequences
INPUT_IDS = []
ATTENTION_MASKS = []
TOTAL_SEQUENCES = len(SEQUENCES)
for counter, sequence in enumerate(SEQUENCES.values()):
    #tokens = tokenizer(sequence,truncation=True, padding='max_length', return_tensors="pt",add_special_tokens=True, max_length=200)
    #tokenize sequences without truncation
    tokens = TOKENIZER(sequence,truncation=False, padding='max_length', return_tensors="pt",add_special_tokens=True, max_length=200)
    #print( tokenizer.decode(tokens['input_ids'][0]))
    INPUT_IDS.append(tokens['input_ids'])
    ATTENTION_MASKS.append(tokens['attention_mask'])
    # Calculate and print the percentage of completion
    percent_complete = ((counter + 1) / TOTAL_SEQUENCES) * 100
    # Check and print the progress at each 2% interval
    if (counter + 1) == TOTAL_SEQUENCES or int(percent_complete) % 2 == 0:
        # Ensures the message is printed once per interval and at 100% completion
        if (counter + 1) == TOTAL_SEQUENCES or (int(percent_complete / 2) != int(((counter) / TOTAL_SEQUENCES) * 100 / 2)):
            print(f"Progress: {percent_complete:.2f}%")
TOKENIZER.decode(INPUT_IDS[0][0])


# Convert lists to tensors and create a dataset
INPUT_IDS = torch.cat(INPUT_IDS, dim=0)
ATTENTION_MASKS = torch.cat(ATTENTION_MASKS, dim=0)
DATASET = TensorDataset(INPUT_IDS, ATTENTION_MASKS)
DATA_LOADER = DataLoader(DATASET, batch_size=BATCH_SIZE)

# Initialize a list to store embeddings
MEAN_REPRESENTATIONS = {layer: [] for layer in LAYERS}
SEQUENCE_LABELS = []
with torch.no_grad():
    TOTAL_BATCHES = len(DATA_LOADER)  # Correctly calculate the total number of batches here

    for batch_idx, batch in enumerate(DATA_LOADER):
        labels = list(SEQUENCES.keys())[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        INPUT_IDS, attention_mask = [b.to(DEVICE, non_blocking=True) for b in batch]
        
        outputs = MODEL(input_ids=INPUT_IDS, attention_mask=attention_mask, output_hidden_states=True)
        # Extracting layer representations and moving them to CPU
        representations = {layer: outputs.hidden_states[layer].to(device="cpu") for layer in LAYERS}
        
        # TODO add optional argument to return mean pooled full embedding even if cdr3_path is specified
        if CDR3_PATH is None:
            # Append labels to SEQUENCE_LABELS
            SEQUENCE_LABELS.extend(labels)
            for layer in LAYERS:
                if POOLING:
                    MEAN_REPRESENTATIONS[layer].extend(
                        representations[layer][i, 1: len(SEQUENCES[label]) + 1].mean(0).clone()
                        for i, label in enumerate(labels)
                    )
                else:
                    MEAN_REPRESENTATIONS[layer].extend(
                        representations[layer][i, 1: len(SEQUENCES[label]) + 1].clone()
                        for i, label in enumerate(labels)
                    )
                        
        else:
            for counter, label in enumerate(labels):
                try:
                    cdr3_sequence = CDR3_DICT[label]
                except KeyError:
                    if label not in MISSING_KEYS:
                        print(f'No cdr3 sequence found for {label}')
                    continue
                #print(f'Processing {label}')

                # load sequence without spaces
                full_sequence = SEQUENCES[label].replace(' ', '')

                # remove '-' from cdr3_sequence
                cdr3_sequence = cdr3_sequence.replace('-', '')

                # get position of cdr3_sequence in sequence
                try:
                    start = full_sequence.find(cdr3_sequence) - CONTEXT
                except ValueError:
                    print("Context window too large")
                try:
                    end = start + len(cdr3_sequence) + CONTEXT
                except ValueError:
                    print("Context window too large")
                SEQUENCE_LABELS.append(label)

                for layer in LAYERS:
                    if POOLING:
                        mean_representation = representations[layer][counter, start : end].mean(0).clone()
                    else:
                        mean_representation = representations[layer][counter, start : end].clone()
                    # We take mean_representation[0] to keep the [array] instead of [[array]].
                    MEAN_REPRESENTATIONS[layer].append(mean_representation)

        # print the progress
        #print(
        #    f"Processing {batch_idx + 1} of {TOTAL_BATCHES} batches ({INPUT_IDS.size(0)} sequences)"
        #)

# Clear GPU memory
torch.cuda.empty_cache()

# Stacking representations of each layer into a single tensor and save to output file
for layer in LAYERS:
    MEAN_REPRESENTATIONS[layer] = torch.vstack(MEAN_REPRESENTATIONS[layer])
    OUTPUT_PATH_LAYER = OUTPUT_PATH.replace('.pt', f'_layer_{layer}.pt')
    if not POOLING:
        OUTPUT_PATH_LAYER = OUTPUT_PATH_LAYER.replace('.pt', '_full.pt')
    torch.save(MEAN_REPRESENTATIONS[layer], OUTPUT_PATH_LAYER)

OUTPUT_FILE_IDX = OUTPUT_PATH.replace('.pt', '_idx.csv')
with open(OUTPUT_FILE_IDX, 'w') as f:
    f.write('index,sequence_id\n')
    for i, label in enumerate(SEQUENCE_LABELS):
        f.write(f'{i},{label}\n')
print(f"Saved sequence indices to {OUTPUT_FILE_IDX}")