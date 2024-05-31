import pathlib
import argparse

from Bio import SeqIO

# Parsing command-line arguments for input and output file paths
parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("fasta_path", type=pathlib.Path, help="Fasta path + filename.fa")
parser.add_argument("output_dir", type=pathlib.Path, help="Output directory")
parser.add_argument("n_batches", type=int, help="Number of batches  to split the fasta into")
args = parser.parse_args()


# Storing the input and output file paths
input_file = args.fasta_path


# Store output_dir and create it if it doesn't exist.
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

num_batches = args.n_batches


total_records =  len([1 for line in open(input_file) if line.startswith(">")])

batch_size = total_records // num_batches
remainder = total_records % num_batches
batch_sizes = [batch_size + 1 if i < remainder else batch_size for i in range(num_batches)]

current_batch_size = batch_sizes[0]
current_batch_num = 1
current_batch = []

for idx, record in enumerate(SeqIO.parse(input_file, "fasta")):
  if len(current_batch) < batch_sizes[current_batch_num - 1]:
    current_batch.append(record)
  else:
    SeqIO.write(
      current_batch,
      f'{output_dir}/{input_file.stem}_batch_{current_batch_num}_of_{len(batch_sizes)}.fa',
      'fasta')
    print(f'Batch {current_batch_num} of {len(batch_sizes)} done')
    current_batch_num += 1
    current_batch = []

if current_batch:
  SeqIO.write(
    current_batch,
    f'{output_dir}/{input_file.stem}_batch_{current_batch_num}_of_{len(batch_sizes)}.fa',
    'fasta')
  print(f'Batch {current_batch_num} of {len(batch_sizes)} done')
  
  
