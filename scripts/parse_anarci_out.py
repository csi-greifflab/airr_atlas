import pandas as pd
import os
import argparse
import multiprocessing

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='Path to the data directory containing ANARCI output CSV files')
parser.add_argument('--datasetname', type=str, help='Name of the dataset')
parser.add_argument('--output_dir', type=str, help='Path to the output directory')
parser.add_argument('--num_cores', type=int, default=1, help='Number of cores to use for parallel processing')
args = parser.parse_args()


# Get a list of all CSV files in the data directory
csv_files = [file for file in os.listdir(args.input_dir) if file.endswith('.csv')]

#debug
#filepath = '/doctorai/userdata/airr_atlas/data/sequences/wang_H_full_chains_batches/wang_H_full_chains_batch_1_of_24.csv_H.csv'

# Initialize an empty list to store the dataframes
dfs = []

# Define a function to process each CSV file
def process_csv(file):
    # Load the CSV file
    filepath = os.path.join(args.input_dir, file)
    df = pd.read_csv(filepath)
    
    # Get index of column starting with '95' (Chothia/Martin numbering for CDR3 start)
    cdr3_start = df.columns.get_loc(df.filter(regex='^95').columns[0])
    # Get index of last column starting with '102' (Chothia/Martin numbering for CDR3 end)
    cdr3_end = df.columns.get_loc(df.filter(regex='^102').columns[-1])

    # Drop all columns except Id and columns between cdr3_start and cdr3_end
    df = df.iloc[:, [0] + list(range(cdr3_start, cdr3_end + 1))]
    
    # Paste all columns into a single string column called 'cdr3_sequence'
    df['cdr3_sequence'] = df.iloc[:, 1:].apply(lambda row: ''.join(row.dropna().astype(str)), axis=1)
    
    # Drop all columns except 'Id' and 'cdr3_sequence'
    df = df[['Id', 'cdr3_sequence']]
    
    return df

# Create a pool of worker processes
pool = multiprocessing.Pool(processes=args.num_cores)

# Apply the process_csv function to each CSV file in parallel
results = pool.map(process_csv, csv_files)

# Close the pool of worker processes
pool.close()
pool.join()

# Append the dataframes to the list
dfs.extend(results)

# Merge all dataframes into a single dataframe
merged_df = pd.concat(dfs, ignore_index=True)

# Rename 'Id' column to 'id'
merged_df.rename(columns={'Id': 'id'}, inplace=True)

# Sort df by 'id' column
merged_df.sort_values(by='id', inplace = True)



# Create the output directory if it doesn't exist
output_filepath = os.path.join(args.output_dir, f'{args.datasetname}_cdr3.csv')
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

# Save the merged dataframe to a CSV and a feather file
merged_df.to_csv(output_filepath, index=False)
merged_df.to_feather(output_filepath.replace('.csv', '.feather'))
