import pandas as pd

import_file_path = "/doctorai/userdata/airr_atlas/data/saprot/trastuzumab_structure_aware_sequences.feather"
output_file_path = "/doctorai/userdata/airr_atlas/data/saprot/trastuzumab_structure_aware_sequences_light.fa"
in_file = pd.read_feather(import_file_path)

with open(output_file_path, "w") as fasta_file:
    for index, row in in_file.iterrows():
        fasta_file.write(f'>{row["name"]}\n{row["combined_light"]}\n')

