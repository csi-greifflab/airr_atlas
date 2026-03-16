#!/bin/bash
# NOTE: Paths refer to HPC server. Modify for your environment.

#source /home/marinafr/.bashrc
#conda activate /home/marinafr/immuneML/immuneml_env

workers=32

script="/doctorai/marinafr/2023/airr_atlas/analysis/scripts/OHE_script.py"

python "$script" --input_path /doctorai/marinafr/2023/airr_atlas/analysis/data/ohe/test/test.csv --folder /doctorai/marinafr/2023/airr_atlas/analysis/output/ohe/test/immuneml --output_path /doctorai/marinafr/2023/airr_atlas/analysis/output/ohe/test --labels batch --junction_aa sequence_aa --yaml_template /doctorai/marinafr/2023/airr_atlas/analysis/scripts/test.yaml
