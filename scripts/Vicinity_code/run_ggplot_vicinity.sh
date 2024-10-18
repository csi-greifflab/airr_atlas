#!/bin/bash

# Check if the correct number of arguments was passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 input_ED input_LD output_path"
    exit 1
fi

# Assign arguments to variables
input_ED=$1
input_LD=$2
output_path=$3

# Source the Conda configuration
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate R4.3.3


# Run the R script and capture stderr in a variable
error_message=$(Rscript ./Vicinity_ggplot.r "$input_ED" "$input_LD" "$output_path" 2>&1)

# Check the exit status
if [ $? -eq 0 ]; then
   echo "R script ran successfully."
else
   echo "R script failed to run: $error_message" >&2
   echo "$error_message" > /path/to/error_log.txt  # Save error message to a file if needed
   exit 1
fi