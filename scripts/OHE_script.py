import yaml
import argparse
import subprocess
import os
import shutil
import torch
import time

# Function to update the yaml content
def update_yaml(input_path, folder_path, labels, sequence_aa_column, yaml_template):
    yaml_template['definitions']['datasets']['my_dataset']['params']['path'] = input_path
    yaml_template['definitions']['datasets']['my_dataset']['params']['column_mapping'][sequence_aa_column] = 'junction_aa'
    yaml_template['definitions']['datasets']['my_dataset']['params']['result_path'] = folder_path
    yaml_template['instructions']['my_instruction']['analyses']['my_analysis_1']['labels'] = labels
    return yaml_template

# Main function to handle arguments and modify YAML
def main():
    """
    Main function to generate and run immuneML YAML configuration.
    This function performs the following steps:
    1. Parses command-line arguments.
    2. Loads a YAML template file.
    3. Updates the YAML template with user-provided parameters.
    4. Saves the updated YAML configuration to a new file.
    5. Runs immuneML with the updated YAML file using a specified conda environment.
    6. Converts the output tensor file to int16 format and saves it.
    7. Renames the labels CSV file.
    8. Removes specific folders and files in the output path.
    Command-line arguments:
    --input_path (str): Path to the input CSV file.
    --folder_path (str): Path to the immuneML results folder.
    --output_path (str): Path to the output results folder.
    --labels (list of str): List of labels for analysis.
    --junction_aa (str): Name of the column with the the sequence to be OHE.
    --yaml_template (str): Path to the YAML template file.
    

    HOW TO RUN:
    The script uses a YAML file as a template. 
    It modifies this template dynamically and saves it as updated_template.yaml (or a similarly named file). 
    After updating the template, the script invokes the immuneML program, which requires two key arguments:
	    1.	A YAML configuration file (e.g., updated_template.yaml).
	    2.	An output folder where results will be stored.

    The process is fully automated, with the script handling the preparation of the YAML file, running immuneML, and managing the outputs.
    Once the program completes, the specified output folder will contain the results, including:
	•	A .pt file (e.g., containing the OHE encoded dataset in PyTorch-compatible data).
	•	A .csv file (e.g., containing the labels of each row of the OHE dataset).


    Usage Examples:
    --------------------
    Example 1 (Basic usage):
    python script.py --input_path /path/to/input.csv --folder_path /path/to/folder --output_path /path/to/output \
                     --labels cancer_loc --junction_aa sequence_aa --yaml_template ./template_OHE.yaml


    Example of required csv input file:
    sequence_id,sequence_aa,cancer_loc
    PCall_CESC3_T_AACTTTCCATTGCGGC-1_3,CARGGPRWGSTFYPGDLW,CESC_Cancer
    PCall_CESC3_T_ACTGATGGTAGGCTGA-1_3,CAREMGYCSRSNCGRGGFDFW,CESC_Cancer
    PCall_CESC3_T_ATTACTCTCCTTGACC-1_3,CARGGRIATTW,CESC_Cancer
    PCall_CESC3_T_CATATGGTCTCGTATT-1_3,CAKGGITPGDYW,CESC_Cancer
    """
    parser = argparse.ArgumentParser(description='Generate and run immuneML YAML configuration')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the immuneML results folder - specify one for all the analysis')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output results folder ! THIS IS THE FOLDER WHERE THE TENSOR FILE WILL BE SAVED !')
    parser.add_argument('--labels', nargs='+', required=True, help='List of labels for analysis')
    parser.add_argument('--junction_aa', type=str, required=True, help='Name of the column with the the sequence to be OHE')
    parser.add_argument('--yaml_template', type=str, required=True, help='Path to the YAML template file')
    args = parser.parse_args()

    # Load the template YAML file
    with open(args.yaml_template, 'r') as yaml_file:
        yaml_template = yaml.safe_load(yaml_file)

    # Update YAML file with user parameters
    updated_yaml = update_yaml(args.input_path, args.folder_path, args.labels, args.junction_aa, yaml_template)

    # Save the updated YAML to a new file
    updated_yaml_file = 'updated_config.yaml'
    with open(updated_yaml_file, 'w') as yaml_file:
        yaml.dump(updated_yaml, yaml_file, default_flow_style=False)

    # Run immuneML with the updated YAML file using the specified conda environment
    try:
        subprocess.run(['conda', 'run', '--prefix', '/doctorai/userdata/envs/immuneML_ohe', 'immune-ml', updated_yaml_file, args.output_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running immuneML: {e}")
        # Ask the user if they want to execute the command
        start_time = time.time()
        execute_command = input("Do you want to remove the immuneML folder? (y/n): ").strip().lower()
        elapsed_time = time.time() - start_time

        if elapsed_time > 30:
            print("Timeout exceeded. Command execution skipped.")
        elif execute_command == 'y':
            # Delete the folder if it exists
            if os.path.exists(args.output_path):
                shutil.rmtree(args.output_path)
                print("immuneML folder removed.")
        else:
            print("Command execution skipped.")

    tensor_name = os.path.basename(args.output_path)
    # Read the .pt tensor from the output path and convert it into int16
    tensor_file = os.path.join(args.output_path, 'my_instruction/analysis_my_analysis_1/report/design_matrix.pt')
    if os.path.exists(tensor_file):
        tensor = torch.load(tensor_file)
        tensor = tensor.to(torch.int16)
        new_tensor_file = os.path.join(args.output_path,f'{tensor_name}_OHE.pt'  )
        torch.save(tensor, new_tensor_file)
        print(f"Converted tensor saved as {new_tensor_file}")
    else:
        print(f"Tensor file {tensor_file} not found.")

    # Rename the labels.csv file
    labels_file = os.path.join(args.output_path, 'my_instruction/analysis_my_analysis_1/report/labels.csv')
    if os.path.exists(labels_file):
        new_labels_file = os.path.join(args.output_path, f'{tensor_name}_labels.csv')
        os.rename(labels_file, new_labels_file)
        print(f"Labels file renamed to {new_labels_file}")
    else:
        print(f"Labels file {labels_file} not found.")
    
    # Remove specific folders and files in the output path
    folders_to_remove = ['my_instruction', 'HTML_output']
    files_to_remove = ['index.html']

    for folder in folders_to_remove:
        folder_path = os.path.join(args.output_path, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Removed {folder} folder.")

    for file in files_to_remove:
        file_path = os.path.join(args.output_path, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file}.")


# Test function to quickly validate the script
def test():
    test_args = [
        '--input_path', '/doctorai/niccoloc/pancancer_b_cells_metadata_filtered.csv',
        '--folder_path', '/doctorai/niccoloc/OHE',
        '--output_path', '/doctorai/niccoloc/test1',
        '--labels', 'cancer_loc',
        '--junction_aa', 'sequence_aa',
        '--yaml_template', '/doctorai/niccoloc/res1/full_ohe_cancer.yaml'
    ]
    import sys

    sys.argv = ['script_name'] + test_args
    main()


test()

if __name__ == "__main__":
    main()


