import os
import json

import numpy as np
from matplotlib import pyplot as plt


def load_json_file(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_experiment_and_number(file_name):
    """Extract experiment name and number from file name, return '0' if no number."""
    # Split the file name to remove extension
    base_name = os.path.splitext(file_name)[0]
    # Split by underscore
    parts = base_name.split('_')
    # Assume last part is the number if it's numeric
    if len(parts) > 1 and parts[-1].isdigit():
        return '_'.join(parts[:-1]), parts[-1]
    else:
        return '_'.join(parts), '0'


def create_data_dict(base_dir):
    """Create a nested dictionary from JSON files in the base directory."""
    data_dict = {}

    # Walk through the base directory
    for subdir, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            date_folder = os.path.join(subdir, dir_name)
            if dir_name not in data_dict:
                data_dict[dir_name] = {}

            # Process each JSON file in the date folder
            for file_name in os.listdir(date_folder):
                if file_name.endswith('.json'):
                    file_path = os.path.join(date_folder, file_name)
                    experiment_name, number = extract_experiment_and_number(file_name)

                    # Load JSON file
                    try:
                        file_data = load_json_file(file_path)
                        # Initialize experiment dictionary if it doesn't exist
                        if experiment_name not in data_dict[dir_name]:
                            data_dict[dir_name][experiment_name] = {}
                        # Add data to the appropriate experiment number
                        data_dict[dir_name][experiment_name][number] = file_data
                    except json.JSONDecodeError:
                        print(f"Error: Failed to decode JSON file {file_name}.")
                    except Exception as e:
                        print(f"Error: {e}")

    return data_dict


def plotter(data_dictionary, date, experiment, number):
    data = data_dictionary[date][experiment][number]['data']
    sweep = data_dictionary[date][experiment][number]['sweep']

    data_keys = data.keys()
    sweep_keys = sweep.keys()

    data_vec = data[data_keys[-1]]
    sweep_vec = sweep[sweep_keys[-1]]

    print(data_vec)
    print(sweep_vec)


# Define the base directory
base_dir = 'C:/Users/owner/Documents/GitHub/janis-lab/data'

# Create the dictionary
data_dictionary = create_data_dict(base_dir)
