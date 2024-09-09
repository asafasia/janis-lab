import json
import os
from datetime import datetime


class Saver:
    def __init__(self, base_dir='data'):
        """Initializes Saver with a base directory for saving files."""
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def _generate_filename(self, file_name):
        """Generates a filename based on the current date and ensures uniqueness."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        folder_path = os.path.join(self.base_dir, date_str)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # base_filename = 'data'

        file_extension = '.json'
        file_path = os.path.join(folder_path, file_name + file_extension)

        # Ensure unique file name by appending a number if the file exists
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(folder_path, f"{file_name}_{counter}{file_extension}")
            counter += 1

        return file_path

    def save(self, file_name, data_dict, sweep, meta_data=None, args=None):
        """Saves the dictionary and metadata to a uniquely named file."""
        file_path = self._generate_filename(file_name)

        combined_data = {
            'data': data_dict,
            'sweep': sweep,
            'metadata': meta_data or {},
            'args': args or {}
        }

        with open(file_path, 'w') as f:
            json.dump(combined_data, f, indent=4)

        print(f"Data saved to {file_path}")

    def load(self, file_path):
        """Loads the dictionary and metadata from a file in the specified date folder."""

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, 'r') as f:
            combined_data = json.load(f)

        data_dict = combined_data.get('data', {})
        meta_data = combined_data.get('metadata', {})
        return data_dict, meta_data


# Example usage:
if __name__ == "__main__":
    saver = Saver()

    # Example data and metadata
    data = {'x': [1, 2, 3, 4], 'y': [2, 3, 1, 2]}
    metadata = {'qubit': 'q1', 'version': '1.0'}

    # Save the data and metadata
    saver.save('example', data, metadata)

    # Load the data and metadata (assuming we are loading from the current date and first file)
    loaded_data, loaded_metadata = saver.load('data/2024-08-26/example.json')
    print('Loaded Data:', loaded_data)
    print('Loaded Metadata:', loaded_metadata)
