import json
import sys

from experiment_utils.configuration import args_path


def modify_json(qubit, element, key_to_change, new_value):
    """Modify a specific key in a JSON file and save the changes.

    Args:
        qubit (str): Qubit name.
        element (str): Element name
        key_to_change (str): Key in the JSON file to modify.
        new_value (any): New value to assign to the specified key.
    """

    file_path = args_path

    with open(file_path, 'r') as file:
        data = json.load(file)

    data[qubit][element][key_to_change] = new_value

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated '{key_to_change}' to '{new_value}' in '{file_path}'.")


if __name__ == "__main__":
    key = 'IQ_bias'
    new_value = {"I": 0.1, "Q": 0.2}
    modify_json("qubit1", "qubit", key, new_value)
