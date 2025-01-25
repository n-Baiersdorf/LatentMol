import json
import re
from tqdm import tqdm

# File paths
json_file = "data/output_molsequences_1/Länge_4.json"
molecule_sequence_data = "data/nmr_temp/test_output2.txt"

def load_json_data(file_path):
    """Load and parse JSON data."""
    with open(file_path, "r") as f:
        return json.load(f)

def load_txt_data(file_path):
    """Load and parse TXT data."""
    data = {}
    with open(file_path, "r") as f:
        for line in f:
            match = re.match(r"(\d+)_spectrum_.*?=", line)
            if match:
                key = match.group(1)
                if key not in data:
                    data[key] = []
                # Extract the sequence data from the line
                sequence = eval(line.split("=", 1)[1].strip())
                data[key].append(sequence)
    return data

def match_ids(json_data, txt_data):
    """Match IDs from JSON with TXT data."""
    result = {}
    for json_id_version in tqdm(json_data, desc="Processing IDs from JSON"):
        json_id = json_id_version.split("_VERSION_")[0]
        if json_id in txt_data:
            # Include all matches for this ID
            result[json_id_version] = txt_data[json_id]
        else:
            result[json_id_version] = []  # No matches found
    return result

def pair(output_path, molecule_sequence_data_path, spectra_path):

    




    # Load data
    print("Loading JSON data...")
    molecule_data = load_json_data(molecule_sequence_data_path)
    print("Loading TXT data...")
    spectrum_data= load_txt_data(spectra_path)

    # Match IDs and build dataset
    print("Matching IDs...")
    matched_data = match_ids(molecule_data, spectrum_data)

    # Output the result
    output_file = output_path
    print(f"Saving matched data to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(matched_data, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    output_path = "test.json"
    spectra_path = "data/nmr_temp/test_output2.txt"
    molecule_sequence_data_path = "data/output_molsequences_1/Länge_4.json"
    
    pair(output_path, molecule_sequence_data_path, spectra_path)



# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.