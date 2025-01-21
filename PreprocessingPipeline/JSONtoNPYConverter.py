import numpy as np
import json
import os
from tqdm import tqdm

class JsonToNpyConverter:
    def __init__(self, json_dir, npy_dir, atoms_or_bonds):
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)

        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        desc = "Atoms: JSON to NPY " if atoms_or_bonds == 1 else "Bonds: JSON to NPY "

        for filename in tqdm(json_files, desc=desc):
            with open(os.path.join(json_dir, filename), 'r') as f:
                data = json.load(f)
            
            # Überprüfen, ob die Daten eine Liste von Listen sind
            if isinstance(data, list) and all(isinstance(item, list) for item in data):
                # Convert to a NumPy array with a specific dtype (e.g., float32)
                np_array = np.array(data, dtype=np.float32)
            else:
                print(f"Warnung: Die Daten in '{filename}' sind nicht im erwarteten Format.")
                continue
            
            # Speichern als typisiertes NumPy-Array
            np.save(os.path.join(npy_dir, filename[:-5] + '.npy'), np_array)

# Beispielaufruf der Klasse
if __name__ == "__main__":
    atom_bond_split_data_sequence_json = "data/split/"
    input_atoms = os.path.join(atom_bond_split_data_sequence_json, "atoms")
    input_bonds = os.path.join(atom_bond_split_data_sequence_json, "bonds")
    output_mixed = "data/mixed/"
    output_atoms = os.path.join(output_mixed, "atoms_npy")
    output_bonds = os.path.join(output_mixed, "bonds_npy")
    JsonToNpyConverter(input_atoms, output_atoms, 1)
    JsonToNpyConverter(input_bonds, output_bonds, 0)
 