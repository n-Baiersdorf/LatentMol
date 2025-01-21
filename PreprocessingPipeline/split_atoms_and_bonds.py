import os
import json
from tqdm import tqdm

class JsonSplitter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.atoms_dir = os.path.join(output_dir, "atoms")
        self.bonds_dir = os.path.join(output_dir, "bonds")

    def create_output_dirs(self):
        os.makedirs(self.atoms_dir, exist_ok=True)
        os.makedirs(self.bonds_dir, exist_ok=True)

    def process_files(self):
        self.create_output_dirs()
        json_files = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        
        for filename in tqdm(json_files, desc="Split Atoms from Bonds"):
            self.process_file(filename)

    def process_file(self, filename):
        input_path = os.path.join(self.input_dir, filename)
        with open(input_path, 'r') as f:
            data = json.load(f)

        atoms = data[::2]
        bonds = data[1::2]

        new_atom_filename = filename.replace("entry_", "atom_")
        new_bond_filename = filename.replace("entry_", "bond_")

        self.save_json(os.path.join(self.atoms_dir, new_atom_filename), atoms)
        self.save_json(os.path.join(self.bonds_dir, new_bond_filename), bonds)

    @staticmethod
    def save_json(filepath, data):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
