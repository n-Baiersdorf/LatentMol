import os
import json
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
from tqdm import tqdm
from misc.PSE_Data import atom_dict_normalized as atom_constants

class MolToSequence:
    def __init__(self, config: Optional[Dict] = None):
        # Bisherige Initialisierung bleibt gleich
        self.config = config if config else {}
        self.SEM_PAD = -1
        self.MAX_BONDS = 4
        self.l_factor = self.config.get("l_factor", 1.0)
        self.bond_index_scale = self.config.get("bond_index_scale", 0.001)
        self.bond_index_offset = self.config.get("bond_index_offset", 0.0)
        self.en_diff_max = self.config.get("en_diff_max", 3.20)
        self.en_diff_lower = self.config.get("en_diff_lower", 0.1)
        self.en_diff_range = self.config.get("en_diff_range", 0.15)
        self.charge_lower = self.config.get("charge_lower", 0.25)
        self.charge_range = self.config.get("charge_range", 0.1)
        self.default_ch_endiff = self.config.get("default_ch_endiff", 0.1)
        self.coords_min = self.config.get("coords_min", 0.01)
        self.coords_range = self.config.get("coords_range", 0.19)
        self.atom_const_min = self.config.get("atom_const_min", 0.55)
        self.atom_const_range = self.config.get("atom_const_range", 0.45)

    def parse_mol_file(self, input_path: str) -> Dict[str, List[List[float]]]:
        molecules_dict = {}
        current_mol_id = None
        current_version = None
        mol_block_lines = []
        
        with open(input_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc="Verarbeite Moleküle"):
                line = line.strip()
                if line.startswith('###'):
                    current_mol_id = line.strip('#')
                elif line.startswith('§§Version_'):
                    if mol_block_lines:
                        self._process_mol_block(
                            current_mol_id,
                            current_version,
                            mol_block_lines,
                            molecules_dict
                        )
                    mol_block_lines = []
                    current_version = line.split('_')[1].split('§')[0]
                elif line and 'CDKD' not in line:
                    mol_block_lines.append(line)

        if mol_block_lines:
            self._process_mol_block(
                current_mol_id,
                current_version,
                mol_block_lines,
                molecules_dict
            )
        return molecules_dict

    def _create_sequences(self, atoms: List[Dict], bonds: List[List[int]]) -> List[List[float]]:
        sequences = []
        for idx, atom in enumerate(tqdm(atoms, desc="Erstelle Sequenzen", leave=False), 1):
            seq = self._create_atom_sequence(atom, idx, bonds, atoms)
            sequences.append(seq)
        return sequences

def molToSequenceFunction(input_path: str, output_dir: str, number: int, config: Dict = None):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"Länge_{number}.json")
    
    try:
        processor = MolToSequence(config=config)
        with tqdm(total=1, desc="Verarbeite Datei") as pbar:
            molecules_dict = processor.parse_mol_file(input_path)
            pbar.update(1)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(molecules_dict, f, indent=2)
            
    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {str(e)}")

if __name__ == "__main__":
    test_input_path = "path/to/your/input/molecules.txt"
    test_output_dir = "data/testTest"
    test_number = 1
    custom_config = {}
    molToSequenceFunction(test_input_path, test_output_dir, test_number, config=custom_config)
