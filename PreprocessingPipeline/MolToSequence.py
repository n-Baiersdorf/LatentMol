from typing import List, Dict, Tuple
import os
import json
from misc.PSE_Data import atom_dict_normalized as atom_constants


"""The MolToSequence Script is the most important script. This Script takes in text a text file with Moltables and 
   transforms them into the LatentMol sequence format."""


class MolToSequence:
    def __init__(self):
        self.SEM_PAD = -1
        self.MAX_BONDS = 4
        self.atom_element_map = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
            'P': 15, 'S': 16, 'Cl': 17, 'Se': 34, 'Br': 35, 'I': 53
        }

    def parse_mol_file(self, input_path: str) -> Dict[str, List[List[float]]]:
        molecules_dict = {}
        current_mol_id = None
        current_version = None
        mol_block_lines = []

        with open(input_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('###'):
                    current_mol_id = line.strip('#')
                elif line.startswith('§§Version_'):
                    if mol_block_lines:
                        self._process_mol_block(current_mol_id, current_version, mol_block_lines, molecules_dict)
                    mol_block_lines = []
                    current_version = line.split('_')[1].split('§')[0]
                elif line and 'CDKD' not in line:
                    mol_block_lines.append(line)

            if mol_block_lines:
                self._process_mol_block(current_mol_id, current_version, mol_block_lines, molecules_dict)

        return molecules_dict

    def _process_mol_block(self, mol_id: str, version: str, lines: List[str], molecules_dict: Dict):
        try:
            atoms, bonds = self._parse_structure(lines)
            normalized_atoms = self._normalize_coordinates(atoms)
            sequences = self._create_sequences(normalized_atoms, bonds)
            key = f"{mol_id}_VERSION_{version}"
            molecules_dict[key] = sequences
        except Exception as e:
            print(f"Error processing {mol_id} version {version}: {str(e)}")

    def _parse_structure(self, lines: List[str]) -> Tuple[List[Dict], List[List[int]]]:
        counts_line = next(i for i, line in enumerate(lines) if 'V2000' in line)
        num_atoms = int(lines[counts_line].split()[0])
        num_bonds = int(lines[counts_line].split()[1])

        atoms = []
        for line in lines[counts_line + 1:counts_line + 1 + num_atoms]:
            parts = line.split()
            atoms.append({
                'coords': [float(parts[0]), float(parts[1])],
                'element': parts[3]
            })

        bonds = []
        for line in lines[counts_line + 1 + num_atoms:counts_line + 1 + num_atoms + num_bonds]:
            parts = line.split()
            for _ in range(int(parts[2])):
                bonds.append([int(parts[0]), int(parts[1])])

        return atoms, bonds

    def _create_sequences(self, atoms: List[Dict], bonds: List[List[int]]) -> List[List[float]]:
        sequences = []
        for idx, atom in enumerate(atoms, 1):
            sequence = self._create_atom_sequence(atom, idx, bonds, atoms)
            sequences.append(sequence)
        return sequences

    def _create_atom_sequence(self, atom: Dict, atom_idx: int, bonds: List[List[int]], atoms: List[Dict]) -> List[float]:
        atom_bonds = self._get_atom_bonds(atom_idx, bonds)
        processed_bonds = self._process_bonds(atom_bonds, atoms)
        padded_bonds = self._pad_bonds(processed_bonds, atom_idx, atom['element'])
        
        sequence = [
            atom['coords'][0],  # Now using normalized x-coordinate
            atom['coords'][1],  # Now using normalized y-coordinate
            *self._get_atom_constants(atom['element'])
        ]
        
        for bond in padded_bonds:
            sequence.extend(bond)
        
        return sequence

    def _get_atom_bonds(self, atom_idx: int, bonds: List[List[int]]) -> List[List[int]]:
        return [bond for bond in bonds if atom_idx in bond]

    def _process_bonds(self, bonds: List[List[int]], atoms: List[Dict]) -> List[List[float]]:
        processed = []
        for bond in bonds:
            atom1_idx = bond[0] - 1
            atom2_idx = bond[1] - 1
            en_diff = self._calculate_en_diff(
                atoms[atom1_idx]['element'],
                atoms[atom2_idx]['element']
            )
            processed.append([
                self._normalize_bond_index(bond[0]),
                self._normalize_bond_index(bond[1]),
                en_diff
            ])
        return processed

    def _normalize_bond_index(self, index: int) -> float:
        return 0.5 + index * 0.01

    def _calculate_en_diff(self, element1: str, element2: str) -> float:
        en1 = atom_constants[element1][5]
        en2 = atom_constants[element2][5]
        return 0.5 * abs(en1 - en2) / 3.28

    def _pad_bonds(self, bonds: List[List[float]], atom_idx: int, element: str) -> List[List[float]]:
        padded_bonds = bonds.copy()
        remaining = self.MAX_BONDS - len(bonds)
        
        if element == 'H' and len(bonds) > 1:
            raise ValueError(f"Hydrogen atom {atom_idx} has multiple bonds")
            
        pad_value = [self.SEM_PAD] * 3 if element == 'H' else [
            self._normalize_bond_index(atom_idx),
            self._normalize_bond_index(atom_idx),
            self.SEM_PAD
        ]
        
        for _ in range(remaining):
            padded_bonds.append(pad_value)
        return padded_bonds

    def _get_atom_constants(self, element: str) -> List[float]:
        try:
            return atom_constants[element]
        except KeyError:
            raise ValueError(f"Element {element} not found in atom constants")

    def _normalize_coordinates(self, atoms: List[Dict]) -> List[Dict]:
        x_coords = [atom['coords'][0] for atom in atoms]
        y_coords = [atom['coords'][1] for atom in atoms]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        normalized_atoms = []
        for atom in atoms:
            x_norm = 1 + 0.5 * (atom['coords'][0] - x_min) / (x_max - x_min)
            y_norm = 1 + 0.5 * (atom['coords'][1] - y_min) / (y_max - y_min)
            normalized_atoms.append({
                'coords': [x_norm, y_norm],
                'element': atom['element']
            })
        
        return normalized_atoms



def molToSequenceFunction(input_path: str, output_dir: str, number: int):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the complete output file path
    output_file = os.path.join(output_dir, f"Länge_{number}.json")
    
    # Process the molecules
    try:
        processor = MolToSequence()
        molecules_dict = processor.parse_mol_file(input_path)
        
        # Write to file, ensuring the parent directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(molecules_dict, f, indent=2)
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    # Example usage
    input_path = "path/to/your/input/molecules.txt"
    output_dir = "data/testTest"
    number = 1
    molToSequenceFunction(input_path, output_dir, number)
