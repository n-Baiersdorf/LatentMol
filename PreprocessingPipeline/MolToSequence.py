import os
import json
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

# For demo purposes, re-import the same atom constants as in the original code
from misc.PSE_Data import atom_dict_normalized as atom_constants

class MolToSequence:
    """
    An updated version of MolToSequence that implements the LatentMol format
    as described in sections 4.1 and 4.3, with configurable normalization
    intervals and placeholders for free electron pairs, H, and C.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        config can override the default normalization intervals from 4.3.
        Defaults are:
         - bond_index in [0, 0.1] (x' = x * 0.001 * l)
         - EN_diff in [0.1, 0.25] (x' = (x / 3.20) * 0.15 + 0.1)
         - charge in [0.25, 0.35] (x' = 0.05 * x * l + 0.3, x in [-1,0,1])
         - coords in [0.01,0.2]
         - atom constants in [0.55,1.0]
        """
        self.config = config if config else {}
        # SEM_PAD used for placeholders (e.g., hydrogen placeholders)
        self.SEM_PAD = -1
        self.MAX_BONDS = 4

        # Defaults from the paper
        self.l_factor = self.config.get("l_factor", 1.0)
        self.bond_index_scale = self.config.get("bond_index_scale", 0.001)
        self.bond_index_offset = self.config.get("bond_index_offset", 0.0)
        self.en_diff_max = self.config.get("en_diff_max", 3.20)
        self.en_diff_lower = self.config.get("en_diff_lower", 0.1)
        self.en_diff_range = self.config.get("en_diff_range", 0.15)
        self.charge_lower = self.config.get("charge_lower", 0.25)
        self.charge_range = self.config.get("charge_range", 0.1)    # .05 * l => we store as 0.1 for x in [-1,0,1]
        self.default_ch_endiff = self.config.get("default_ch_endiff", 0.1)  # or arbitrary fallback
        self.coords_min = self.config.get("coords_min", 0.01)
        self.coords_range = self.config.get("coords_range", 0.19)
        self.atom_const_min = self.config.get("atom_const_min", 0.55)
        self.atom_const_range = self.config.get("atom_const_range", 0.45)

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
                    # encountered new molecule
                    current_mol_id = line.strip('#')
                elif line.startswith('§§Version_'):
                    if mol_block_lines:
                        self._process_mol_block(current_mol_id, current_version,
                                                mol_block_lines, molecules_dict)
                        mol_block_lines = []
                    current_version = line.split('_')[1].split('§')[0]
                elif line and 'CDKD' not in line:
                    mol_block_lines.append(line)
            if mol_block_lines:
                self._process_mol_block(current_mol_id, current_version,
                                        mol_block_lines, molecules_dict)
        return molecules_dict

    def _process_mol_block(self, mol_id: str, version: str,
                           lines: List[str], molecules_dict: Dict):
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
        start_atoms = counts_line + 1
        for line in lines[start_atoms : start_atoms + num_atoms]:
            parts = line.split()
            atoms.append({
                'coords': [float(parts[0]), float(parts[1])],
                'element': parts[3]
            })

        bonds = []
        start_bonds = start_atoms + num_atoms
        for line in lines[start_bonds : start_bonds + num_bonds]:
            parts = line.split()
            # The Mol table might store bond order as parts[2] -> replicate each in single-bond steps:
            bond_count = int(parts[2])
            a1 = int(parts[0])
            a2 = int(parts[1])
            for _ in range(bond_count):
                bonds.append([a1, a2])
        return atoms, bonds

    def _create_sequences(self, atoms: List[Dict], bonds: List[List[int]]) -> List[List[float]]:
        sequences = []
        for idx, atom in enumerate(atoms, 1):
            seq = self._create_atom_sequence(atom, idx, bonds, atoms)
            sequences.append(seq)
        return sequences

    def _create_atom_sequence(self, atom: Dict, atom_idx: int,
                              bonds: List[List[int]], atoms: List[Dict]) -> List[float]:
        atom_bonds = self._get_atom_bonds(atom_idx, bonds)
        processed_bonds = self._process_bonds(atom_bonds, atoms)
        padded_bonds = self._pad_bonds(processed_bonds, atom_idx, atom['element'])
        # Build final token = [x_coord, y_coord, *atom_constants, bond1, bond2, bond3, bond4]
        sequence = [
            atom['coords'][0],
            atom['coords'][1]
        ]
        sequence.extend(self._get_normalized_atom_constants(atom['element']))
        for bond in padded_bonds:
            sequence.extend(bond)
        return sequence

    def _get_atom_bonds(self, atom_idx: int, bonds: List[List[int]]) -> List[List[int]]:
        return [bond for bond in bonds if atom_idx in bond]

    def _process_bonds(self, bonds_list: List[List[int]], atoms: List[Dict]) -> List[List[float]]:
        processed = []
        for bond in bonds_list:
            a1 = bond[0] - 1
            a2 = bond[1] - 1
            en_diff_val = self._calculate_en_diff(atoms[a1]['element'], atoms[a2]['element'])
            # Normalize each index
            b1 = self._normalize_bond_index(bond[0])
            b2 = self._normalize_bond_index(bond[1])
            processed.append([b1, b2, en_diff_val])
        return processed

    def _normalize_bond_index(self, raw_index: int) -> float:
        """
        Bond index is scaled to [0,0.1].
        Formula: x' = x * 0.001 * l  (paper uses 0.001 for 100 max atoms).
        We also allow an offset if desired, but default offset is 0.
        """
        return (raw_index * self.bond_index_scale * self.l_factor) + self.bond_index_offset

    def _calculate_en_diff(self, element1: str, element2: str) -> float:
        """
        According to paper 4.3: x' = (x / ENDiffMax) * 0.15 + 0.1
        with x = absolute difference in electronegativity.
        """
        en1 = atom_constants[element1][5]
        en2 = atom_constants[element2][5]
        raw_diff = abs(en1 - en2)
        return (raw_diff / self.en_diff_max) * self.en_diff_range + self.en_diff_lower

    def _calculate_charge(self, charge_val: int) -> float:
        """
        For charges in [-1,0,1], we do: x' = 0.05 * x * l + 0.3 if default range is 0.1
        Range is [0.25,0.35] in default config if l=1. E.g. for x=+1 => 0.35, for x=-1 => 0.25.
        """
        return (0.05 * charge_val * self.l_factor) + (self.charge_lower + 0.05)

    def _pad_bonds(self, processed_bonds: List[List[float]], atom_idx: int, element: str) -> List[List[float]]:
        """
        Exactly four bond entries are required.
        If fewer, we fill them with free electron pairs or placeholders as described:
          - If 'H' => [SEM_PAD, SEM_PAD, SEM_PAD]
          - If 'C' => [norm_index, SEM_PAD, default_ch_endiff] for implied C-H, or free pairs if needed
          - Otherwise => free electron pairs: [norm_index, norm_index, placeholder_for_ENdiff_or_charge]
        """
        out = deepcopy(processed_bonds)
        needed = self.MAX_BONDS - len(out)
        if needed < 0 and element == 'H':
            # If hydrogen has multiple bonds, it's invalid per the paper
            raise ValueError(f"Hydrogen atom {atom_idx} has multiple bonds.")
        if needed <= 0:
            return out

        # We will fill the missing bond slots
        if element == 'H':
            # Fill empty bond entries with hydrogen placeholders
            for _ in range(needed):
                out.append([self.SEM_PAD, self.SEM_PAD, self.SEM_PAD])
        elif element == 'C':
            # For carbon, fill with implied C-H (index, SEM_PAD, CH_enDiff) or free pairs if we exceed
            # the total single bonds possible. Simplified approach: fill all missing with CH placeholders.
            norm_idx = self._normalize_bond_index(atom_idx)
            # We can compute the EN diff for C-H automatically if we want:
            # This is the raw difference between C and H:
            c_en = atom_constants['C'][5]
            h_en = atom_constants['H'][5]
            raw_ch_diff = abs(c_en - h_en)
            ch_en_diff = (raw_ch_diff / self.en_diff_max) * self.en_diff_range + self.en_diff_lower
            # But user can override with default_ch_endiff if desired:
            # We'll pick the actual difference as it is more correct for "implied" H:
            implied_ch = ch_en_diff if not self.config.get("use_fixed_ch_diff") else self.default_ch_endiff
            for _ in range(needed):
                out.append([norm_idx, self.SEM_PAD, implied_ch])
        else:
            # Generic is free electron pair: [norm_index, norm_index, placeholder]
            # If the free pair is uncharged, we put a placeholder for EN diff. We'll do SEM_PAD or 0.0
            norm_idx = self._normalize_bond_index(atom_idx)
            placeholder_en = self.SEM_PAD
            # We can also choose 0.0 or some special symbol indicating no charge. Using SEM_PAD for now.
            for _ in range(needed):
                out.append([norm_idx, norm_idx, placeholder_en])
        return out

    def _get_normalized_atom_constants(self, element: str) -> List[float]:
        """
        Atom constants from the dictionary are scaled to [0.55, 1.0] by default.
        This is a simple min-max approach for each constant set if the user provided them.
        If an element is missing in the dictionary, we raise error.
        """
        if element not in atom_constants:
            raise ValueError(f"Element {element} not found in atom constants.")
        values = atom_constants[element]  # list of e.g. 12 constants
        # We replicate the approach: ( val - min ) / (max - min ) * 0.45 + 0.55
        # But we do it separately for each property if that's required. This example uses them as a whole range.
        # We'll do a min-max across all elements in the dictionary for each property index
        # That would imply scanning the dict each time. Alternatively, we can do a direct pass since
        # the user in the paper does a general min->max for each property over all possible elements.
        # For demonstration, we'll do an inline approach that extends to each property individually.

        # We'll do a simple local min-max using the entire dictionary for each property index:
        # That can be precomputed. For brevity, we do it on the fly here.
        # We assume each property index i is columns across the dictionary for all possible elements.
        normalized = []
        for i, val in enumerate(values):
            # gather all possible values for property i
            all_vals_for_i = [atom_constants[e][i] for e in atom_constants]
            p_min = min(all_vals_for_i)
            p_max = max(all_vals_for_i)
            if p_max - p_min == 0.0:
                # avoid division by zero; fallback to middle of the range
                scaled = self.atom_const_min + self.atom_const_range * 0.5
            else:
                scaled = ((val - p_min) / (p_max - p_min)) * self.atom_const_range + self.atom_const_min
            normalized.append(scaled)
        return normalized

    def _normalize_coordinates(self, atoms: List[Dict]) -> List[Dict]:
        """
        Coordinates are scaled from min->max into [0.01, 0.2].
        x' = ((x - x_min)/(x_max - x_min)) * 0.19 + 0.01
        """
        x_coords = [a['coords'][0] for a in atoms]
        y_coords = [a['coords'][1] for a in atoms]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        denom_x = x_max - x_min if x_max != x_min else 1.0
        denom_y = y_max - y_min if y_max != y_min else 1.0

        norm_list = []
        for atom in atoms:
            nx = ((atom['coords'][0] - x_min) / denom_x) * self.coords_range + self.coords_min
            ny = ((atom['coords'][1] - y_min) / denom_y) * self.coords_range + self.coords_min
            norm_list.append({'coords': [nx, ny], 'element': atom['element']})
        return norm_list

def molToSequenceFunction(input_path: str, output_dir: str, number: int, config: Dict = None):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"Länge_{number}.json")

    try:
        processor = MolToSequence(config=config)
        molecules_dict = processor.parse_mol_file(input_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(molecules_dict, f, indent=2)
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    # Example usage with defaults from paper
    test_input_path = "path/to/your/input/molecules.txt"
    test_output_dir = "data/testTest"
    test_number = 1
    # Example custom config: override some intervals if desired
    custom_config = {
        # "en_diff_max": 3.2,
        # "use_fixed_ch_diff": True,
        # "default_ch_endiff": 0.2
    }
    molToSequenceFunction(test_input_path, test_output_dir, test_number, config=custom_config)



# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.