import os
import numpy as np
from typing import List, NamedTuple
import re
from tqdm import tqdm
import math
from multiprocessing import Pool

class MoleculeStructure(NamedTuple):
    header: List[str]
    atoms: List[str]
    bonds: List[str]
    footer: List[str]

class MoleculeSampleShuffler:

    @staticmethod
    def generate_permutations(n: int, max_permutations: int) -> List[List[int]]:
        rng = np.random.default_rng()
        total_permutations = math.factorial(n)
        
        if max_permutations is None or max_permutations > total_permutations:
            max_permutations = total_permutations
        
        return [rng.permutation(n).tolist() for _ in range(max_permutations)]

    @staticmethod
    def shuffle_molecule(mol_structure: MoleculeStructure, permutation: List[int]) -> MoleculeStructure:
        index_map = {old_idx + 1: new_idx + 1 for new_idx, old_idx in enumerate(permutation)}
        shuffled_atoms = [mol_structure.atoms[idx] for idx in permutation]
        shuffled_bonds = []

        for bond in mol_structure.bonds:
            parts = bond.split()
            try:
                first_atom, second_atom = map(int, parts[:2])
                new_first = index_map[first_atom]
                new_second = index_map[second_atom]
                shuffled_bonds.append(f"{new_first:>3} {new_second:>3} {' '.join(parts[2:])}\n")
            except (ValueError, KeyError) as e:
                print(f"Error processing bond: {bond}. Error: {e}")
                shuffled_bonds.append(bond)

        return MoleculeStructure(
            header=mol_structure.header,
            atoms=shuffled_atoms,
            bonds=shuffled_bonds,
            footer=mol_structure.footer
        )

    @classmethod
    def process_sample(cls, sample_text: str, max_permutations: int) -> List[str]:
        sample_lines = sample_text.split('\n')
        
        output_lines = [f"{sample_lines[0]}\n"]
        mol_structure = cls.parse_molecule_sample(sample_text)

        output_lines.append('§§Version_0§§\n')
        output_lines.append('\n'.join(
            mol_structure.header +
            mol_structure.atoms +
            mol_structure.bonds +
            mol_structure.footer
        ) + '\n')

        shuffled_molecules = cls.generate_shuffled_molecules(mol_structure, max_permutations)
        for version_num, shuffled_mol in enumerate(shuffled_molecules, 1):
            output_lines.append(f'§§Version_{version_num}§§\n')
            output_lines.append('\n'.join(
                shuffled_mol.header +
                shuffled_mol.atoms +
                shuffled_mol.bonds +
                shuffled_mol.footer
            ) + '\n')

        return output_lines

    @classmethod
    def generate_shuffled_molecules(cls, mol_structure: MoleculeStructure, max_permutations: int) -> List[MoleculeStructure]:
        n = len(mol_structure.atoms)
        permutations = cls.generate_permutations(n, max_permutations)

        # Use imap_unordered for better tqdm integration
        with Pool() as pool:
            results = list(tqdm(pool.imap_unordered(cls.shuffle_molecule_wrapper, [(mol_structure, perm) for perm in permutations]), total=len(permutations), desc="Shuffling molecules", leave=False))

        return results

    @staticmethod
    def shuffle_molecule_wrapper(args):
        mol_structure, perm = args
        return MoleculeSampleShuffler.shuffle_molecule(mol_structure, perm)

    @classmethod
    def parse_molecule_sample(cls, sample_text: str) -> MoleculeStructure:
        lines = sample_text.split('\n')
        v2000_index = next((i for i, line in enumerate(lines) if 'V2000' in line), -1)

        if v2000_index == -1:
            raise ValueError("No V2000 format found in the sample")

        header = lines[:v2000_index + 1]
        num_atoms = int(lines[v2000_index].split()[0])
        num_bonds = int(lines[v2000_index].split()[1])
        
        atoms = lines[v2000_index + 1:v2000_index + 1 + num_atoms]
        bonds = lines[v2000_index + 1 + num_atoms:v2000_index + 1 + num_atoms + num_bonds]
        
        footer = ['M END']
        
        return MoleculeStructure(
            header=header,
            atoms=atoms,
            bonds=bonds,
            footer=footer
        )

    @classmethod
    def process_file(cls, input_path: str, output_path: str, max_permutations: int):
        with open(input_path, 'r') as f:
            file_content = f.read()

        samples = re.findall(r'(###\d+###[\s\S]*?(?=###\d+###|$))', file_content)

        # Wrap the entire processing loop with tqdm to track overall progress of samples
        results = []
        with tqdm(total=len(samples), desc="Processing samples") as pbar:
            for sample in samples:
                result = cls.process_sample(sample, max_permutations)
                results.extend(result)
                pbar.update(1)  # Update progress bar after each sample is processed

        with open(output_path, 'w') as out_f:
            out_f.writelines(results)

    @staticmethod
    def remove_undesired_lines(input_path: str, output_path: str):
        with open(input_path, 'r') as f:
            lines = f.readlines()

        filtered_lines = []
        seen_ids = set()

        for line in lines:
            if line.startswith('###'):
                if line not in seen_ids:
                    seen_ids.add(line)
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)

        with open(output_path, 'w') as out_f:
            out_f.writelines(filtered_lines)

    @staticmethod
    def shuffle_molecules(input_path: str, output_dir: str, max_permutations: int):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        base_filename = os.path.basename(input_path)
        output_file = os.path.join(output_dir, f"shuffled_{base_filename}")

        temp_output_path = "temp_output.txt"
        
        MoleculeSampleShuffler.process_file(input_path, temp_output_path, max_permutations)
        
        MoleculeSampleShuffler.remove_undesired_lines(temp_output_path, output_file)
        
        os.remove(temp_output_path)


if __name__ == "__main__":
    input_file = "data_II/data_length_sorted/molecules_8.txt"
    output_directory = "output/length_8"  # Specify your desired output directory here
    max_permutations_limit = 1000000  # Set to a specific number to limit permutations or None for all

    MoleculeSampleShuffler.shuffle_molecules(input_file, output_directory, max_permutations_limit)