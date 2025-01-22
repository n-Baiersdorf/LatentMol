import os
import shutil
from tqdm import tqdm


"""The MoleculeElementFilter Script is receiving a directory filled with files containing one Moltable each.
   It checks all elements present and based on the specified selection filteres all the moltables out that contain 
   elements not part of the specified list. This is because some elements might be present in such a small amount,
   and removing them may make the dataset more robust."""


class MoleculeElementFilter:
    # Comprehensive Periodic Table dictionary
    PERIODIC_TABLE = {
        'H': 'Hydrogen', 'He': 'Helium', 'Li': 'Lithium', 'Be': 'Beryllium', 'B': 'Boron', 
        'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen', 'F': 'Fluorine', 'Ne': 'Neon', 
        'Na': 'Sodium', 'Mg': 'Magnesium', 'Al': 'Aluminum', 'Si': 'Silicon', 'P': 'Phosphorus', 
        'S': 'Sulfur', 'Cl': 'Chlorine', 'Ar': 'Argon', 'K': 'Potassium', 'Ca': 'Calcium', 
        'Sc': 'Scandium', 'Ti': 'Titanium', 'V': 'Vanadium', 'Cr': 'Chromium', 'Mn': 'Manganese', 
        'Fe': 'Iron', 'Co': 'Cobalt', 'Ni': 'Nickel', 'Cu': 'Copper', 'Zn': 'Zinc', 
        'Ga': 'Gallium', 'Ge': 'Germanium', 'As': 'Arsenic', 'Se': 'Selenium', 'Br': 'Bromine', 
        'Kr': 'Krypton', 'Rb': 'Rubidium', 'Sr': 'Strontium', 'Y': 'Yttrium', 'Zr': 'Zirconium', 
        'Nb': 'Niobium', 'Mo': 'Molybdenum', 'Tc': 'Technetium', 'Ru': 'Ruthenium', 'Rh': 'Rhodium', 
        'Pd': 'Palladium', 'Ag': 'Silver', 'Cd': 'Cadmium', 'In': 'Indium', 'Sn': 'Tin', 
        'Sb': 'Antimony', 'Te': 'Tellurium', 'I': 'Iodine', 'Xe': 'Xenon', 'Cs': 'Cesium', 
        'Ba': 'Barium', 'La': 'Lanthanum', 'Ce': 'Cerium', 'Pr': 'Praseodymium', 'Nd': 'Neodymium', 
        'Pm': 'Promethium', 'Sm': 'Samarium', 'Eu': 'Europium', 'Gd': 'Gadolinium', 'Tb': 'Terbium', 
        'Dy': 'Dysprosium', 'Ho': 'Holmium', 'Er': 'Erbium', 'Tm': 'Thulium', 'Yb': 'Ytterbium', 
        'Lu': 'Lutetium', 'Hf': 'Hafnium', 'Ta': 'Tantalum', 'W': 'Tungsten', 'Re': 'Rhenium', 
        'Os': 'Osmium', 'Ir': 'Iridium', 'Pt': 'Platinum', 'Au': 'Gold', 'Hg': 'Mercury', 
        'Tl': 'Thallium', 'Pb': 'Lead', 'Bi': 'Bismuth', 'Po': 'Polonium', 'At': 'Astatine', 
        'Rn': 'Radon', 'Fr': 'Francium', 'Ra': 'Radium', 'Ac': 'Actinium', 'Th': 'Thorium', 
        'Pa': 'Protactinium', 'U': 'Uranium', 'Np': 'Neptunium', 'Pu': 'Plutonium', 'Am': 'Americium', 
        'Cm': 'Curium', 'Bk': 'Berkelium', 'Cf': 'Californium', 'Es': 'Einsteinium', 'Fm': 'Fermium', 
        'Md': 'Mendelevium', 'No': 'Nobelium', 'Lr': 'Lawrencium', 'Rf': 'Rutherfordium', 'Db': 'Dubnium', 
        'Sg': 'Seaborgium', 'Bh': 'Bohrium', 'Hs': 'Hassium', 'Mt': 'Meitnerium', 'Ds': 'Darmstadtium', 
        'Rg': 'Roentgenium', 'Cn': 'Copernicium', 'Nh': 'Nihonium', 'Fl': 'Flerovium', 'Mc': 'Moscovium', 
        'Lv': 'Livermorium', 'Ts': 'Tennessine', 'Og': 'Oganesson'
    }

    def __init__(self, input_directory, valid_files_dir, invalid_files_dir, allowed_elements):
        self.input_directory = input_directory
        self.valid_files_dir = valid_files_dir
        self.invalid_files_dir = invalid_files_dir
        self.allowed_elements = set(allowed_elements)
        
        os.makedirs(self.valid_files_dir, exist_ok=True)
        os.makedirs(self.invalid_files_dir, exist_ok=True)

    def filter_molecules(self):
        files = [f for f in os.listdir(self.input_directory) if f.endswith('.txt')]

        for filename in tqdm(files, desc="Element-Filter", unit="file"):
            file_path = os.path.join(self.input_directory, filename)
            try:
                with open(file_path, 'r') as file:
                    content = file.readlines()
                    if self.is_valid_molecule(content):
                        shutil.copy(file_path, self.valid_files_dir)
                    else:
                        shutil.copy(file_path, self.invalid_files_dir)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    def is_valid_molecule(self, content):
        in_mdl_table = False
        in_atom_block = False

        for line in content:
            if 'V2000' in line:
                in_mdl_table = True
                continue

            if in_mdl_table:
                if line.strip() and not line.startswith(("M ", "B ", ">")):
                    in_atom_block = True
                elif line.startswith(("B ", "M END")):
                    in_atom_block = False
                    break

                if in_atom_block:
                    element = self.extract_element(line)
                    if element and element not in self.allowed_elements:
                        return False

        return True

    def extract_element(self, line):
        parts = line.split()
        for part in parts:
            if part in self.PERIODIC_TABLE:
                return part
        return None

if __name__ == "__main__":
    input_dir = 'data/temp/split_db'
    valid_output_dir = 'data/temp/valid_files'
    invalid_output_dir = 'data/temp/invalid_files'
    allowed_elements = ["H", "C", "O", "N", "F", "P", "S", "Cl", "Se", "Br", "I", "Si"]
    
    molecule_filter = MoleculeElementFilter(input_dir, valid_output_dir, invalid_output_dir, allowed_elements)
    molecule_filter.filter_molecules()