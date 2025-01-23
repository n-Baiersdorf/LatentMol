from PreprocessingPipeline.ChemicalDataSplitter import ChemicalDataSplitter
from PreprocessingPipeline.MoleculeElementFilter import MoleculeElementFilter
from PreprocessingPipeline.M_EntrySorter import M_EntrySorter
from PreprocessingPipeline.MolToSequence import molToSequenceFunction
from PreprocessingPipeline.ChemicalDataSplitter import ChemicalDataSplitter
from PreprocessingPipeline.LengthSorter import MoleculeFileSorter
from PreprocessingPipeline.DirectoryAnalyzer import DirectoryAnalyzer
from PreprocessingPipeline.LengthFilter import LengthFilterer
from PreprocessingPipeline.augmentation.MoleculeSampleShuffler import MoleculeSampleShuffler
from PreprocessingPipeline.TextFileCombiner import TextFileCombiner
from misc.download_data import download_and_extract_pubchem_compounds

import os
import re


'''This Main Script Downloads PubChem Data in MolTable format and transforms it into LatentMol's relativistic
   graph sequences for deep learning with Transformers.
   For this it works through all the steps calling external scripts.'''



ATOM_DIMENSION = 12 # Number of atom-Constants --> depends on the Dictionary 
MAX_BONDS = 4 # max number of bonds: default is 4 --> Octet rule (As each bond contains two electrons [Hydrogen gets handeled specially])
MOL_MAX_LENGTH = 50 # max length of molecules: the "length" refers to the number of atoms notated in the moltable --> many implicit Hydrogens are not counted
MAX_PERMUTATIONS = 5 # These are the augmented Versions --> Set it to the value that you want. Pobably something line 1000 would be appropriate. Although that would result in huge amounts of samples.

class Verarbeiter():
    def __init__(self, ):
        t = 1
        print(t)

    def prepare_raw_data(self, db_file_directory):
        base_dir = "data"

        self._prepare_raw_data_substep_I(db_file_directory, base_dir)

        analyzer = DirectoryAnalyzer("data/temp/V_original_data") 
        analyzer.scan_directory()
        data = analyzer.get_data()

        
        for i in data.numbers:
            print("----------------------------------------")
            print(f"Starte mit verarbeitung von Nummer: {i} von {MOL_MAX_LENGTH}")

            self._prepare_raw_data_substep_II(base_dir, i)
    

    def _prepare_raw_data_substep_I(self, db_file_directory, base_dir):
        # Process all the molecule files from the database file directory
        
        counter = 0

        for filename in os.listdir(db_file_directory):
            if filename.endswith(('.sdf', 'sd', 'txt')):  # Check for .sdf files
                counter = counter + 1
                file_path = os.path.join(db_file_directory, filename)
                split_dir = os.path.join(base_dir,f'temp/split_db')
               
               # 1: Split each molecule from the current .sdf file
                print(f"Processing file: {filename}")
                splitter = ChemicalDataSplitter(file_path, split_dir)
                splitter.split_file()

                # 2: Filter out too long molecules
                filtered_split_dir = os.path.join(base_dir,f'temp/II_split_db_filtered_lengthwise')
                sorter = LengthFilterer(max_length=MOL_MAX_LENGTH)
                sorter.filter_and_copy(split_dir, filtered_split_dir)

                # 3: Filter molecules for valid elements
                valid_output_dir = os.path.join(base_dir,f'temp/II_valid_files')
                invalid_output_dir = os.path.join(base_dir, f'temp/obsolete/invalid_files')
                allowed_elements = ["H", "C", "O", "N", "F", "P", "S", "Cl", "Br", "I"]
                
                molecule_filter = MoleculeElementFilter(filtered_split_dir, valid_output_dir, invalid_output_dir, allowed_elements)
                molecule_filter.filter_molecules()

                # 4: filtering for isotopes...
                output_dir_chg = os.path.join(base_dir, f'temp/obsolete/with_CHG')
                output_dir_iso = os.path.join(base_dir,f'temp/obsolete/with_ISO')
                output_dir_end = os.path.join(base_dir,f'temp/obsolete/with_M_END') # doesn't work! To be removed... (does not impact functionality)

                significant_output = os.path.join(base_dir, f'temp/IV_with_Other_M')
                
                organizer = M_EntrySorter(valid_output_dir, output_dir_chg, output_dir_iso, output_dir_end, significant_output)
                organizer.organize_files()

                # 5: Sort in lengths
                sorter = MoleculeFileSorter()
                output_dir = os.path.join(base_dir, f'temp/V_original_data')
                sorter.sort_files(significant_output, output_dir)

    def _prepare_raw_data_substep_II(self, base_dir1, number):
        input_dir = os.path.join(base_dir1, "temp/V_original_data",str(number))

        # 1: Combine all individual files into one long file based on the file length 
        output_dir = f"data/temp_II/original_data/molecules_{number}.txt"
        combiner = TextFileCombiner(input_dir, output_dir)
        combiner.run()

        # 2: apply Augmentation
        output_directory = f"data/temp_II/augmented_data/"  # Specify your desired output directory here
        MoleculeSampleShuffler.shuffle_molecules(output_dir, output_directory, MAX_PERMUTATIONS)

        
    def extract_numbers_from_filenames(self, directory): 
        numbers = []
        for filename in os.listdir(directory):
            # Extract numbers from the filename using regex
            found_numbers = re.findall(r'\d+', filename)
            # Convert the found numbers to integers and add to the list
            numbers.extend(map(int, found_numbers))
        return sorted(numbers)
       
 
    def _make_sequence_data(self):
        numbers = self.extract_numbers_from_filenames("data/temp_II/augmented_data") 
        
        print("hello world")
        for i in numbers:
            print("----------------------------------------")
            print(f"Start with MolToSequence conversion of the molecules with length: {i} out of {MOL_MAX_LENGTH}")

            input_file = os.path.join(f"data/temp_II/augmented_data/shuffled_molecules_{i}.txt")
            output_dir = f"data/output_molsequences_{MAX_PERMUTATIONS}"   # f"data/Output_{MAX_PERMUTATIONS}"
            molToSequenceFunction(input_file, output_dir, i)


            config = {
                    "en_diff_max": 1.79,
                    "use_fixed_ch_diff": True,
                    "default_ch_endiff": 0.25
                }

            test_number = 1
            molToSequenceFunction(input_file, output_dir, test_number, config=config)



def create_directory_structure():
    # List of directories to create
    directories = ["data", "data/src", "data/src/pubchem_compounds"]
    
    # Loop through each directory and create it if it doesn't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)



if __name__ == "__main__":
    # Setup
    create_directory_structure()

    # Script begin
    download_and_extract_pubchem_compounds(
        output_dir="data/src/pubchem_compounds",  # Directory to store extracted files
        start=1,                         # Start of compound range
        end=500000,                     # End of compound range (adjust as needed)
        step=500000                      # Step size for each file batch
    )
    
    db_file_directory = "data/src/pubchem_compounds/"
    
    this = Verarbeiter()
    this.prepare_raw_data(db_file_directory)  # prepare the raw data
    this._make_sequence_data()                # convert it into the LatentMol Sequence format


# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.
