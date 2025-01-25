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
from PreprocessingPipeline.SpectraExtracta import extract_spectra
from PreprocessingPipeline.SpectrumToSequence import SpectrumToSequence
from nmr_sequence_dataset_creation.PairCreator import pair
from nmr_sequence_dataset_creation.DatasetCreator import create_dataset
from nmr_sequence_dataset_creation.BigDatasetCreator import load_datasets_from_directory, CustomCombinedDataset
import os
import re


'''This Main Script Downloads PubChem Data in MolTable format and transforms it into LatentMol's relativistic
   graph sequences for deep learning with Transformers.
   For this it works through all the steps calling external scripts.'''



ATOM_DIMENSION = 12 # Number of atom-Constants --> depends on the Dictionary 
MAX_BONDS = 4 # max number of bonds: default is 4 --> Octet rule (As each bond contains two electrons [Hydrogen gets handeled specially])
MOL_MAX_LENGTH = 35 # max length of molecules: the "length" refers to the number of atoms notated in the moltable --> many implicit Hydrogens are not counted
MAX_PERMUTATIONS = 10                                   # These are the augmented Versions --> Set it to the value that you want. Pobably something line 1000 would be appropriate. Although that would result in huge amounts of samples.

class Verarbeiter():
    def __init__(self, ):
        t = 1
        print(t)

      
    

    def prepare_raw_data(self, db_file_directory):
        # Process all the molecule files from the database file directory
        base_dir = "data"


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

    def prepare_molecules(self, base_dir1, number):
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
       
 
    def make_sequence_data(self):
        numbers = self.extract_numbers_from_filenames("data/temp_II/augmented_data") 
        
        for i in numbers:
            print("----------------------------------------")
            print(f"Start with MolToSequence conversion of the molecules with length: {i} out of {MOL_MAX_LENGTH}")

            input_file = os.path.join(f"data/temp_II/augmented_data/shuffled_molecules_{i}.txt")
            output_dir = f"molsequences"   # f"data/Output_{MAX_PERMUTATIONS}"
            molToSequenceFunction(input_file, output_dir, i)


            config = {
                    "en_diff_max": 1.79,
                    "use_fixed_ch_diff": True,
                    "default_ch_endiff": 0.25
                }

            test_number = 1
            molToSequenceFunction(input_file, output_dir, test_number, config=config)


    def make_nmr_dataset(self):
        numbers = self.extract_numbers_from_filenames("data/temp_II/augmented_data") 
        for number in numbers:        
            self._make_nmr_subdataset(number)
        
            
        datasets = load_datasets_from_directory("data/nmr_temp/length_datasets")
        combined_dataset = CustomCombinedDataset(datasets)
        
        # Dataset speichern
        combined_dataset.save_dataset('data/datasets/MixedNMRSpectrumDataset.pt')




    def _make_nmr_subdataset(self, number):
        # 1. Extract Spectra from raw data
        raw_molecule_data_path = f"data/temp_II/original_data/molecules_{number}.txt"
        raw_spectra_path = f"data/nmr_temp/raw_spectra/spectra_{number}.txt"
        spectra_dir_path = f"data/nmr_temp/raw_spectra/"
        extract_spectra(raw_molecule_data_path, raw_spectra_path, spectra_dir_path)

        # 2. Reformat Spectra to Sequences
        spectra_path = f'data/nmr_temp/sequence_spectra/seqspectra_{number}.txt'
        parser = SpectrumToSequence(raw_spectra_path, spectra_path)
        parser.process_all()

        # 3. Produce ID Pair Dictionary
        matched_dict_name = f"data/nmr_temp/matchings/matchings_{number}.json"
        molecule_sequence_data_path = f"data/molsequences/Länge_{number}.json"
        pair(matched_dict_name,  molecule_sequence_data_path, spectra_path)

        # 4. Create and Save Subdataset based on length
        save_path=f"data/nmr_temp/length_datasets/nmr_dataset_{number}.pt"
        input_sequences = f"data/molsequences/Länge_{number}.json"
        matched_labels = matched_dict_name
        create_dataset(input_sequences, matched_labels, save_path)







def create_directory_structure():
    # List of directories to create
    directories = ["data", "data/datasets", "data/src", "data/src/pubchem_compounds", 'data/nmr_temp/sequence_spectra', 'data/nmr_temp/matchings/', "data/nmr_temp/length_datasets/"]
    
    # Loop through each directory and create it if it doesn't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)



if __name__ == "__main__":
    # Setup
    create_directory_structure()



    db_file_directory = "NMRShiftDB2"
    
    this = Verarbeiter()
    '''this.prepare_raw_data(db_file_directory)  # prepare the raw data




    analyzer = DirectoryAnalyzer("data/temp/V_original_data") 
    analyzer.scan_directory()
    data = analyzer.get_data()

    base_dir = "data"
    for i in data.numbers:
        print("----------------------------------------")
        print(f"Starte mit verarbeitung von Nummer: {i} von {MOL_MAX_LENGTH}")

        this.prepare_molecules(base_dir, i)
    '''


    # this.make_sequence_data()                # convert it into the LatentMol Sequence format
    this.make_nmr_dataset()


# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.