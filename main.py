from PreprocessingPipeline.split_data_entries import ChemicalDataSplitter
from PreprocessingPipeline.sort_molecules_for_viability_via_element_analysis import MoleculeFilter
from PreprocessingPipeline.JSONtoNPYConverter import JsonToNpyConverter
from PreprocessingPipeline.MEntrySorter import NMRShiftDBOrganizer
from PreprocessingPipeline.MOL_to_SEQUENCE import MoleculeProcessor, molToSequenceFunction
from PreprocessingPipeline.split_data_entries import ChemicalDataSplitter
from PreprocessingPipeline.split_atoms_and_bonds import JsonSplitter
from PreprocessingPipeline.AtomBondMerger import JSONListMerger
from PreprocessingPipeline.LengthSorter import MoleculeFileSorter
from PreprocessingPipeline.DirectoryAnalyzer import DirectoryAnalyzer
from PreprocessingPipeline.LengthFilter import LengthFilterer
from PreprocessingPipeline.DatasetCreator import DatasetCreator
from PreprocessingPipeline.augmentation.shuffle_for_augmentation import MoleculeSampleShuffler
from PreprocessingPipeline.antisplit_molecules import TextFileCombiner
from misc.download_data import download_and_extract_pubchem_compounds
# from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import os
import tensorflow as tf
import wandb

temp = "temporary"
data = "data"
data_source = "data/src"


""" This is the Main Script. From here everything else will be called and managed..."""
ATOM_DIMENSION = 12 # Number of atom-Constants
MAX_BONDS = 4 # Maximale Anzahl von Bindungen Oktett-regel
MOL_MAX_LENGTH = 200 # Parameter zum Einstellen der maximalen Länge von Molekülen
MAX_PERMUTATIONS = 5

class Verarbeiter():
    def __init__(self, ):
        t = 1
        print(t)

    def prepare_data(self, db_file):
        base_dir = "data"

        self._prepare_data_step_I(db_file_directory, base_dir)

        analyzer = DirectoryAnalyzer("data/temp/V_original_data")
        analyzer.scan_directory()
        data = analyzer.get_data()

        
        for i in data.numbers:
            print("----------------------------------------")
            print(f"Starte mit verarbeitung von Nummer: {i} von {MOL_MAX_LENGTH}")

            self._prepare_data_step_II(base_dir, i)
    

    def _prepare_data_step_I(self, db_file_directory, base_dir):
        # Process all the molecule files from the database file directory
        
        counter = 0

        for filename in os.listdir(db_file_directory):
            if filename.endswith('.sdf'):  # Check for .sdf files
                
                counter = counter + 1

                file_path = os.path.join(db_file_directory, filename)

                split_dir = os.path.join(base_dir,f'temp/{filename}/split_db')


                # 1: Split each molecule from the current .sdf file
                print(f"Processing file: {filename}")
                splitter = ChemicalDataSplitter(file_path, split_dir)
                splitter.split_file()

        
                
                # 2: Filter out too long molecules
                filtered_split_dir = os.path.join(base_dir,f'temp/{filename}/II_split_db_filtered_lengthwise')
                sorter = LengthFilterer(max_length=MOL_MAX_LENGTH)
                sorter.filter_and_copy(split_dir, filtered_split_dir)

                # 3: Filter molecules for valid elements
                valid_output_dir = os.path.join(base_dir,f'temp/{filename}/II_valid_files')
                invalid_output_dir = os.path.join(base_dir, f'temp/{filename}/obsolete/invalid_files')
                allowed_elements = ["H", "C", "O", "N", "F", "P", "S", "Cl", "Br", "I"]
                
                molecule_filter = MoleculeFilter(filtered_split_dir, valid_output_dir, invalid_output_dir, allowed_elements)
                molecule_filter.filter_molecules()

                # 4: filtering for isotopes...
                output_dir_chg = os.path.join(base_dir, f'temp/{filename}/obsolete/with_CHG')
                output_dir_iso = os.path.join(base_dir,f'temp/{filename}/obsolete/with_ISO')
                output_dir_end = os.path.join(base_dir,f'temp/{filename}/obsolete/with_M_END') # doesn't work! To be removed... (does not impact functionality)

                significant_output = os.path.join(base_dir, f'temp/{filename}/IV_with_Other_M')
                
                organizer = NMRShiftDBOrganizer(valid_output_dir, output_dir_chg, output_dir_iso, output_dir_end, significant_output)
                organizer.organize_files()


                

                # 5: Sort in lengths
                sorter = MoleculeFileSorter()
                output_dir = os.path.join(base_dir, f'temp/{filename}/V_original_data')
                sorter.sort_files(significant_output, output_dir)



    def _prepare_data_step_II(self, base_dir1, number):

        input_dir = os.path.join(base_dir1, "temp/V_original_data",str(number))

        # 0: 
        output_dir = f"data/temp_II/original_data/molecules_{number}.txt"
        combiner = TextFileCombiner(input_dir, output_dir)
        combiner.run()

        # 1: apply Augmentation
        output_directory = f"data/temp_II/augmented_data/length_{number}"  # Specify your desired output directory here
        
        MoleculeSampleShuffler.shuffle_molecules(output_dir, output_directory, MAX_PERMUTATIONS)
        
        # 2: reformate MOL-Format to Sequence        
        input_file = os.path.join(output_directory, f"shuffled_molecules_{number}.txt")
        output_dir = f"data/Output_{MAX_PERMUTATIONS}"
        molToSequenceFunction(input_file, output_dir, number)

        

    def _prepare_data_step_III():
        exit
        

    def _prepare_data_step_IV():
        # 5: Create Datasets for length
        save_path_dataset = os.path.join("subdatasets", f"length_{number}")
        
        creator = DatasetCreator()
        dataset = creator.create_and_save_dataset(
            npy_output,
            save_path_dataset,
            dataset_name=f"dataset_{number}",
            shuffle=True
        )


if __name__ == "__main__":

    # Präambel: Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print("Name:", gpu.name, "  Type:", gpu.device_type)
    else:
        print("No GPUs detected")

    # Script Beginn
    
    '''download_and_extract_pubchem_compounds(
        output_dir="data/raw/pubchem_compounds",  # Directory to store extracted files
        start=1,                         # Start of compound range
        end=500000,                     # End of compound range (adjust as needed)
        step=500000                      # Step size for each file batch
    )'''
    
    #db_file = "Compound_000000001_000500000.sdf"
    #db_file = "nmrshiftdb2.nmredata.sd"
    db_file_directory = "data/src/pubchem_compounds"
    
    this = Verarbeiter()
    this.prepare_data(db_file_directory)
    




