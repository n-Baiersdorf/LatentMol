import os
import shutil
from tqdm import tqdm

"""The 'M-Entries' are a thing in Moltables. They include additional information such as Charges, non-standart
   Isotopes and stuff like that, if present in the molecule. All moltables end with 'M  END'. By filtering any
   molecules with special M Entries, the data gets more uniform, for that the Preprocessing Pipeline is not designed
   to handel these special cases. Later this could change."""

class M_EntrySorter:
    def __init__(self, input_directory, output_directory_chg, output_directory_iso, output_directory_other, output_directory_end):
        self.input_directory = input_directory
        self.output_directory_chg = output_directory_chg
        self.output_directory_iso = output_directory_iso
        self.output_directory_other = output_directory_other
        self.output_directory_end = output_directory_end

        # Create output directories if they do not exist
        os.makedirs(self.output_directory_chg, exist_ok=True)
        os.makedirs(self.output_directory_iso, exist_ok=True)
        os.makedirs(self.output_directory_other, exist_ok=True)
        os.makedirs(self.output_directory_end, exist_ok=True)

    def organize_files(self):
        # List all .txt files in the input directory
        files = [f for f in os.listdir(self.input_directory) if f.endswith('.txt')]
        
        # Use tqdm to show progress
        for filename in tqdm(files, desc="special Molecule-Filter", unit="file"):
            file_path = os.path.join(self.input_directory, filename)
            try:
                with open(file_path, 'r') as file:
                    content = file.readlines()
                    
                    # Initialize flags
                    has_chg = False
                    has_iso = False
                    has_other_m_entries = False
                    
                    # Check for the start of the MDL table
                    in_mdl_table = False
                    
                    for line in content:
                        # Start processing MDL table when we encounter "V2000"
                        if "V2000" in line:
                            in_mdl_table = True

                        if in_mdl_table:
                            # Check for other M entries with two spaces
                            if line.startswith("M  "):
                                if "CHG" in line:
                                    has_chg = True
                                elif "ISO" in line:
                                    has_iso = True
                                elif any(keyword in line for keyword in ["ALS", "APO", "RAD", "RGP", 
                                                                           "LOG", "LIN", "SUB", 
                                                                           "UNS", "RBC", "STY", 
                                                                           "SST", "SCN", "SAL"]):
                                    has_other_m_entries = True

                        # Continue processing until we reach the end of the MDL table
                        if in_mdl_table and line.startswith(">"):
                            break
                    
                    # Determine where to copy the file based on the flags
                    if not (has_chg or has_iso or has_other_m_entries):
                        shutil.copy(file_path, self.output_directory_end)  # Only contains M END
                    elif has_chg:
                        shutil.copy(file_path, self.output_directory_chg)
                    elif has_iso and not has_chg:  # Only ISO and not CHG
                        shutil.copy(file_path, self.output_directory_iso)
                    elif has_other_m_entries:  # Other M entries that are not CHG or ISO
                        shutil.copy(file_path, self.output_directory_other)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

if __name__ == "__main__":
    # Example usage:
    input_dir = 'data/temp/split_db'
    output_dir_chg = 'data/temp/M/with_CHG'
    output_dir_iso = 'data/temp/M/with_ISO'
    output_dir_other = 'data/temp/M/with_Other_M'
    output_dir_end = 'data/temp/M/with_M_END'

    organizer = M_EntrySorter(input_dir, output_dir_chg, output_dir_iso, output_dir_other, output_dir_end)
    organizer.organize_files()

# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.