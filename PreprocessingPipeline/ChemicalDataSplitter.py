import os
import re
from tqdm import tqdm


'''The ChemicalDataSplitter class takes in a long input text file - i.e. from PubChem - and splits all the entries 
   into single text files in the output directory.'''

class ChemicalDataSplitter:

    def __init__(self, input_file, output_directory):
        self.input_file = input_file
        self.output_directory = output_directory

    def extract_db_id(self, entry):
        # Extract ID from format '> <number>'
        nmredata_match = re.search(r'> \s*(\d+)', entry)
        if nmredata_match:
            return nmredata_match.group(1)
        
        # Extract ID from 'nmrshiftdb2 <number>'
        nmrshiftdb2_match = re.search(r'nmrshiftdb2\s+(\d+)', entry)
        if nmrshiftdb2_match:
            return nmrshiftdb2_match.group(1)
        
        # Extract ID from alternative formats (e.g., numeric-only lines)
        alternative_match = re.search(r'^\s*(\d+)\s*$', entry, re.MULTILINE)
        if alternative_match:
            return alternative_match.group(1)
        
        # Handle other cases (e.g., '-OEChem-...' format)
        oec_chem_match = re.search(r'-OEChem-(\d+)', entry)
        if oec_chem_match:
            return oec_chem_match.group(1)
        
        # If no pattern matches, return None
        return None


    def split_file(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        with open(self.input_file, 'r') as file:
            content = file.read()

        entries = content.split('$$$$')

        # Initialize tqdm progress bar
        pbar = tqdm(total=len(entries), desc="Splitting Database")

        for entry in entries:
            entry = entry.strip()

            if entry:  # Ignore empty entries
                db_id = self.extract_db_id(entry)

                if db_id:
                    filename = f'entry_{db_id}.txt'
                    with open(os.path.join(self.output_directory, filename), 'w') as output_file:
                        output_file.write(entry)
                else:
                    print(f"Warning: Could not validate DB_ID for an entry. Entry {entry} was skipped.")

                # Update progress bar
                pbar.update(1)

        # Close progress bar
        pbar.close()

        print(f"Splitting complete. Files created in {self.output_directory}")

# Example usage:
# splitter = ChemicalDataSplitter('input.txt', 'output_directory')
# splitter.split_file()