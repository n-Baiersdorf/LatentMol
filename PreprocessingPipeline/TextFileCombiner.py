import os
import re


"""This Script takes in a directory with many small text files each containing but one moltable. It then combines
   them into one quite long text file.
   It uses '###ID###' as a divider where ID is the ID of the molecule in the database."""


class TextFileCombiner:
    def __init__(self, input_directory, output_file):
        self.input_directory = input_directory
        self.output_file = output_file
        self.divider_template = "###{}###"
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_files(self):
        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            for filename in sorted(os.listdir(self.input_directory)):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.input_directory, filename)
                    id_match = re.search(r'entry_(\d+)\.txt', filename)
                    if id_match:
                        id_num = id_match.group(1)
                        padded_id = id_num.zfill(8)  # Pad the ID to 8 digits
                        
                        divider = self.divider_template.format(padded_id)
                        outfile.write(f"{divider}\n")
                        
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                        
                        outfile.write("\n")  # Additional newline after each entry

    def run(self):
        self.process_files()
        print(f"All files have been combined into {self.output_file}.")

if __name__ == "__main__":
    # Usage of the class with a specified output directory
    combiner = TextFileCombiner("data_II/original_data/10", "chosen_directory/ausgabe.txt")
    combiner.run()
