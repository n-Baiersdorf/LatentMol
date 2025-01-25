import sys
import os
import re

def extract_data_from_block(block_lines):
    """
    Given the lines for a single entry (block), extract:
      - The 'Spectrum' sections (like '> <Spectrum 13C 1>' plus subsequent lines
        until the next '> <' or end of block),
      - The 'Solvent' section (the '> <Solvent>' line plus subsequent lines until
        the next '> <' or end of block).
    Returns a list of lines to keep for this block.
    """
    keep_lines = []
    i = 0
    while i < len(block_lines):
        line = block_lines[i]
        # Check if this line starts an interesting section
        if re.match(r'^> <Spectrum\b', line) or re.match(r'^> <Solvent>', line):
            # Keep this line
            keep_lines.append(line)
            i += 1
            # Gather subsequent lines until next '> <' or end of block
            while i < len(block_lines) and not re.match(r'^> <', block_lines[i]):
                keep_lines.append(block_lines[i])
                i += 1
        else:
            i += 1
    return keep_lines

def extract_spectra(input_file, output_file, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Split into blocks by lines of the form ###SomeID###
    # We'll keep track of each block's ID line separately.
    blocks = []
    current_block_lines = []
    current_id_line = None
    
    for line in lines:
        line_stripped = line.strip('\n')
        if line_stripped.startswith('###') and line_stripped.endswith('###'):
            # If there's an existing block, store it before starting a new one
            if current_id_line or current_block_lines:
                blocks.append((current_id_line, current_block_lines))
            # Reset for the new block
            current_id_line = line_stripped
            current_block_lines = []
        else:
            current_block_lines.append(line_stripped)

    # Add the last block if present
    if current_id_line or current_block_lines:
        blocks.append((current_id_line, current_block_lines))

    # Now process each block to extract the needed lines
    with open(output_file, 'w', encoding='utf-8') as out:
        for id_line, block_lines in blocks:
            if id_line:
                out.write(id_line + '\n')  # Keep the ###ID### line
            extracted = extract_data_from_block(block_lines)
            for l in extracted:
                out.write(l + '\n')

if __name__ == '__main__':
    input_file_path = "data/temp_II/original_data/molecules_4.txt"
    output_file_path = "data/nmr_temp/test2.txt"
    output_dir_path = "data/nmr_temp/"

    
    extract_spectra(input_file_path, output_file_path, output_dir_path)


# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.