import re
from tqdm import tqdm

MULTIPLICITY_MAP = {
    'S': 1,  # Singlet
    'D': 2,  # Doublet
    'T': 3,  # Triplet
    'Q': 4,  # Quartet
    'M': 5,  # Multiplet (if it appears, or you can extend this map)
    # etc. if needed
}

# Map numeric portions like "1H", "13C", "19F" etc. to spelled-out strings
SPECTRUM_NAME_MAP = {
    '1H': 'H',
    '13C': 'C',
    '19F': 'F',
    '15N': 'N',
    '17O': 'O',
    '31P': 'P'
    # Extend as desired to handle other cases, e.g. '17O': 'seventeenO' etc.
}

class SpectrumToSequence:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.stats = {}  # track occurrences per raw spectrum type

    def parse_file(self):
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return f.read()

    def convert_multiplicity(self, mult_char):
        return MULTIPLICITY_MAP.get(mult_char.upper(), 9)

    def spelled_spectrum_name(self, raw_type):
        """
        Convert e.g. '13C' => 'thirteenC'. If not in map, attempt basic fallback.
        """
        if raw_type in SPECTRUM_NAME_MAP:
            return SPECTRUM_NAME_MAP[raw_type]
        # fallback tries to replace digits with underscores or just strip digits
        digits = re.findall(r'\d+', raw_type)
        if digits:
            spelled_digits = ''.join(digits)
            return raw_type.replace(''.join(digits), spelled_digits)
        return raw_type

    def parse_spectrum_lines(self, lines):
        """
        Lines might be something like '77.2;0.0D;7|83.6;0.0S;6|...'.
        We split by '|' and parse each chunk into [float, float, intMultiplicity, int].
        Now includes SOS and EOS tokens.
        """
        result = []
        # Add SOS token
        result.append([-555.0, -555.0, -555, -555])

        pieces = lines.strip('|').split('|')

        for piece in pieces:
            part = piece.strip()
            if not part:
                continue

            sub_parts = part.split(';')
            if len(sub_parts) < 3:
                continue

            shift_str = sub_parts[0].strip()
            second_str = sub_parts[1]
            last_str = sub_parts[2].strip()

            match_float = re.findall(r'^[0-9.]+', second_str)
            if match_float:
                try:
                    second_float_val = float(match_float[0])
                except ValueError:
                    # Skip this piece if it doesn't convert cleanly
                    continue
                remainder = second_str[len(match_float[0]):]
            else:
                second_float_val = 0.0
                remainder = second_str

            multiplicity_char = remainder[:1].upper() if remainder else 'S'

            try:
                shift_val = float(shift_str)
            except ValueError:
                shift_val = 0.0

            try:
                last_int = int(last_str)
            except ValueError:
                last_int = 0

            result.append([
                shift_val,
                second_float_val,
                self.convert_multiplicity(multiplicity_char),
                last_int
            ])

        # Add EOS token
        result.append([-999.0, -999.0, -999, -999])
        return result

    def process_entry(self, entry_text, current_id):
        """
        Process a single entry block, now using ID in the spectrum names.
        current_id should be passed without the ### symbols.
        """
        lines = entry_text.split('\n')
        output_lines = []
        i = 0
        spectrum_index_map = {}

        while i < len(lines):
            line = lines[i].strip()

            # The pattern below might need a proper regex call, e.g.:
            # match = re.match(r'^>\s*(\S+)', line)
            # if match:
            if line.startswith('>'):
                match = re.match(r'^>\s*(\S+)', line)
                if match:
                    spectrum_type_raw = match.group(1)
                    spelled_type = self.spelled_spectrum_name(spectrum_type_raw)
                    self.stats[spectrum_type_raw] = self.stats.get(spectrum_type_raw, 0) + 1

                    if spelled_type not in spectrum_index_map:
                        spectrum_index_map[spelled_type] = 1
                    else:
                        spectrum_index_map[spelled_type] += 1

                    j = i + 1
                    data_lines = []

                    while j < len(lines):
                        next_line = lines[j].strip()
                        if next_line.startswith('>') or not next_line:
                            break
                        data_lines.append(next_line)
                        j += 1

                    combined = '|'.join(data_lines)
                    parsed = self.parse_spectrum_lines(combined)
                    new_spectrum_title = f"{current_id}_spectrum_{spelled_type}_{spectrum_index_map[spelled_type]}"
                    output_lines.append(f"{new_spectrum_title} = {parsed}")

                    i = j
                    continue

            i += 1

        return "\n".join(output_lines)

    def process_all(self):
        raw_text = self.parse_file()
        split_pattern = r'(###\d+###)'
        splitted = re.split(split_pattern, raw_text)
        results = []
        pbar_range = range(1, len(splitted), 2)

        for idx in tqdm(pbar_range, desc="Processing entries"):
            current_id = splitted[idx].strip('#')  # Remove ### from ID
            if idx + 1 < len(splitted):
                entry_text = splitted[idx + 1]
            else:
                entry_text = ""
            processed = self.process_entry(entry_text, current_id)
            if processed.strip():
                results.append(processed)

        final_text = "\n".join(results)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)

        print("\nSummary of Spectrum Counts:")
        for k, v in sorted(self.stats.items()):
            print(f" {k}: {v} occurrences")

def main():
    # Example usage:
    input_path = 'data/nmr_temp/test2.txt'
    output_path = 'data/nmr_temp/test_output2.txt'
    parser = SpectrumToSequence(input_path, output_path)
    parser.process_all()

if __name__ == '__main__':
    main()


    
# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.