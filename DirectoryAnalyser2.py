import os
import re

def extract_numbers_from_filenames(directory):
    numbers = []
    for filename in os.listdir(directory):
        # Extract numbers from the filename using regex
        found_numbers = re.findall(r'\d+', filename)
        # Convert the found numbers to integers and add to the list
        numbers.extend(map(int, found_numbers))
    return numbers

directory_path = 'your_directory_path'
numbers = extract_numbers_from_filenames(directory_path)
print(numbers)
