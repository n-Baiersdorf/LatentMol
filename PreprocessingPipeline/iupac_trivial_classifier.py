import os
import re
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

class ChemicalNameClassifier:
    """
    A class to classify chemical names as IUPAC, trivial, or both by analyzing text files.
    
    The classifier uses a scoring system based on various characteristics of chemical names
    to determine their classification.
    """
    
    def __init__(self, input_directory: str, output_directories: Dict[str, str] = None):
        """
        Initialize the classifier with input and output directory paths and set up logging.
        
        Args:
            input_directory (str): Path to the directory containing chemical name files
            output_directories (Dict[str, str], optional): Dictionary mapping classification
                types to output directory paths. Keys should be 'iupac_only', 'trivial_only',
                and 'both'. If not provided, subdirectories will be created in input directory.
        
        Raises:
            ValueError: If the directory paths are invalid
        """
        self.input_directory = Path(input_directory)
        if not self.input_directory.is_dir():
            raise ValueError(f"Invalid input directory path: {input_directory}")
            
        # Set up output directories
        self.output_directories = {
            'iupac_only': self.input_directory / 'iupac_only',
            'trivial_only': self.input_directory / 'trivial_only',
            'both': self.input_directory / 'both'
        }
        
        if output_directories:
            for key, path in output_directories.items():
                if key not in ['iupac_only', 'trivial_only', 'both']:
                    raise ValueError(f"Invalid output directory key: {key}")
                self.output_directories[key] = Path(path)
            
        # Set up logging
        logging.basicConfig(
            filename='chemical_classifier.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # IUPAC characteristics
        self.iupac_prefixes = {
            'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl', 'hexyl',
            'heptyl', 'octyl', 'nonyl', 'decyl', 'cyclo'
        }
        
        self.iupac_suffixes = {
            'ane', 'ene', 'yne', 'ol', 'al', 'one', 'oic acid',
            'ate', 'ide', 'oxy', 'amino'
        }
        
        # Initialize statistics
        self.stats = {
            'iupac_only': 0,
            'trivial_only': 0,
            'both': 0,
            'errors': 0,
            'total_processed': 0
        }

    def create_subdirectories(self) -> None:
        """Create output directories if they don't exist."""
        for directory in self.output_directories.values():
            directory.mkdir(parents=True, exist_ok=True)

    def classify_name(self, filename: str) -> Tuple[str, Dict[str, int]]:
        """
        Classify a chemical name from a file using a scoring system.
        
        Args:
            filename (str): Name of the file to process
            
        Returns:
            Tuple[str, Dict[str, int]]: Classification and scores dictionary
        """
        try:
            with open(self.input_directory / filename, 'r', encoding='utf-8') as file:
                chemical_name = file.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading file {filename}: {str(e)}")
            raise
            
        scores = {'iupac': 0, 'trivial': 0}
        
        # Rule 1: Check for numbers (IUPAC indicator)
        if re.search(r'\d', chemical_name):
            scores['iupac'] += 2
            
        # Rule 2: Check for semicolon (indicates both)
        if ';' in chemical_name:
            return 'both', {'iupac': 5, 'trivial': 5}
            
        # Rule 3: Check for IUPAC prefixes and suffixes
        for prefix in self.iupac_prefixes:
            if re.search(rf'\b{prefix}[-]?', chemical_name.lower()):
                scores['iupac'] += 1
                
        for suffix in self.iupac_suffixes:
            if chemical_name.lower().endswith(suffix):
                scores['iupac'] += 1
                
        # Rule 4: Analyze length and complexity
        if len(chemical_name) > 30:
            scores['iupac'] += 1
        if len(chemical_name) < 10:
            scores['trivial'] += 1
            
        # Rule 5: Check for IUPAC punctuation
        if '-' in chemical_name:
            scores['iupac'] += 1
        if '(' in chemical_name and ')' in chemical_name:
            scores['iupac'] += 1
            
        # Rule 6: Case sensitivity check
        if chemical_name.islower():
            scores['iupac'] += 1
        if not chemical_name.islower() and not chemical_name.isupper():
            scores['trivial'] += 1
            
        # Determine classification based on scores
        if scores['iupac'] > scores['trivial'] + 2:
            return 'iupac', scores
        elif scores['trivial'] > scores['iupac']:
            return 'trivial', scores
        elif abs(scores['iupac'] - scores['trivial']) <= 2 and scores['iupac'] > 0:
            return 'both', scores
        else:
            return 'trivial', scores

    def classify_files(self) -> None:
        """
        Process all .txt files in the directory and sort them into appropriate subdirectories.
        Shows progress using tqdm.
        """
        self.create_subdirectories()
        
        try:
            txt_files = list(self.input_directory.glob('*.txt'))
            self.logger.info(f"Found {len(txt_files)} .txt files to process")
            
            # Use tqdm to create progress bar
            for file_path in tqdm(txt_files, desc="Processing files", unit="file"):
                try:
                    classification, scores = self.classify_name(file_path.name)
                    
                    # Update statistics
                    self.stats['total_processed'] += 1
                    self.stats[f'{classification}_only' if classification != 'both' else 'both'] += 1
                    
                    # Move file to appropriate subdirectory
                    destination = self.output_directories[f'{classification}_only' if classification != 'both' else 'both'] / file_path.name
                    shutil.move(str(file_path), str(destination))
                    
                    self.logger.info(
                        f"Classified {file_path.name} as {classification} "
                        f"(IUPAC score: {scores['iupac']}, Trivial score: {scores['trivial']})"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path.name}: {str(e)}")
                    self.stats['errors'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error accessing directory: {str(e)}")
            raise

    def generate_report(self) -> str:
        """
        Generate a summary report of the classification results.
        
        Returns:
            str: Formatted report string
        """
        report = [
            "Chemical Name Classification Report",
            "================================",
            f"Total files processed: {self.stats['total_processed']}",
            f"IUPAC names: {self.stats['iupac_only']}",
            f"Trivial names: {self.stats['trivial_only']}",
            f"Both IUPAC and trivial: {self.stats['both']}",
            f"Errors encountered: {self.stats['errors']}",
            f"Success rate: {((self.stats['total_processed'] - self.stats['errors']) / self.stats['total_processed'] * 100):.2f}%",
            "\nOutput Directories:",
            f"IUPAC only: {self.output_directories['iupac_only']}",
            f"Trivial only: {self.output_directories['trivial_only']}",
            f"Both: {self.output_directories['both']}"
        ]
        
        # Write report to file
        report_path = self.input_directory / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
        return '\n'.join(report)


if __name__ == "__main__":
    try:
        # Example of specifying custom output directories
        output_dirs = {
            'iupac_only': 'data/chemnames/classified/iupac',
            'trivial_only': 'data/chemnames/classified/trivial',
            'both': 'data/chemnames/classified/both'
        }
        
        # Initialize the classifier with input directory and custom output directories
        classifier = ChemicalNameClassifier(
            input_directory="data/chemnames/unsorted",
            output_directories=output_dirs
        )
        
        # Process all files in the directory
        classifier.classify_files()
        
        # Generate and print the report
        report = classifier.generate_report()
        print(report)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")