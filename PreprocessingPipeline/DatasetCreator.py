import os
import numpy as np
import tensorflow as tf
from glob import glob
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import hashlib
from pathlib import Path
import re

@dataclass
class DatasetConfig:
    """Configuration for dataset creation and validation"""
    max_file_size: int = 1024 * 1024 * 1024  # 1GB
    min_dimensions: Tuple[int, int] = (1, 1)
    max_dimensions: Tuple[int, int] = (10000, 10000)
    allowed_dtypes: List[np.dtype] = None
    shuffle_buffer_size: int = 10000
    
    def __post_init__(self):
        if self.allowed_dtypes is None:
            self.allowed_dtypes = [np.float32, np.float64]

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DatasetCreator:
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.logger = self._setup_logger()
        self.dataset_stats = {}  # Store statistics about created datasets
        
    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger('DatasetCreator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _validate_numpy_array(self, arr: np.ndarray, file_path: str) -> None:
        """Validate numpy array dimensions and dtype"""
        if arr.dtype not in self.config.allowed_dtypes:
            raise DataValidationError(
                f"Invalid dtype {arr.dtype} in {file_path}. "
                f"Allowed dtypes: {self.config.allowed_dtypes}"
            )
        
        if len(arr.shape) != 2:
            raise DataValidationError(
                f"Invalid dimensions {arr.shape} in {file_path}. "
                "Expected 2D array."
            )
            
        if (arr.shape[0] < self.config.min_dimensions[0] or
            arr.shape[1] < self.config.min_dimensions[1] or
            arr.shape[0] > self.config.max_dimensions[0] or
            arr.shape[1] > self.config.max_dimensions[1]):
            raise DataValidationError(
                f"Invalid dimensions {arr.shape} in {file_path}. "
                f"Expected dimensions between {self.config.min_dimensions} "
                f"and {self.config.max_dimensions}"
            )

    def get_file_pairs(self, directory: str) -> List[Tuple[str, str]]:
        """
        Get valid pairs of numpy files from directory.
        Expects files named like 'entry_XXXXX.YY_sequence.npy'
        Creates all possible pairs including self-pairs.
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        # Get all .npy files
        all_files = list(directory_path.glob('*.npy'))
        if not all_files:
            raise FileNotFoundError(f"No .npy files found in: {directory}")

        # self.logger.info(f"Found {len(all_files)} .npy files in directory")
        
        # Extract base names (everything before the sequence number)
        file_groups: Dict[str, List[Path]] = {}
        pattern = re.compile(r'(entry_\d+)\.\d+_sequence\.npy$')
        
        # First pass: count versions per molecule to validate dataset size
        molecule_versions = {}
        for file in all_files:
            match = pattern.match(file.name)
            if match:
                base_name = match.group(1)
                molecule_versions[base_name] = molecule_versions.get(base_name, 0) + 1
        
        # Calculate expected total pairs
        total_expected_pairs = sum(versions * versions for versions in molecule_versions.values())
        # self.logger.info(f"Dataset statistics:")
        # self.logger.info(f"Number of unique molecules: {len(molecule_versions)}")
        # for molecule, versions in molecule_versions.items():
            # self.logger.info(f"  {molecule}: {versions} versions -> {versions * versions} pairs")
        # self.logger.info(f"Expected total pairs: {total_expected_pairs}")
        
        # Store statistics for later validation
        self.dataset_stats = {
            'molecule_versions': molecule_versions,
            'expected_total_pairs': total_expected_pairs,
            'unique_molecules': len(molecule_versions)
        }
        
        # Second pass: create pairs
        for file in all_files:
            match = pattern.match(file.name)
            if match:
                base_name = match.group(1)
                file_groups.setdefault(base_name, []).append(file)
                self.logger.debug(f"Matched file {file.name} to base name {base_name}")
            else:
                self.logger.warning(f"File {file.name} doesn't match expected pattern")

        # Create all possible pairs within each group, including self-pairs
        file_pairs = []
        group_hashes = {}
        actual_pairs_created = 0
        
        for base_name, files in file_groups.items():
            sorted_files = sorted(files)
            expected_pairs = len(sorted_files) * len(sorted_files)
            
            # Log group information
            #self.logger.info(f"Processing group {base_name} with {len(sorted_files)} files")
            
            # Compute hashes for all files in the group
            for file in sorted_files:
                group_hashes[str(file)] = self._compute_file_hash(str(file))
            
            # Create ALL pairs including self-pairs
            pairs_for_group = 0
            for file1 in sorted_files:
                for file2 in sorted_files:
                    file_pairs.append((str(file1), str(file2)))
                    pairs_for_group += 1
                    
            actual_pairs_created += pairs_for_group
            
            # Validate pairs for this molecule
            if pairs_for_group != expected_pairs:
                raise DataValidationError(
                    f"Invalid number of pairs for {base_name}. "
                    f"Expected {expected_pairs}, got {pairs_for_group}"
                )
            
            # self.logger.info(f"Created {pairs_for_group} pairs for {base_name}")

        # Final validation
        if actual_pairs_created != total_expected_pairs:
            raise DataValidationError(
                f"Total pairs mismatch. Expected {total_expected_pairs}, "
                f"got {actual_pairs_created}"
            )

        self.file_hashes = group_hashes
        # self.logger.info(f"Successfully created {len(file_pairs)} total pairs")
        return file_pairs

    def load_numpy_pair(self, file_paths: Tuple[str, str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load and validate a pair of numpy arrays"""
        file1, file2 = file_paths
        try:
            # Check file sizes
            for file_path in (file1, file2):
                if os.path.getsize(file_path) > self.config.max_file_size:
                    raise DataValidationError(f"File too large: {file_path}")

            # Load arrays
            arr1, arr2 = np.load(file1), np.load(file2)
            
            # Validate arrays
            self._validate_numpy_array(arr1, file1)
            self._validate_numpy_array(arr2, file2)
            
            return arr1, arr2
            
        except Exception as e:
            self.logger.error(f"Error loading files {file1} or {file2}: {str(e)}")
            return None

    def create_and_save_dataset(
        self,
        directory: str,
        save_path: str,
        dataset_name: str = "dataset",
        shuffle: bool = True,
        num_parallel_calls: int = None
    ) -> tf.data.Dataset:
        """Create and save an unbatched TensorFlow dataset with validation"""
        if num_parallel_calls is None:
            num_parallel_calls = os.cpu_count() or 1

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Get file pairs with validation
        file_pairs = self.get_file_pairs(directory)
        
        # Save dataset statistics
        stats_path = save_path / 'dataset_stats.txt'
        with open(stats_path, 'w') as f:
            f.write(f"Dataset Name: {dataset_name}\n")
            f.write(f"Creation Time: {tf.timestamp()}\n")
            f.write(f"Source Directory: {directory}\n")
            f.write(f"Number of unique molecules: {self.dataset_stats['unique_molecules']}\n")
            f.write("Molecules and their versions:\n")
            for molecule, versions in self.dataset_stats['molecule_versions'].items():
                f.write(f"  {molecule}: {versions} versions -> {versions * versions} pairs\n")
            f.write(f"Total expected pairs: {self.dataset_stats['expected_total_pairs']}\n")
        
        def data_generator():
            with ThreadPoolExecutor(max_workers=num_parallel_calls) as executor:
                futures = []
                for pair in file_pairs:
                    futures.append(executor.submit(self.load_numpy_pair, pair))
                
                pairs_processed = 0
                for future in tqdm(futures, desc="Processing pairs"):
                    result = future.result()
                    if result is not None:
                        pairs_processed += 1
                        yield result
                
                # Validate final dataset size
                if pairs_processed != self.dataset_stats['expected_total_pairs']:
                    raise DataValidationError(
                        f"Dataset size mismatch. Expected {self.dataset_stats['expected_total_pairs']}, "
                        f"processed {pairs_processed} pairs"
                    )

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.float32)
            )
        )

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.config.shuffle_buffer_size,
                reshuffle_each_iteration=True
            )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Configure distribution strategy
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        dataset = dataset.with_options(options)
        
        # Save dataset
        tf.data.Dataset.save(dataset, str(save_path))
        
        # Final validation
        self._verify_saved_dataset(save_path, expected_pairs=self.dataset_stats['expected_total_pairs'])
        
        return dataset

    def _verify_saved_dataset(self, save_path: Path, expected_pairs: int) -> None:
        """Verify the integrity and size of the saved dataset"""
        try:
            loaded_dataset = tf.data.Dataset.load(str(save_path))
            
            # Count actual elements in dataset
            element_count = 0
            for _ in loaded_dataset:
                element_count += 1
                
            if element_count != expected_pairs:
                raise DataValidationError(
                    f"Saved dataset size mismatch. Expected {expected_pairs}, "
                    f"got {element_count} elements"
                )
                
           #  self.logger.info(f"Dataset verification successful. Contains {element_count} pairs")
            
        except Exception as e:
            self.logger.error(f"Dataset verification failed: {str(e)}")
            raise

    @classmethod
    def load_saved_dataset(
        cls,
        save_path: str,
        verify_integrity: bool = True,
        shuffle: bool = False,
        shuffle_buffer_size: Optional[int] = None
    ) -> tf.data.Dataset:
        """Load a saved dataset with optional integrity verification and shuffling"""
        save_path = Path(save_path)
        if not save_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {save_path}")
            
        try:
            dataset = tf.data.experimental.load(str(save_path))
            
            if shuffle:
                buffer_size = shuffle_buffer_size or 10000
                dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
                
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            if verify_integrity:
                # Verify dataset structure
                for element in dataset.take(1):
                    if len(element) != 2:
                        raise DataValidationError("Invalid dataset structure")
                        
            return dataset
            
        except Exception as e:
            cls._setup_logger().error(f"Error loading dataset: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    creator = DatasetCreator()
    creator.load_saved_dataset()


# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.