import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Define a custom PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        """
        Initializes the dataset.

        :param inputs: Dictionary of input sequences (each value is a list of lists, treated as one sequence).
        :param labels: Dictionary of corresponding target sequences.
        """
        self.inputs = []
        self.labels = []

        # Erwartete Länge der Eingabesequenzen
        expected_length = 26

        # Iterate over each input ID
        for key, input_sequences in tqdm(inputs.items(), desc="Processing Inputs"):
            if key in labels:
                try:
                    # Prüfen, ob jede Sequenz die erwartete Länge hat
                    input_tensor = torch.tensor(input_sequences, dtype=torch.float32)
                    if input_tensor.shape[1] != expected_length:
                        print(f"Skipping input with invalid length: {input_tensor.shape}")
                        continue

                    # Each label (list of lists) is processed individually
                    for label in labels[key]:
                        label_tensor = torch.tensor(label, dtype=torch.float32)
                        self.inputs.append(input_tensor)
                        self.labels.append(label_tensor)

                except Exception as e:
                    print(f"Error processing key {key}: {e}")
                    continue

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def create_dataset(input_sequences="data/output_molsequences_1/Länge_4.json",
                   matched_labels="matched_data.json", 
                   save_path="custom_dataset.pt"):
    # Load input sequences from JSON file
    print("Loading input sequences...")
    with open(input_sequences, "r") as f:
        input_sequences = json.load(f)

    # Validate input sequences
    print("Validating input sequences...")
    expected_length = 26
    for key, sequences in input_sequences.items():
        for seq in sequences:
            if len(seq) != expected_length:
                print(f"Invalid input length detected for key {key}: {len(seq)}")
    
    # Load labels (matched data) from the dictionary file
    print("Loading matched data (labels)...")
    with open(matched_labels, "r") as f:
        matched_labels = json.load(f)

    # Create the PyTorch Dataset
    print("Creating PyTorch Dataset...")
    dataset = CustomDataset(input_sequences, matched_labels)

    # Save the Dataset using torch.save
    dataset_path = save_path
    print(f"Saving dataset to {dataset_path}...")
    torch.save(dataset, dataset_path)


def test_dataset(dataset_path):
    # Example: Load and verify the Dataset
    print("Loading dataset to verify...")
    loaded_dataset = torch.load(dataset_path)

    # Print a sample
    print("Sample from the dataset:")
    for i in range(3):  # Display 3 samples
        sample_input, sample_label = loaded_dataset[i]
        print(f"Input shape: {sample_input.shape}, Input data: {sample_input}")
        print(f"Label shape: {sample_label.shape}, Label data: {sample_label}")


    print("-----------------")

    x, y=loaded_dataset.__getitem__(5)
    print(f"Eingabe: {x}")
    print(f"Label: {y}")
    print(f"Shapes x and y: {x.shape}  {y.shape}")

if __name__ == "__main__":
    save_path="custom_dataset.pt"
    create_dataset()
    test_dataset(save_path)



# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.