import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
import os

def transformer_collate_fn(batch):
    # Assume batch contains tuples of (input_seq, target_seq)
    input_sequences = [item[0] for item in batch]
    target_sequences = [item[1] for item in batch]
    
    # Find maximum lengths
    max_input_len = max(seq.size(0) for seq in input_sequences)
    max_target_len = max(seq.size(0) for seq in target_sequences)
    
    # Create padding and masks
    padded_inputs = []
    padded_targets = []
    input_masks = []
    target_masks = []
    
    for input_seq, target_seq in zip(input_sequences, target_sequences):
        # Input padding and mask
        input_len = input_seq.size(0)
        input_padding = max_input_len - input_len
        padded_input = torch.nn.functional.pad(input_seq, (0, input_padding))
        input_mask = torch.ones(max_input_len)
        input_mask[input_len:] = 0
        
        # Target padding and mask
        target_len = target_seq.size(0)
        target_padding = max_target_len - target_len
        padded_target = torch.nn.functional.pad(target_seq, (0, target_padding))
        target_mask = torch.ones(max_target_len)
        target_mask[target_len:] = 0
        
        padded_inputs.append(padded_input)
        padded_targets.append(padded_target)
        input_masks.append(input_mask)
        target_masks.append(target_mask)
    
    return {
        'input_data': torch.stack(padded_inputs),
        'target_data': torch.stack(padded_targets),
        'input_masks': torch.stack(input_masks),
        'target_masks': torch.stack(target_masks)
    }

class CustomCombinedDataset(Dataset):
    def __init__(self, datasets, sampling_rates=None):
        self.datasets = datasets
        self.concat = ConcatDataset([ds for _, ds in datasets])
        self.sampling_rates = {}
        
        if sampling_rates is not None:
            self.sampling_rates = sampling_rates
        
        for ds_name, _ in self.datasets:
            if ds_name not in self.sampling_rates:
                self.sampling_rates[ds_name] = 1.0

        self.weights = []
        for ds_name, ds_obj in datasets:
            ds_len = len(ds_obj)
            rate = self.sampling_rates[ds_name]
            self.weights += [rate] * ds_len

        self.weights = torch.DoubleTensor(self.weights)

    def __len__(self):
        return len(self.concat)
    
    def __getitem__(self, idx):
        return self.concat[idx]
    
    def save_dataset(self, path):
        save_dict = {
            'datasets': self.datasets,
            'sampling_rates': self.sampling_rates,
            'weights': self.weights
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load_dataset(cls, path):
        data = torch.load(path)
        return cls(data['datasets'], data['sampling_rates'])

    def get_loader(self, batch_size=8, num_workers=0, replacement=True):
        sampler = WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self.concat),
            replacement=replacement
        )
        return DataLoader(
            self.concat,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=transformer_collate_fn
        )

def load_datasets_from_directory(directory):
    pt_files = sorted([f for f in os.listdir(directory) if f.endswith('.pt')],
                     key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    datasets = []
    for filename in pt_files:
        name = filename.split('.')[0]
        dataset = torch.load(os.path.join(directory, filename))
        datasets.append((name, dataset))
    
    return datasets

if __name__ == "__main__":
    # Example usage
    directory = "path/to/your/datasets/"
    datasets = load_datasets_from_directory(directory)
    
    # Create combined dataset
    combined_dataset = CustomCombinedDataset(datasets)
    
    # Save dataset
    combined_dataset.save_dataset('combined_dataset.pt')
    
    # Load dataset
    loaded_dataset = CustomCombinedDataset.load_dataset('combined_dataset.pt')
    
    # Create DataLoader with padding
    loader = loaded_dataset.get_loader(
        batch_size=32,
        num_workers=4,
        replacement=True
    )




# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.