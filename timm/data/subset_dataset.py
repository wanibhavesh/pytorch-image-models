""" Dataset Subset Wrapper

Wrapper to use only a percentage of a dataset by randomly sampling indices.
"""
import random
import torch.utils.data as data
from typing import Optional


class SubsetDataset(data.Dataset):
    """
    Dataset wrapper that uses only a random subset of the original dataset.
    
    This wrapper randomly samples a percentage of indices from the original dataset
    at initialization and uses only those indices throughout training. This ensures
    consistent subset usage across epochs while maintaining randomness in selection.
    
    Args:
        dataset: The original dataset to subset
        data_percent: Percentage of data to use (0.0 to 1.0)
        seed: Random seed for reproducible sampling
    """
    
    def __init__(
        self,
        dataset: data.Dataset,
        data_percent: float = 1.0,
        seed: int = 42
    ):
        assert 0.0 < data_percent <= 1.0, f"data_percent must be between 0.0 and 1.0, got {data_percent}"
        
        self.dataset = dataset
        self.data_percent = data_percent
        
        # Calculate number of samples to use
        original_length = len(dataset)
        self.subset_length = int(original_length * data_percent)
        
        # Randomly sample indices
        random.seed(seed)
        all_indices = list(range(original_length))
        self.sampled_indices = sorted(random.sample(all_indices, self.subset_length))
        
        print(f"SubsetDataset: Using {self.subset_length}/{original_length} samples ({data_percent*100:.1f}%)")
    
    def __getitem__(self, index):
        # Map subset index to original dataset index
        original_index = self.sampled_indices[index]
        return self.dataset[original_index]
    
    def __len__(self):
        return self.subset_length
    
    def __getattr__(self, name):
        # Delegate attribute access to the original dataset
        # This allows the wrapper to be transparent for other dataset methods
        return getattr(self.dataset, name) 