""" Dataset Subset Wrapper

Wrapper to use only a percentage of a dataset by randomly sampling indices.
"""
import random
import torch.utils.data as data
from torch.utils.data import Subset
from typing import Optional


class SubsetDataset(Subset):
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
        
        # Calculate number of samples to use
        original_length = len(dataset)
        subset_length = int(original_length * data_percent)
        
        # Randomly sample indices
        random.seed(seed)
        all_indices = list(range(original_length))
        sampled_indices = sorted(random.sample(all_indices, subset_length))
        
        # Initialize the parent Subset class with dataset and indices
        super().__init__(dataset, sampled_indices)
        
        # Store for reference
        self.data_percent = data_percent
        
        print(f"SubsetDataset: Using {subset_length}/{original_length} samples ({data_percent*100:.1f}%)") 