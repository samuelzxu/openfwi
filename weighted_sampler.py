import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np

class WeightedStratifiedSampler(Sampler):
    """
    Weighted sampler that takes into account data type strata and their corresponding losses.
    Supports both distributed and non-distributed training.
    """
    def __init__(self, dataset, data_types, weights=None, distributed=False, num_replicas=None, rank=None, shuffle=True):
        """
        Args:
            dataset: Dataset to sample from
            data_types: List of possible data types
            weights: Optional dictionary mapping data types to their weights
            distributed: Whether to use distributed sampling
            num_replicas: Number of distributed processes
            rank: Rank of current process
            shuffle: Whether to shuffle indices
        """
        self.dataset = dataset
        self.data_types = data_types
        self.shuffle = shuffle
        self.distributed = distributed
        
        # Initialize uniform weights if none provided
        if weights is None:
            weights = {dt: 1.0 for dt in data_types}
        self.weights = weights
        
        # Create indices per data type
        self.indices_by_type = {dt: [] for dt in data_types}
        for idx in range(len(dataset)):
            path = dataset.df.iloc[idx]['absolute_x_path']
            for dt in data_types:
                if dt in path:
                    self.indices_by_type[dt].append(idx)
                    break
        
        # Set up distributed sampling if needed
        if distributed:
            if num_replicas is None:
                num_replicas = torch.distributed.get_world_size()
            if rank is None:
                rank = torch.distributed.get_rank()
            
            self.num_replicas = num_replicas
            self.rank = rank
            self.epoch = 0
            
            # Adjust number of samples per replica
            self.num_samples = self.calculate_num_samples()
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = len(dataset)
            
    def calculate_num_samples(self):
        """Calculate number of samples for distributed sampling"""
        total_samples = sum(len(indices) for indices in self.indices_by_type.values())
        return (total_samples + self.num_replicas - 1) // self.num_replicas
        
    def update_weights(self, new_weights):
        """Update sampling weights based on validation loss"""
        total_weight = sum(new_weights.values())
        self.weights = {dt: w/total_weight for dt, w in new_weights.items()}
        
    def __iter__(self):
        # Set random seed for shuffling
        if self.shuffle:
            g = torch.Generator()
            if self.distributed:
                g.manual_seed(self.epoch)
            else:
                g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            
        # Calculate number of samples to draw from each stratum
        total_samples = self.num_samples
        samples_per_type = {dt: int(self.weights[dt] * total_samples) for dt in self.data_types}
        
        # Adjust for rounding errors
        remaining = total_samples - sum(samples_per_type.values())
        if remaining > 0:
            # Add remaining samples to data types proportionally
            for dt in sorted(self.data_types, 
                           key=lambda x: self.weights[x] * total_samples - samples_per_type[x],
                           reverse=True)[:remaining]:
                samples_per_type[dt] += 1
                
        # Generate indices
        indices = []
        for dt in self.data_types:
            stratum_indices = self.indices_by_type[dt]
            if self.shuffle:
                stratum_indices = torch.randperm(len(stratum_indices), generator=g).tolist()
            # Sample with replacement if we need more samples than available
            if samples_per_type[dt] > len(stratum_indices):
                indices.extend(np.random.choice(stratum_indices, 
                                             size=samples_per_type[dt], 
                                             replace=True).tolist())
            else:
                indices.extend(stratum_indices[:samples_per_type[dt]])
            print(f"Sampled {samples_per_type[dt]} samples from {dt} stratum")
                
        if self.shuffle:
            # Shuffle all sampled indices
            indices = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in indices]
            
        # Handle distributed case
        if self.distributed:
            # Pad indices to ensure equal split across GPUs
            if len(indices) < self.total_size:
                indices.extend(indices[:(self.total_size - len(indices))])
            else:
                indices = indices[:self.total_size]
            # Split indices between processes
            indices = indices[self.rank:self.total_size:self.num_replicas]
            
        return iter(indices)
        
    def __len__(self):
        return self.num_samples
        
    def set_epoch(self, epoch):
        """Set epoch for distributed sampling"""
        if self.distributed:
            self.epoch = epoch 