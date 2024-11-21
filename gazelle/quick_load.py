from datasets import load_dataset, concatenate_datasets
from multiprocessing import Pool
import functools

def load_shard(shard_idx, dataset_name, num_proc):
    """Load a single shard of the dataset"""
    return load_dataset(
        dataset_name,
        split=f'train[{shard_idx}%{num_proc}::{num_proc}]',
        num_proc=1
    )

def load_dataset_parallel(dataset_name, num_proc=4):
    """Load dataset using parallel processing"""
    with Pool(num_proc) as pool:
        # Create partial function with fixed arguments
        load_fn = functools.partial(load_shard, 
                                  dataset_name=dataset_name,
                                  num_proc=num_proc)
        
        # Map function across shards
        shards = pool.map(load_fn, range(num_proc))
    
    # Concatenate all shards
    final_dataset = concatenate_datasets(shards)
    return final_dataset

# Example usage
