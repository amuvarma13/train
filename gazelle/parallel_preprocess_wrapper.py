from accelerate import Accelerator
import torch.distributed as dist
from datasets import concatenate_datasets

def parallel_preprocess_wrapper(dataset, preprocess_fn):
    """
    Each GPU preprocesses 1/n of the data, then results are gathered
    """
    accelerator = Accelerator()
    
    # Calculate chunk size for each GPU
    total_size = len(dataset)
    chunk_size = total_size // accelerator.num_processes
    start_idx = accelerator.process_index * chunk_size
    end_idx = start_idx + chunk_size if accelerator.process_index < accelerator.num_processes - 1 else total_size
    
    # Each GPU processes its chunk
    print(f"GPU {accelerator.process_index} processing items {start_idx} to {end_idx}")
    local_chunk = dataset.select(range(start_idx, end_idx))
    processed_chunk = preprocess_fn(local_chunk)
    
    # We need to gather the processed chunks using dist.all_gather_object
    gathered_chunks = [None] * accelerator.num_processes
    dist.all_gather_object(gathered_chunks, processed_chunk)
    
    if accelerator.is_main_process:
        print("Main process concatenating chunks...")
    
    # All processes concatenate the chunks
    final_data = processed_chunk  # Initialize with local chunk
    for i, chunk in enumerate(gathered_chunks):
        if i != accelerator.process_index:  # Skip own chunk as it's already included
            if chunk is not None:
                if isinstance(chunk, (list, tuple)):
                    final_data.extend(chunk)
                else:
                    final_data.append(chunk)
    
    return final_data