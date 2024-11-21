from accelerate import Accelerator
import torch.distributed as dist

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
    
    # Gather all processed chunks
    all_chunks = accelerator.gather(processed_chunk)
    
    if accelerator.is_main_process:
        # Flatten the gathered chunks if needed
        if isinstance(all_chunks, (list, tuple)):
            final_data = []
            for chunk in all_chunks:
                if isinstance(chunk, (list, tuple)):
                    final_data.extend(chunk)
                else:
                    final_data.append(chunk)
        else:
            final_data = all_chunks
    else:
        final_data = None
        
    # Broadcast final data to all processes
    final_data = accelerator.broadcast(final_data)
    
    return final_data