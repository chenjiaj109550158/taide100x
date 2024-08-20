import torch
import torch.nn as nn


def print_cuda_mem_usage():
    # Print CUDA memory usage
    allocated_memory = torch.cuda.memory_allocated('cuda:0')  # Memory allocated by tensors
    reserved_memory = torch.cuda.memory_reserved('cuda:0')    # Total memory reserved by the allocator

    print(f"CUDA memory allocated by tensors: {allocated_memory / (1024 ** 2):.2f} MB")
    print(f"Total CUDA memory reserved by the allocator: {reserved_memory / (1024 ** 2):.2f} MB")

    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"Total GPU memory: {total_memory / (1024 ** 2):.2f} MB")