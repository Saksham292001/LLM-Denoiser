import os
import torch

def get_device():
    """Returns 'cuda' if GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(d):
    """Ensures a directory exists."""
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
