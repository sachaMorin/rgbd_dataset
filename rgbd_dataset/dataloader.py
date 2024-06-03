from typing import List, Dict
import numpy as np


def collate_fn_np_single(data: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    # Avoid conversion to tensors and batch dim
    if len(data) != 1:
        raise ValueError("This collate_fn is only for single observations")
    return data[0]


def collate_fn_np_single_cb():
    return collate_fn_np_single
