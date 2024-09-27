import numpy as np

def invert_se3(se3_matrix: np.ndarray) -> np.ndarray:
    rotation_inv = se3_matrix[:3, :3].T
    translation_inv = -rotation_inv @ se3_matrix[:3, 3]
    
    inverted_se3_matrix = np.eye(4)
    inverted_se3_matrix[:3, :3] = rotation_inv
    inverted_se3_matrix[:3, 3] = translation_inv
    
    return inverted_se3_matrix