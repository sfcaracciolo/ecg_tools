import numpy as np 

def sliding_window_from_centers(s: np.ndarray, centers: np.ndarray, size: int) -> tuple:
    onsets = centers - size // 2
    mask = (onsets >= 0) & (onsets <= s.size - size)
    onsets = onsets[mask]
    matrix = np.lib.stride_tricks.sliding_window_view(s, size)[onsets, :]
    return matrix, onsets