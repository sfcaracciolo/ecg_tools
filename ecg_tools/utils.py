import numpy as np 

def sliding_window_from_centers(s: np.ndarray, centers: np.ndarray, size: int) -> tuple:
    onsets = centers - size // 2
    mask = (onsets >= 0) & (onsets <= s.size - size)
    np.logical_not(mask, out=mask)
    matrix = np.lib.stride_tricks.sliding_window_view(s, size)[onsets, :]
    # matrix = np.lib.stride_tricks.sliding_window_view(s, size)[onsets[mask], :]
    return matrix, onsets, mask