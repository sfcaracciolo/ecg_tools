import numpy as np 

def sliding_window_from_centers(s: np.ndarray, centers: np.ndarray, size: int) -> tuple:
    matrix_rows = s.size - size + 1
    onsets = np.mod(centers - size // 2, matrix_rows)
    mask = (onsets >= 0) & (onsets <= s.size - size)
    filter = np.logical_not(mask)
    matrix = np.lib.stride_tricks.sliding_window_view(s, size)[onsets, :]
    return matrix, onsets, filter