from typing import Literal
import numpy as np
import scipy as sp

def time(x: np.ndarray, fs: float):
    return np.arange(x.size) / fs

def lowpass_filter(x: np.ndarray, order: int, fc: float, fs: float) -> np.ndarray :
    sos = sp.signal.butter(order, fc, fs=fs, btype='lowpass', analog=False, output='sos')
    return sp.signal.sosfiltfilt(sos, x)

def highpass_filter(x: np.ndarray, order: int, fc: float, fs: float) -> np.ndarray :
    sos = sp.signal.butter(order, fc, fs=fs, btype='highpass', analog=False, output='sos')
    return sp.signal.sosfiltfilt(sos, x)

def notch_filter(x: np.ndarray, order: int, fc: float, fs: float, bw: float) -> np.ndarray:
    f1, f2 = fc - bw/2., fc + bw/2.
    sos = sp.signal.butter(order, (f1, f2), fs=fs, btype='bandstop', analog=False, output='sos')
    return sp.signal.sosfiltfilt(sos, x)

def median_filter(x: np.ndarray, size: int, return_baseline: bool = False) -> np.ndarray:
    baseline = sp.ndimage.median_filter(x, size=size)
    return baseline if return_baseline else x - baseline

def spline_filter(x: np.ndarray, fiducials: np.ndarray, size: int = 0, order: int = 3, return_baseline: bool = False) -> np.ndarray:
    windows, onsets = sliding_window(x, fiducials, size, roll=False)
    medians = np.quantile(windows, .5, axis=1, method='closest_observation')[:, np.newaxis]
    ixs_med = np.argwhere(windows == medians)
    _, ixs = np.unique(ixs_med[:,0], return_index=True)
    j = ixs_med[ixs, 1] + onsets
    t = np.arange(x.size)
    baseline = sp.interpolate.UnivariateSpline(j, x[j], k=order, s=0, ext='extrapolate')(t)
    return baseline if return_baseline else x - baseline

def isoline_correction(x: np.ndarray, engine: Literal['numpy', 'scipy'] = 'scipy', bins: int = 10, return_isoline: bool = False) -> np.ndarray:
    if engine == 'scipy':
        isoline = sp.stats.mode(x, axis=0, nan_policy='raise').mode
    elif engine == 'numpy':
        hist, bin_edges = np.histogram(x, bins=bins)
        isoline = bin_edges[hist.argmax()]
    return isoline if return_isoline else x - isoline

def adapt_notch_filter(x: np.ndarray, fs: float, f0: float = 50., alpha: float = 10e-6, return_adapt: bool = False) -> np.ndarray:
    """
    Nonlinear PLI filter implemented from Laguna book (page 476) which is un adaption to [1]
    [1] P. S. Hamilton, "60Hz filtering for ECG signals: to adapt or not to adapt?,"
    """
    N = 2*np.cos(2*np.pi*f0/fs)
    v = np.zeros_like(x)
    e = np.zeros_like(x)
    v1, v2, e1 = 0., 0., 0.

    for i in range(x.size):

        v[i] = N*v1 - v2
        e[i] = x[i] - v[i]
        dei = e[i] - e1 
        v[i] += alpha * np.sign(dei)

        v2 = v1
        v1 = v[i]
        e1 = e[i]   

    return v if return_adapt else e

def average_filter(x: np.ndarray, size: int) -> np.ndarray:
    """
    Moving average filter.
    
    x: signal to apply moving average.
    size: window size of the filter.
    """
    w = np.full(size, 1./size)
    return np.convolve(x, w, mode = 'same')

def beat_matrix(x: np.ndarray, r_pos: np.ndarray, size: int, **kwargs) -> np.ndarray:
    beats, _, = sliding_window(x, r_pos, size, **kwargs)
    return beats

def sliding_window(x: np.ndarray, centers: np.ndarray, size: int, roll: bool = False) -> tuple:
    matrix_rows = x.size - size + 1
    onsets = np.mod(centers - size // 2, matrix_rows)
    matrix = np.lib.stride_tricks.sliding_window_view(x, size)[onsets, :]
    if roll:
        mask = (onsets >= 0) & (onsets <= x.size - size)
        matrix = matrix[mask]
    return matrix, onsets