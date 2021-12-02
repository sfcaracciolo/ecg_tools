import numpy as np
import scipy as sp
from scipy import ndimage, stats
from . import utils

# all functions get signals by column in C order.

# def moving_average_filter(s: np.ndarray, size: int, in_place: bool = False) -> np.ndarray:
#     """
#     Moving average filter.
    
#     s: signal to apply moving average.
#     size: window size of the filter.
#     """
#     w = np.ones(size, dtype=np.float32)
#     y = np.convolve(s, w, mode = 'same')
#     y /= size

#     if in_place:
#         s[:] = y
#         y = s

#     return y

def nonlinear_adaptative_notch_filter(s: np.ndarray, alpha: float = 10., f0: float = 50., fs: int = 1024, unit=1.) -> np.ndarray:
    """
    Nonlinear PLI filter implemented from Laguna book (page 476) which is un adaption to [1]
    
    s: signal to remove PLI in uV.
    alpha: rate in uV
    fs: sampling frequency
    f0: frequency to adapt
    unit: scale of s to uV

    [1] P. S. Hamilton, "60Hz filtering for ECG signals: to adapt or not to adapt?," Proceedings of the 15th Annual International Conference of the IEEE Engineering in Medicine and Biology        Societ, 1993, pp. 779-780, doi: 10.1109/IEMBS.1993.978829.
    """
    s = s[:, np.newaxis] if s.ndim == 1 else s
    _, m = s.shape
    y = np.empty_like(s)
    N = 2*np.cos(2*np.pi*f0/fs)

    v2 = np.zeros(m, dtype=np.float32)
    v1 = np.zeros(m, dtype=np.float32)
    x1 = np.zeros(m, dtype=np.float32)
    v = np.zeros(m, dtype=np.float32)
    x = np.zeros(m, dtype=np.float32)
    ep = np.zeros(m, dtype=np.float32)

    # it = np.nditer(s, flags=['external_loop'], order='C')
    index = 0

    for z in s:
        # x[:] = unit*z
        x[:] = z
        x *= unit
        # v = N*v1 - v2 + 1.
        v[:] = v1
        v *= N
        v -= v2
        v += 1
        # ep = (x - x1) - (v - v1)
        ep[:] = x
        ep -= x1
        ep += v1
        ep -= v
        # v += alpha*np.sign(ep) 
        np.sign(ep, out=ep)
        ep *= alpha
        v += ep
        # y[i] = x - v
        yi = y[index, :]
        yi[:] = x
        yi -= v

        v2[:] = v1
        v1[:] = v
        x1[:] = x
        index += 1

    y /= unit
    return y

def bdr_median_filter(s: np.ndarray, size: int, in_place: bool = False) -> np.ndarray:
    """
    Baseline drift removal filter based on median filter. Recomendations: size should be near to RR duration.
    
    s: signal to remove baseline.
    size: window size of median filter.
    """

    s = s[:, np.newaxis] if s.ndim == 1 else s
    y = np.empty_like(s)
    it = np.nditer(s, flags=['external_loop'], order='F')

    index = 0
    for z in it:
        sp.ndimage.median_filter(z, size=size, output=y[:, index])
        index += 1

    if in_place:
        s -= y
        y = s
    else:
        y[:] = s - y
    
    return y

def beat_matrix(s: np.ndarray, r_pos: np.ndarray, window: int = 0, mode = None) -> np.ndarray:
    if mode == None:
        size = window
    else:
        rr = np.diff(r_pos)
        # size = eval(f'np.{mode}(rr)')
        size = mode(rr)

    beats, _, _ = utils.sliding_window_from_centers(s, r_pos, size)
    return beats

def isoline_correction(s: np.ndarray, limits: tuple = (0, None), engine: str = 'scipy', bins: int = 10, in_place: bool = False) -> np.ndarray:
    """
    Isoline correction remove the offset based on stat mode in order to set the isoline to 0 V. 
    
    s: signal to remove the offset.
    limits: range of s where the offset is computed. Useful to avoid side effects. Example: (5, 10) dischard the first five samples and the last ten
    engine: can be scipy or numpy. Define what library use to compute the mode.
    bins: works when engine is numpy. Define amount of bins to the mode estimation.
    """
    s = s[:, np.newaxis] if s.ndim == 1 else s
    _, m = s.shape
    iso = np.empty((1,m), dtype=np.float32)
    start, stop = limits
    stop = -stop if stop is not None else None

    s_view = s[start:stop,:]
    it = np.nditer(s_view, flags=['external_loop'], order='F')

    if engine == 'scipy':
        # scipy version
        iso[:] = sp.stats.mode(s_view, axis=0, nan_policy='raise').mode
    elif engine == 'numpy':
        # numpy version
        index = 0
        for z in it:
            hist, bin_edges = np.histogram(z, bins=bins)
            iso[0, index] = bin_edges[hist.argmax()]
            index += 1
    else:
        raise ValueError('engine does not exist.')

    if in_place:
        s -= iso
        y = s
    else:
        y = s - iso
    
    return y

def bdr_spline_filter(s: np.ndarray, fiducials: np.ndarray, size: int = 0, order: int = 3, in_place: bool = False) -> np.ndarray:
    """
    Baseline drift removal filter based on spline filter.
    
    s: signal to remove baseline.
    fiducials: array with PQ or TP fiducials points where isolectric is defined. Unit: samples.
    size: window size around fiducials where the isoelectric is searched.
    order: spline order.
    """

        
    s = s[:, np.newaxis] if s.ndim == 1 else s
    n, m = s.shape

    fiducials = fiducials[:, np.newaxis] if fiducials.ndim == 1 else fiducials
    i, j = fiducials.shape
    fiducials = np.broadcast_to(fiducials, (i, m)) if (m > 1) and (j == 1) else fiducials

    y = np.empty_like(s)
    it = np.nditer(s, flags=['external_loop'], order='F')
    t = np.arange(n)

    index = 0
    for z in it:
        windows, onsets = utils.sliding_window_from_centers(z, fiducials[:,index], size)
        medians = np.quantile(windows, .5, axis=1, interpolation='nearest')[:, np.newaxis]
        ixs = np.argwhere(windows == medians)[:, 1]
        ixs += onsets

        y[:,index] = sp.interpolate.UnivariateSpline(
            ixs,
            z[ixs],
            k=order,
            s=0,
            ext='extrapolate'
        )(t)
        index += 1

    if in_place:
        s -= y
        y = s
    else:
        y *= -1.
        y += s

    return y
