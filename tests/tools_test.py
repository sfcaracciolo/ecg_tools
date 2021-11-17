
import sys
import numpy as np
from ecg_tools import tools
np.set_printoptions(precision=1,threshold=sys.maxsize)

def test_sliding_window_from_centers():
    s = np.random.normal(size=30)
    r_pos = np.arange(40, step=4)
    size = 4
    # print('\nwindow', size)
    # print('r_pos', r_pos)
    # print('s', s)
    out, _ = tools.sliding_window_from_centers(s, r_pos, size)
    # print(out)
    assert 1 == 1

def test_bdr_spline_filter():
    s = np.random.normal(size=30)
    fiducials = np.arange(40, step=4)
    size = 4
    # print('\nwindow', size)
    # print('fiducials', fiducials)
    # print('s', s)
    out = tools.bdr_spline_filter(s, fiducials, size)
    # print(out)
    assert 1 == 1