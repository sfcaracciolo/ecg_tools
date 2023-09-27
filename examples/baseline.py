import wfdb 
import matplotlib.pyplot as plt 
from src.ecg_tools import highpass_filter, median_filter, splines_filter, time
from pathlib import Path
import numpy as np 
filename = Path(__file__).stem

rc = wfdb.rdrecord('data/PTB/s0021are')
start, end  = int((0*60 + 0)*rc.fs), int((0*60 + 40)*rc.fs)
ecg = rc.p_signal[start:end, 1]
pqs = np.load('data/pqs.npy')
t = time(ecg, rc.fs)

cases = (
    ['Highpass', lambda x: highpass_filter(x, 4, .5, rc.fs)],
    ['Median', lambda x: median_filter(x, 640)],
    ['Spline', lambda x: splines_filter(x, pqs, size=5, order=3)]
)

for label, fun in cases:
    fecg = fun(ecg)

    fig = plt.figure(0, figsize = (8.5, 3.5))
    ax = fig.add_subplot(111)
    ax.set_title(f'{label} filter')
    ax.plot(t, ecg, '-k', label='input')
    ax.plot(t, fecg, '--r', label='output')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [mV]')
    ax.legend()

    # inset axes....
    axins1 = ax.inset_axes([0.6, 0.25, 0.3, 0.5])
    axins1.plot(t, fecg, '-k')
    axins1.axhline(0, color='k', linewidth=1, linestyle='--')

    # sub region of the original image
    w = 3000
    x1, x2, y1, y2 = t[t.size//2 - w // 2], t[t.size//2 + w // 2], fecg.min(), fecg.max()
    axins1.set_xlim(x1, x2)
    axins1.set_ylim(y1, y2)
    axins1.set_xticks([])
    axins1.set_yticks([])

    ax.indicate_inset_zoom(axins1, edgecolor="black")

    plt.savefig(f"figs/{filename}_{label.lower()}.png", dpi = 300, orientation = 'portrait', bbox_inches = 'tight')
    plt.show()