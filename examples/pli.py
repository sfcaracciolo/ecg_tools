import matplotlib.pyplot as plt 
from src.ecg_tools import notch_filter, adapt_notch_filter, time
from pathlib import Path
import numpy as np 
filename = Path(__file__).stem

fc, fs, bw = 51., 1024., 2.
ecg = np.load('data/raw_bp_2_300.npy')
t = time(ecg, fs)

cases = (
    ['Notch', lambda x: notch_filter(x, 4, fc, fs, bw)],
    ['Adaptative', lambda x: adapt_notch_filter(x, fs, f0=fc, alpha=.8e-3)]
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

    plt.savefig(f"figs/{filename}_{label.lower()}.png", dpi = 300, orientation = 'portrait', bbox_inches = 'tight')
    plt.show()