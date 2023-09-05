import wfdb 
import matplotlib.pyplot as plt 
from src.ecg_tools import beat_matrix, time
from pathlib import Path
import numpy as np 
import scipy as sp 
filename = Path(__file__).stem

rc = wfdb.rdrecord('data/PTB/s0021are')
start, end  = int((0*60 + 0)*rc.fs), int((0*60 + 40)*rc.fs)
ecg = rc.p_signal[start:end, 1]
t = time(ecg, rc.fs)

r_pos, _ = sp.signal.find_peaks(ecg, distance=.6*rc.fs)
size = np.diff(r_pos).mean()
m = beat_matrix(ecg, r_pos, int(size))

n = np.arange(m.shape[0])
m += n[:, np.newaxis]*.5

fig = plt.figure(figsize = (2.5, 8.5))
ax = fig.add_subplot(111)
ax.set_title('Beat matrix')
ax.plot(m.T, '-k')

plt.savefig(f"figs/{filename}.png", dpi = 300, orientation = 'portrait', bbox_inches = 'tight')
plt.show()