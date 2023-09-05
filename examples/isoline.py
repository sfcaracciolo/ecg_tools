import wfdb 
import matplotlib.pyplot as plt 
from src.ecg_tools import isoline_correction, highpass_filter, time
from pathlib import Path
filename = Path(__file__).stem

rc = wfdb.rdrecord('data/mit-bih-normal-sinus-rhythm-database-1.0.0/16272')
start, end  = int((0*60 + 0)*rc.fs), int((0*60 + 5)*rc.fs)
ecg = rc.p_signal[start:end, 0]
t = time(ecg, rc.fs)

fecg = highpass_filter(ecg, 4, .5, rc.fs)
ifecg = isoline_correction(fecg, engine='numpy', bins=100)

fig = plt.figure(0, figsize = (8.5, 3.5))
ax = fig.add_subplot(111)
ax.set_title('Isoline raw')
ax.plot(t, fecg, '-k')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión [mV]')
ax.axhline(0, color='r', linewidth=1, linestyle='--')
plt.savefig(f"figs/{filename}_raw.png", dpi = 300, orientation = 'portrait', bbox_inches = 'tight')
plt.show()

fig = plt.figure(1, figsize = (8.5, 3.5))
ax = fig.add_subplot(111)
ax.plot(t, ifecg, '-k')
ax.set_title('Isoline correction')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Tensión [mV]')
ax.axhline(0, color='r', linewidth=1, linestyle='--')
plt.savefig(f"figs/{filename}_correction.png", dpi = 300, orientation = 'portrait', bbox_inches = 'tight')
plt.show()
