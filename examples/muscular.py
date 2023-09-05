import wfdb 
import matplotlib.pyplot as plt 
from src.ecg_tools import lowpass_filter, average_filter, time
from pathlib import Path
filename = Path(__file__).stem

rc = wfdb.rdrecord('data/mit-bih-normal-sinus-rhythm-database-1.0.0/16265')
start, end  = int((10*60 + 21.5)*rc.fs), int((10*60 + 25)*rc.fs)
ecg = rc.p_signal[start:end, 0]
t = time(ecg, rc.fs)

cases = (
    ['Low pass', lambda x: lowpass_filter(x, 4, 45., rc.fs)],
    ['Moving average', lambda x: average_filter(x, 3)]
)

for label, fun in cases:
    fecg = fun(ecg)

    fig = plt.figure(figsize = (8.5, 3.5))
    ax = fig.add_subplot(111)
    ax.set_title(f'{label} filter')
    ax.plot(t, ecg, '-k', label='input')
    ax.plot(t, fecg, '--r', label='output')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [mV]')
    ax.legend()

    plt.savefig(f"figs/{filename}_{label.lower().replace(' ', '_')}.png", dpi = 300, orientation = 'portrait', bbox_inches = 'tight')
    plt.show()