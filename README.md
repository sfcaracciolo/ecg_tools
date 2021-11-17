# ecg tools

Common preprocessing methods for ECG:
*   Baseline drift removal
    * Based on median filter
    * Based on splines (require fiducials)
*   Power Line Interference removal
    * Adaptative nonlinear notch filter
*   Isoline correction
    * Based on statistical mode (Numpy and Scipy estimation methods)

Yapa: beat_matrix function perform a beat matrix of ecg based on RR series.

# Installation
First setup your venv and activate it, then run
```
pip install git+https://github.com/sfcaracciolo/ecg_tools.git
```

# Usage

Always use C order Numpy ndarrays and ECG channels by column. All functions works for arbitrary amount of channels.

# Contact
Please, if you use this fragment, contact me at scaracciolo@conicet.gov.ar