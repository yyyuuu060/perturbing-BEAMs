# perturbing-BEAMs-EEG-adversarial-attack-to-deep-learning-models-for-brain-disease-diagnosing

Source of data：https://archive.physionet.org/physiobank/database/chbmit/
This database is described in Ali Shoeb. Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology, September 2009.

Note: We first trimmed the raw data by EDFbrowser and divided it into disease data and non-disease data


The required environment:

python=3.7 or 3.8
mne_python=0.23.0
pytorch= 1.8.0
pywt
scipy
pandas
sklearn


The file function:

data_press.py        EEG is converted to BEAM

Train.py     Training model

GPBEAM(FGSM).py

GPBEAM(IFGSM).py

GPBEAM(PGD).py

GPBEAM_DE.py

Time_Convert.py    ：Convert BEAM adversarial samples back to EEEG adversarial samples

Time_attack:EEG adversarial samples attack EEG-related models


Note: We have made some changes to topomap.py for the MNE library. You need to replace the MNE library's topomap.py with our topomap.py

The address of topomap in the MNE library E:\Anaconda\envs\torch\Lib\site-packages\mne\viz
