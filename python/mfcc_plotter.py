import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file_path = '/Users/Archish/Documents/CodeProjects/Python/IPF/datafiles/all_data/ipf_1.wav'

y, sr = librosa.load(file_path, res_type='kaiser_fast')

max_pad_len = 174
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, dct_type=3)
pad_width = max_pad_len - mfccs.shape[1]
mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('IPF MFCC')
plt.tight_layout()
plt.show()
