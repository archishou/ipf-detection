from scipy.io import wavfile

import matplotlib.pyplot as plt

filepath = '/Users/Archish/Documents/CodeProjects/Python/IPF/datafiles/all_data/healthy_1.wav'

# Read the wave file to numpy array
fs,  signal = wavfile.read(filepath)
channel_0 = signal[:,0]
channel_1 = signal[:,1]

# Plotting
fig, axes = plt.subplots(nrows=2)
axes[0].plot(channel_0, 'b')
axes[1].plot(channel_1, 'r')
plt.show()
