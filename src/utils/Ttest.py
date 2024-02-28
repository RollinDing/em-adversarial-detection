"""
Plot the T tests of samples from class 1 and class2
To compare the high t-value point of spectrum, time domain signal, frequency domain signal
Input:
2 classes:  C1 C2
Spectrums: precomputed

"""
from turtle import title
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def draw_spectrum(array):
    df_cm = pd.DataFrame(array.astype(np.int32))
    plt.figure(figsize=(18, 6))
    color = plt.get_cmap('viridis') 
    ax = sns.heatmap(df_cm, annot=False, fmt="g", xticklabels=[], yticklabels=[],cmap=color, cbar_kws={'label':"T-statistics"}, linewidths=0.0, rasterized=True)
    # plt.xticks(range(0, 1, 283))
    # plt.ylabel(range(0, 1716, 22))
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    ax.figure.axes[-1].yaxis.label.set_size(30)
    plt.xlabel('Time', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.tight_layout()
    plt.grid("off")
    plt.savefig(f"imgs/Ttest-stft.pdf")
    plt.close()


Batch = 0
C1 = 0
C2 = 1

stft = np.load(f"meta/fm/train/{Batch}/stft.npy")
raw = np.load(f"meta/fm/train/{Batch}/raw.npy")
label = np.load(f"meta/fm/train/{Batch}/label.npy")



C1stft = stft[label==C1]
C2stft = stft[label==C2]
Ttest_stft = stats.ttest_ind(C1stft, C2stft)[0]
print("STFT: ", abs(Ttest_stft).max())


C1raw = raw[label==C1]
C2raw = raw[label==C2]
Ttest_raw = stats.ttest_ind(C1raw, C2raw)[0]
xr = np.linspace(0, 1, Ttest_raw.shape[0])
xs = np.linspace(0, 1, Ttest_stft.shape[1])
print("TIME: ", abs(Ttest_raw).max())
plt.figure()
draw_spectrum(Ttest_stft)

plt.figure(figsize=(12, 8))
plt.plot(xr, Ttest_raw, color='black', label="Time domain Trace")
plt.ylabel("T-statistics", fontsize=60)
plt.xlabel("Time", fontsize=60)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.tight_layout()
plt.savefig(f"imgs/Ttest-raw.pdf")
plt.close()

C1raw = raw[label==C1]
C2raw = raw[label==C2]
C1fft = np.abs(np.fft.fft(C1raw, axis=1))
C2fft = np.abs(np.fft.fft(C2raw, axis=1))
Ttest_fft = stats.ttest_ind(C1fft, C2fft)[0]

N = len(Ttest_raw)
T = 1.0 / 10e9
y_f = Ttest_fft
x_f = np.linspace(0.0, 1.0/(2.0*T), N//2)

print("FFT:", abs(Ttest_fft).max())
plt.figure(figsize=(12, 8))
# plt.plot(Ttest_fft, label="Frequency Trace on 150 MHz")
plt.plot(x_f, y_f[:N//2], c='black')
plt.vlines(x=150e6, ymin=-170, ymax=170, colors="red", label="operating frequency")
plt.ylabel("T-statistics", fontsize=60)
plt.xlabel("Frequency", fontsize=60)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.tight_layout()
plt.savefig(f"imgs/Ttest-fft.pdf")
plt.close()

