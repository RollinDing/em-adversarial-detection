from PIL.Image import NONE
import numpy as np
import os
import sys
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import stft
from multiprocessing import Pool

BATCH = 0
TRACE_DIRECTORY = "../data/data/fm_lenet-20211102/org/{BATCH}"
OUTPUT_DIRECTORY = "meta/stft/fm"
RATE = 0.05
n_channels = 4
LABEL_FILE = "data/downsampled/y_test.npy"
NUM = 10000
MAX_LEN = 10000

class ShortTermFFT():
    def __init__(self, trace_id):
        self.input_path = TRACE_DIRECTORY
        self.output_path = OUTPUT_DIRECTORY
        self.index = trace_id
        self.RATE = RATE 
        self.n_channels = n_channels
        pass

    def save(self):
        """
        Load a single raw trace, do resampling and stft,
        Save it to output_path
        """
        
        if not os.path.exists(self.output_path + "/"+str(self.index)+".npy"):
            if os.path.exists(self.input_path + "/"+str(self.index)+".npy"):
                trace = np.load(self.input_path + "/"+str(self.index)+".npy")
                resampled_trace = signal.resample(trace, int(trace.shape[0] * RATE))
                _, _, Zxx = stft(resampled_trace, fs=5e8, nperseg=512)
                
                freq15 = 156
                # compute short term fourier transform 
                stft_trace = np.abs(Zxx)[freq15-self.n_channels:freq15+self.n_channels, :-1]
            else:
                stft_trace = np.zeros([self.n_channels*2, 1])
            # If we got the  output file
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
            np.save(self.output_path + "/"+str(self.index)+"_zxx.npy", Zxx)
            # np.save(self.output_path + "/"+str(self.index)+".npy", stft_trace)
            return Zxx.shape

    def load(self, adversarial_index=None):
        """
        Load the stft traces from output directory
        MAX_LEN: The max length of a trace
        adversarial_index: index of adversarial samples
        """
        labels     = np.load(LABEL_FILE)
        if adversarial_index is None:
            adversarial_index=np.ones(labels.shape[0], dtype=np.int8)
        else:
            adversarial_index.astype(np.int8)
        if adversarial_index[self.index] == 1:
            return np.load(self.output_path + "/"+str(self.index)+".npy")
        else:
            return np.zeros([self.n_channels*2, 1])

def STFT_saver(i):
    t = ShortTermFFT(i)
    debugger = t.save()

def STFT_loader(i, adversarial_index=None, adversarial_labels=None):
    t = ShortTermFFT(i)
    Fxx = t.load(adversarial_index)
    return Fxx

if __name__ == "__main__":
    with Pool(20) as p:
        list(tqdm(p.imap_unordered(STFT_saver, range(NUM)), total=NUM))

    # if NAME == "org":
    #     adversarial_index = None
    #     adversarial_labels = None
    # else:
    #     adversarial_index, adversarial_labels = find_adversarial_examples(ADVERSARIAL_TRACE_PATH)
    
    maxlength = 0
    Fxx = np.zeros(shape=[NUM, int(2*n_channels), MAX_LEN], dtype=np.float16)
    for idx in tqdm(range(NUM)):
        trc_len = STFT_loader(idx).shape[1]
        maxlength = max(trc_len, maxlength)
        fxx = STFT_loader(idx)
        Fxx[idx, :fxx.shape[0], :trc_len] = fxx 
    Fxx = Fxx[:, :, :maxlength]
    labels     = np.load(LABEL_FILE)
    # Fxx, labels, adversarial_labels = remove_empty_samples(Fxx, labels, adversarial_labels)
    print(Fxx.shape, labels.shape)
    # if adversarial_labels is not None:
    #     print(adversarial_labels.shape)
    # output_path = "meta/stft/"+TRACE_DIRECTORY+"/"+NAME+"/"+str(batch)
    # np.save(output_path+"/Fxx.npy", Fxx)
    # np.save(output_path+"/labels.npy", labels)
    # if adversarial_labels is not None:
    #     np.save(output_path+"/adv_labels.npy", adversarial_labels)
    # Fxx, labels, adversarial_labels = remove_empty_samples(Fxx, labels, adversarial_labels)
    # print(Fxx.shape, labels.shape)
