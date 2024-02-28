"""
Load and save the raw data as processed data

"""
import os
import sys
sys.path.append("./src/")
import numpy as np
from tqdm import tqdm
from scipy import signal
from utils.plotter import Plotter
from scipy.fft import fft, fftfreq
from scipy.signal import stft

class Loader:
    def __init__(self, trace_path, label_path, trace_num, trace_len, rate, output_path) -> None:
        self.trace_path = trace_path
        self.label_path = label_path
        self.trace_num  = trace_num
        self.trace_len  = trace_len
        self.rate       = rate
        self.output_path= output_path
        self.label      = np.load(self.label_path)
        if not os.path.exists(self.output_path+"/raw.npy"):
            print("> First time load the traces, start loading...")
            self.data       = self.load_traces()
            self.save_traces()
        else:
            print("> The trace is pre-loaded, start resuming...")
            self.data       = np.load(self.output_path+"/raw.npy")
        pass

    def load_traces(self):
        # The code is to save the downsampled traces
        data = np.zeros(shape=[self.trace_num, int(self.trace_len * self.rate)])
        
        for index in tqdm(range(self.trace_num)):
            try:
                trace = np.load(self.trace_path+f"/{index}.npy")
                # resample the trace
                resampled_trace = signal.resample(trace, int(trace.shape[0] * self.rate))
                data[index, :resampled_trace.shape[0]] = resampled_trace
                # data[index, :trace.shape[0]] = trace
            except FileNotFoundError:
                print("oops! No such File", self.trace_path)
                pass
            except ValueError:
                print("The trace value is invalid")
                pass
        return data
    
    def save_traces(self):
        """
        Save the traces as meta data to the self.output_path
        """
        print("> Start saving traces...")
        np.save(self.output_path+"/raw.npy", self.data)
        np.save(self.output_path+"/label.npy", self.label)

    def display(self):
        print("> Start ploting some sample traces...")
        myPlotter = Plotter(4, 2)
        for idx in range(4):
            self.label = self.label.flatten()
            aver = self.data[self.label==idx].mean(axis=0)
            myPlotter.drawPlot(aver, idx, 0)
        for idx in range(4):
            sample = self.data[self.label==idx][0]
            myPlotter.drawPlot(sample, idx, 1)
        myPlotter.show()
        pass
    
    def stft(self, fs=2e10, nperseg=256, n_channel=22, band=False):
        """
        Conduct short-term fourier transform and save it
        fs: sample frequency
        nperseg: number of segments per sliding window
        """
        if not os.path.exists(self.output_path+"/stft.npy"):
            print("> Conduct Short-Term Fourier Transform...")
            traces = self.data
            self.n_channel = n_channel
            self.max_len   = 800
            print("> Create empty Zxxs matrix..")
            Zxxs = np.zeros([self.trace_num, self.n_channel, self.max_len])
            for idx in tqdm(range(self.trace_num)):
                trace = traces[idx]
                _, _, Zxx = stft(trace, fs=fs, window='hann', nperseg=nperseg)
                freq15 = 156
                if not band:
                    Zxxs[idx, :, :Zxx.shape[1]] = abs(Zxx[:self.n_channel, :])
                else:
                    Zxxs[idx, :, :Zxx.shape[1]] = np.abs(Zxx)[freq15-int(n_channel/2):freq15+int(n_channel/2), :]
                self.max_len = Zxx.shape[1]
            
                # Zxxs.append(np.expand_dims(Zxx, axis=0))
                
            # Zxxs = np.concatenate(Zxxs, axis=0)
            self.Zxxs = Zxxs[:, :, :self.max_len]
            print("> Start saving stft traces...", self.Zxxs.shape)
            np.save(self.output_path+"/stft.npy", self.Zxxs)
        else:
            print("> Short-Term Fourier Transform is pre-computed...")
            self.Zxxs = np.load(self.output_path+"/stft.npy")

    def stft_winsize(self, fs=2e10, nperseg=256):
        """
        Conduct short-term fourier transform and save it
        fs: sample frequency
        nperseg: number of segments per sliding window
        """
        print("> Conduct Short-Term Fourier Transform...")
        traces = self.data
        self.n_channel = 22
        self.max_len   = 400
        print("> Create empty Zxxs matrix..")
        Zxxs = np.zeros([self.trace_num, self.n_channel, self.max_len])
        for idx in tqdm(range(self.trace_num)):
            trace = traces[idx]
            _, _, Zxx = stft(trace, fs=fs, window='hann', nperseg=nperseg)
            Zxxs[idx, :, :Zxx.shape[1]] = abs(Zxx[:self.n_channel, :])
            self.max_len = Zxx.shape[1]
            
            # Zxxs.append(np.expand_dims(Zxx, axis=0))
                
            # Zxxs = np.concatenate(Zxxs, axis=0)
            self.Zxxs = Zxxs[:, :, :self.max_len]
            print("> Start saving stft traces...", self.Zxxs.shape)
            
        

    def display_stft(self):
        print("> Start ploting some Spectrum...")
        myPlotter = Plotter(2, 4)
        for idx in range(4):
            self.label = self.label.flatten()
            aver = self.Zxxs[self.label==idx].mean(axis=0)
            myPlotter.drawImage(aver, 0, idx)
        for idx in range(4):
            aver = self.Zxxs[self.label==(idx+4)].mean(axis=0)
            myPlotter.drawImage(aver, 1, idx)
        myPlotter.show()
        pass

if __name__ == "__main__":
    TRACE_PATH = "../data/data/fm_lenet-20211102"
    ATTACK = "cw"
    BATCH = 0
    RATE  = 1
    trace_num = 10_000
    if BATCH > 3:
        trace_len = 33_000
    else:
        trace_len = 36_000
    
    label_path = f"data/{ATTACK}/y_adv.npy"
    trace_path = f"{TRACE_PATH}/{ATTACK}/{str(BATCH)}"
    output_path = f"meta/raw/{ATTACK}/{str(BATCH)}"
    myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
    myloader.stft(nperseg=256)