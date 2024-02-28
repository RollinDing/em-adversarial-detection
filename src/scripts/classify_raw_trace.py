import numpy as np
import sys
from scipy import signal
from scipy.sparse.construct import rand
sys.path.append("./src/")
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models.classify import classifyMLP

TRACE_PATH = "../data/data/fm_pico_20211105"
LABEL_PATH = "data/downsampled/y_test.npy"
TYPE = "org"
BATCH = "0"

trace_num = 4_000
trace_len = 70_000

def load_traces(TRACE_PATH, TYPE, BATCH, trace_num, trace_len):
    data = np.zeros(shape=[trace_num, trace_len])
    path = f"{TRACE_PATH}/{BATCH}"
    RATE = 0.1
    for index in tqdm(range(trace_num)):
        try:
            trace = np.load(path+f"/{index}.npy")
            resampled_trace = signal.resample(trace, int(trace.shape[0] * RATE))
            data[index, :resampled_trace.shape[0]] = resampled_trace
        except FileNotFoundError:
            print("oops! No such File")
            pass
        except ValueError:
            print("The trace value is invalid")
            pass
        
    return data

traces = load_traces(TRACE_PATH, TYPE, BATCH, trace_num, trace_len)
labels = np.load(LABEL_PATH)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(traces, labels, test_size=0.2, random_state=42)
    classifyMLP(X_train, y_train, X_test, y_test, trainable=True, output_directory=f"saved_models/mlp/")