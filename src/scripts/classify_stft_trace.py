import sys
import numpy as np
from tqdm import tqdm
sys.path.append("./src/")
from scipy.signal import stft
from utils.plotter import Plotter
from models.classify import classifyVGG
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.loader import Loader
import argparse

def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='Classification or Reterive the classifier of Short-Time Fourier Transform trace')
    parser.add_argument('--attack-method', choices=['org', 'pgd', 'cw'], type=str,
        help='choose the adverserial dataset with certain adversarial attack method, org means the original dataset')
    parser.add_argument('--batch', default=0, type=int,
        help='Select the batch to analyze from [0-27]')
    parser.add_argument('--rate', default=1, type=int,
        help='the downsample rate of preprocessing when analyzing the raw trace')
    parser.add_argument('--date', type=int,
        help='the date when the dataset is collected.')
    parser.add_argument('--channels', default=22, type=int, 
        help='the number of channel that the detector will persever for further classification, 22 by default')
    parser.add_argument('--windowsize', default=256, type=int,
        help='the size of short term fourier transform window')
    args = parser.parse_args()
    return args

def stft_classification(args):
    TRACE_PATH = f"../data/data/fm_lenet-{args.date}"
    ATTACK = args.attack_method
    BATCH = args.batch
    RATE  = args.rate

    trace_num = 10_000
    if BATCH <= 3:
        trace_len = 36_000
    elif BATCH <= 5 :
        trace_len = 33_000
    elif BATCH <= 17:
        trace_len = 10_000
    else:
        trace_len = 6_000
    
    
    label_path = f"../data/data/fm/{ATTACK}/y_adv.npy"
    trace_path = f"{TRACE_PATH}/{str(BATCH)}"
    output_path = f"meta/raw/{ATTACK}/{str(BATCH)}"
    myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
    myloader.stft(nperseg=args.windowsize)
    
    Zxxs = myloader.Zxxs
    Zxxs = Zxxs[:, :args.channels, :]
    labels = myloader.label
    labels = labels[:trace_num]
    
    v_min = Zxxs.min(axis=(0, 1), keepdims=True)
    v_max = Zxxs.max(axis=(0, 1), keepdims=True)
    Zxxs = (Zxxs - v_min)/(v_max - v_min)
    X_train, X_test, y_train, y_test = train_test_split(Zxxs, labels, test_size=0.2, random_state=42)
    classifyVGG(X_train, y_train, X_test, y_test, trainable=True, output_directory=f"saved_models/vgg/{ATTACK}/{BATCH}/")

def main():
    args = parse_args()
    stft_classification(args)

if __name__ == "__main__":
    main()
    