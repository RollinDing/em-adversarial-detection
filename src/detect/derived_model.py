"""
Train derived model to classify the EM traces into N classes for different datasets.
"""
from secrets import choice
import sys
import logging
from turtle import title
sys.path.append("./src/")
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.loader import Loader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import time 
import os

from utils.args import parse_args
from models.classify import classifyVGG

def train_derived_model(args):
    pass

def load_data(args):
    trace_path    = args.trace_path
    attack_method = args.attack_method
    segment       = args.segment_id
    rate          = args.rate
    
    trace_num     = 10_000
    if segment > 3:
        trace_len = 33_000
    else:
        trace_len = 36_000
    
    label_path = f"data/{args.dataset}/{attack_method}/y_adv.npy"
    trace_path = f"{trace_path}/{attack_method}/{str(segment)}"
    output_path = f"meta/raw/{attack_method}/{str(segment)}"

    myloader = Loader(trace_path, label_path, trace_num, trace_len, rate, output_path)
    myloader.stft(nperseg=256)

    return myloader.data, myloader.label


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(args)

    # Load data
    data, label = load_data(args)
    
    logging.info(data.shape)
    

if __name__ == "__main__":
    main()
