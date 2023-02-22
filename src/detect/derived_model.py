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
    benign_logits_lst = []
    adv_logits_lst = []
    SEG_NUM = args.segment_num

    for segment in range(SEG_NUM):
        data, label, Zxxs = load_data(args, segment)
        
        v_min = Zxxs.min(axis=(0, 1), keepdims=True)
        v_max = Zxxs.max(axis=(0, 1), keepdims=True)
        Zxxs = (Zxxs - v_min)/(v_max - v_min + 1e-4)
        
        X_train, X_test, y_train, y_test = train_test_split(Zxxs, label, test_size=0.2, random_state=42)

        model_path = f"saved_models/{args.dataset}/{args.attack_method}/{segment}/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        clf = classifyVGG(X_train, y_train, X_test, y_test, trainable=True, output_directory=model_path)

        benign_logits_lst.append(clf.predict(X_test))
        adv_logits_lst.append(clf.predict_proba(X_test))

    pass

def load_data(args, segment):
    trace_path    = args.trace_path
    attack_method = args.attack_method
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

    return myloader.data, myloader.label, myloader.Zxxs


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(args)
    train_derived_model(args)

    

if __name__ == "__main__":
    main()
