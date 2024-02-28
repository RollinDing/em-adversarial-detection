import sys
import logging
from turtle import title
from tensorflow.python.ops.gen_math_ops import Log
sys.path.append("./src/")
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.classify import classifyVGG
from utils.loader import Loader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from utils.GradCAM import GradCAM
import argparse
import time 
import os

from sklearn.decomposition import PCA

def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='Classification or Reterive the classifier of Short-Time Fourier Transform trace')
    parser.add_argument('--attack-method', choices=['org', 'train', 'pgd', 'cw', 'targetcw', 'fgm'], type=str,
        help='choose the adverserial dataset with certain adversarial attack method, org means the original dataset, train means using the training dataset for analyze.')
    parser.add_argument('--batch-num', default=1, type=int,
        help='Select the batches to analyze from [0-27]')
    parser.add_argument('--dataset', default='fm', choices=['fm', 'cifar10'], type=str,
        help="The dataset of both original cnn model and adversarial attack")
    parser.add_argument('--rate', default=1, type=float,
        help='the downsample rate of preprocessing when analyzing the raw trace')
    parser.add_argument('--date', type=int,
        help='the date when the dataset is collected.')
    parser.add_argument('--channels', default=22, type=int, 
        help='the number of channel that the detector will persever for further classification, 22 by default')
    parser.add_argument('--winsize', default=256, type=int,
        help='the size of short term fourier transform window')
    parser.add_argument('--target', default=None, type=int,
        help='the target class to train the derived model')
    parser.add_argument('--gradcam', default=False, type=bool,
        help='using GradCAM to record the gradient of EM classifiers')
    parser.add_argument('--robust', default=False, type=bool,
        help='Testing on robust model or not')
    args = parser.parse_args()
    return args

def timeit(func, iterations, *args):
    t0 = time.time()
    for _ in range(iterations):
        func(*args)
    logging.info("Time/iter: %.4f sec" % ((time.time() - t0) / iterations))

def cal_accuracy(y_pred, y_true, save=False):
    """
    Given the prediction and ground truth,
    compute the classification metric, including confusion matrix, classification report 
    """
    # convert the predicted from binary to integer 
    y_pred = np.argmax(y_pred , axis=1)
    logging.info('> prediction labels')
    logging.info(y_pred)
    logging.info('> Confusion Matrix')
    logging.info(confusion_matrix(y_true, y_pred))
    if save:
        plt.figure()
        plt.imshow(confusion_matrix(y_true, y_pred))
        plt.savefig("confusion-matrix.png")
        plt.close()
    logging.info('Classification Report')
    logging.info(classification_report(y_true, y_pred))

def load_data(ATTACK, BATCH, args):
    # Trace path for robust model
    TRACE_PATH = f"../data/data/{args.dataset}_robust-{args.date}"
    RATE  = args.rate
    CHANNELS = args.channels
    target = args.target
    trace_num = 10_000

    if BATCH <= 3:
        trace_len = 36_000
    elif BATCH <= 5 :
        trace_len = 33_000
    elif BATCH <= 17:
        trace_len = 10_000
    else:
        trace_len = 6_000

    logging.info(f"> The {BATCH}th BATCH:")    
    logging.info(f"> The input trace length is {trace_len}")

    if ATTACK == 'org':
        label_path = f"../data/data/{args.dataset}/{ATTACK}/y_adv.npy"
    else:
        label_path = f"../fpga-fashion-mnist/data/robust/{ATTACK}/y_adv.npy"

    trace_path = f"{TRACE_PATH}/{ATTACK}/{str(BATCH)}"
    
    output_path = f"meta/{args.dataset}/robust-{ATTACK}/{str(BATCH)}"

    logging.info(f"output path is {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
    myloader.stft(fs=2e10, nperseg=args.winsize, n_channel=CHANNELS)
    
    Zxxs = myloader.Zxxs
    labels = myloader.label

    v_min = Zxxs.min(axis=(0, 1), keepdims=True)
    v_max = Zxxs.max(axis=(0, 1), keepdims=True)
    Zxxs = (Zxxs - v_min)/(v_max - v_min + 1e-4)

    X_train, X_test, y_train, y_test = train_test_split(Zxxs, labels, test_size=0.2, random_state=42)
    print(f"The shape of training sample of {ATTACK} ", Zxxs.shape)   
    return  X_train, X_test, np.squeeze(y_train), np.squeeze(y_test)


def train_derived_model(args):
    """
    Train the derived model using orginal data
    """

    benign_logits_lst = []
    adv_logits_lst = []
    robust_benign_logits_lst = []
    BATCH_NUM = args.batch_num

    for BATCH in range(0, BATCH_NUM):
        # First adversarial attack for training
        X_train_0, X_test_0, y_train_0, y_test_0 = load_data("org", BATCH, args)
        X_train_1, X_test_1, y_train_1, y_test_1 = load_data("fgm", BATCH, args)
        # load Second  data --> the benign samples
        X_train = np.concatenate([X_train_0, X_train_1], axis=0)
        X_test = np.concatenate([X_test_0, X_test_1], axis=0)
        y_train = np.concatenate([y_train_0, y_train_1], axis=0)
        y_test = np.concatenate([y_test_0, y_test_1], axis=0)

        model_path = f"saved_models/{args.dataset}/robust/{BATCH}/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        model = classifyVGG(X_train, y_train, X_test, y_test, trainable=True, output_directory=model_path)
     
        benign_labels = np.concatenate([y_train_0, y_test_0], axis=0)
        benign = np.concatenate([X_train_0, X_test_0], axis=0)
        benign_logits = model.predict(benign)
        
        robust_benign_labels = np.concatenate([y_train_1, y_test_1], axis=0)
        robust_benign = np.concatenate([X_train_1, X_test_1], axis=0)
        robust_benign_logits = model.predict(robust_benign)

        print(f'benign: {benign_logits.shape}, robust_benign: {robust_benign_logits.shape}')
        if args.target is None:
            ATTACK = f"{args.attack_method}"
        else:
            ATTACK = f"{args.attack_method}{args.target}"

        label_path = f"../fpga-fashion-mnist/data/robust/evaluate/{args.attack_method}/{args.target}/y_adv.npy"
        
        TRACE_PATH = f"../data/data/{args.dataset}_robust-{args.date}"
        RATE  = args.rate
        CHANNELS = args.channels
        target = args.target
        trace_num = 10_000         

        if BATCH <= 3:
            trace_len = 36_000
        elif BATCH <= 5 :
            trace_len = 33_000
        elif BATCH <= 17:
            trace_len = 10_000
        else:
            trace_len = 6_000
        trace_path = f"{TRACE_PATH}/{ATTACK}/{str(BATCH)}"
        if not args.robust:
            output_path = f"meta/{args.dataset}/{ATTACK}/{str(BATCH)}"
        else:
            output_path = f"meta/{args.dataset}/robust-{ATTACK}/{str(BATCH)}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
        myloader.stft(nperseg=args.winsize, n_channel=CHANNELS)
        Zxxs = myloader.Zxxs
        Zxxs = Zxxs[:, :CHANNELS, :]
        # the ground_truth is labels    
        # adv_labels
        adv_labels = myloader.label
        adv_labels = np.squeeze(adv_labels)

        # Zxxs = Zxxs[truth!=target]
        # adv_labels = adv_labels[truth!=target]
        # truth = truth[truth!=target]
        if target:
            Zxxs = Zxxs[adv_labels!=y_test]
            adv_labels = adv_labels[adv_labels!=y_test]
            

            Zxxs = Zxxs[adv_labels==target]
            adv_labels = adv_labels[adv_labels==target]

        v_min = Zxxs.min(axis=(0, 1), keepdims=True)
        v_max = Zxxs.max(axis=(0, 1), keepdims=True)
        Zxxs = (Zxxs - v_min)/(v_max - v_min)

        X_adv = Zxxs
        y_adv = np.squeeze(adv_labels)
        logging.info(f"The sample number is {X_adv.shape[0]}")
        timeit(model.predict, 5, X_adv[:1])
        adv_logits    = model.predict(X_adv)
        benign_logits_lst.append(benign_logits)
        robust_benign_logits_lst.append(robust_benign_logits)
        adv_logits_lst.append(adv_logits)
        cal_accuracy(adv_logits, y_adv)

    Xorg = np.concatenate(benign_logits_lst, axis=1)
    Xadv = np.concatenate(adv_logits_lst, axis=1)
    Xrobust = np.concatenate(robust_benign_logits_lst, axis=1)

    saving_path = f"data/robust-{args.dataset}-org-{ATTACK}/"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    logging.info(f"Saving the benign logits and adversarial logits to... {saving_path}")
    np.save(f"{saving_path}Xorg.npy", Xorg)
    np.save(f"{saving_path}Xadv.npy", Xadv)
    np.save(f"{saving_path}yorg.npy", benign_labels)
    np.save(f"{saving_path}yadv.npy", y_adv)
    np.save(f"{saving_path}Xrobust.npy", Xrobust)
    np.save(f"{saving_path}yrobust.npy", robust_benign_labels)

def main():
    args = parse_args()
    logging.basicConfig(filename=f'log/{time.strftime("%Y%m%d-%H%M%S")}.log', encoding='utf-8', level=logging.INFO)
    train_derived_model(args)

if __name__ == "__main__":
    main()