from secrets import choice
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
        help='Select the batches to analyze from [1-27]')
    parser.add_argument('--dataset', default='fm', choices=['fm', 'cifar10'], type=str,
        help="The dataset of both original cnn model and adversarial attack")
    parser.add_argument('--victim-model', default='lenet', choices=['lenet', 'vgg'], type=str,
        help="the victim model type")
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


def train_derived_model(args):
    """
    Train the derived model using orginal data
    """
    if args.robust:
        TRACE_PATH = f"../data/data/{args.dataset}_robust-{args.date}"
    elif args.dataset == "cifar10" and args.victim_model == "vgg":
        TRACE_PATH = f"../data/em-adversarial-data/{args.dataset}_{args.victim_model}-{args.date}"
    else:
        TRACE_PATH = f"../data/data/{args.dataset}_{args.victim_model}-{args.date}"

    benign_logits_lst = []
    adv_logits_lst = []
    BATCH_NUM = args.batch_num

    for BATCH in range(0, BATCH_NUM):
        ATTACK = "train"
        RATE  = args.rate
        CHANNELS = args.channels
        target = args.target
        trace_num = 10_000

        if args.dataset == 'fm':
            if BATCH <= 3:
                trace_len = 36_000
            elif BATCH <= 5 :
                trace_len = 33_000
            elif BATCH <= 17:
                trace_len = 10_000
            else:
                trace_len = 6_000
        elif args.dataset=='cifar10' and args.victim_model=='vgg':
            if BATCH <= 0:
                trace_len = 56_000
            elif BATCH <= 5 :
                trace_len = 39_0000
            elif BATCH <= 17:
                trace_len = 12_000
        elif args.dataset=='cifar10' and args.victim_model=='lenet':
            if BATCH <= 3:
                trace_len = 41_000
            elif BATCH <= 5 :
                trace_len = 36_000
            elif BATCH <= 17:
                trace_len = 12_000

        logging.info(f"> The {BATCH}th BATCH:")    
        logging.info(f"> The input trace length is {trace_len}")
        if args.dataset == 'fm':
            label_path = f"../data/data/{args.dataset}/{ATTACK}/y_adv.npy"
            # label_path = f"../fpga-fashion-mnist/data/robust/{ATTACK}/y_adv.npy"
        elif args.dataset == 'cifar10' and args.victim_model=='vgg':
            label_path = f"../data/em-adversarial-data/y_{ATTACK}.npy"
        elif args.dataset == 'cifar10' and args.victim_model=='lenet':
            label_path = f"../data/data/{args.dataset}/zero_mean/y_test_cifar_zero_mean.npy"

        trace_path = f"{TRACE_PATH}/{ATTACK}/{str(BATCH)}"
        
        if args.robust:
            output_path = f"meta/{args.dataset}/robust-{ATTACK}/{str(BATCH)}"
        else:
            # output_path = f"meta/{args.dataset}-{args.victim_model}/{ATTACK}/{str(BATCH)}"
            output_path = f"meta/{args.dataset}/{ATTACK}/{str(BATCH)}"


        logging.info(f"output path is {output_path}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
        if args.dataset=="cifar" and args.victim_model=='vgg':
            myloader.stft(fs=5e8, nperseg=args.winsize, n_channel=CHANNELS, band=True)
        else:
            myloader.stft(fs=2e10, nperseg=args.winsize, n_channel=CHANNELS)
        
        Zxxs = myloader.Zxxs
        # Zxxs = Zxxs[:, :CHANNELS, :]
        labels = myloader.label
    
        # n_components = 20
        # pca = PCA(n_components)
        # Zxxs_transformed = np.zeros([Zxxs.shape[0], Zxxs.shape[1], n_components])
        # for i in range(CHANNELS):
        #     Zxxs_transformed[:, i, :] = pca.fit_transform(Zxxs[:, i, :])
        # for i in range(CHANNELS):
        #     Zxxs[:, i, :] = (Zxxs[:, i, :]-Zxxs[:, i, :].mean())/Zxxs[:, i, :].std()

        v_min = Zxxs.min(axis=(0, 1), keepdims=True)
        v_max = Zxxs.max(axis=(0, 1), keepdims=True)
        Zxxs = (Zxxs - v_min)/(v_max - v_min + 1e-4)

        # v_min = Zxxs.min(axis=(0, 1), keepdims=True)
        # v_max = Zxxs.max(axis=(0, 1), keepdims=True)
        # Zxxs = (Zxxs - v_min)/(v_max - v_min)
        X_train, X_test, y_train, y_test = train_test_split(Zxxs, labels, test_size=0.2, random_state=42)
        if args.robust:
            model_path = f"saved_models/{args.dataset}/robust-{ATTACK}/{BATCH}/"
        else:
            # model_path = f"saved_models/{args.dataset}-{args.victim_model}/{ATTACK}/{BATCH}/"
            model_path = f"saved_models/{args.dataset}/{ATTACK}/{BATCH}/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        model = classifyVGG(X_train, y_train, X_test, y_test, trainable=True, output_directory=model_path)
        benign_labels = np.concatenate([y_train, y_test], axis=0)
        benign = np.concatenate([X_train, X_test], axis=0)

        # for idx in range(2, 10):
        #     benign_logits = model.predict(X_test[idx:idx+1])
        #     print(idx)
        #     print(f"The logits of example in class {y_test[idx]} is on {BATCH} is", benign_logits)
        
        continue
        if args.gradcam:
            print("Do gradCAM")
            for n in range(10):    
                y_test = np.squeeze(y_test)
                X = np.mean(np.expand_dims(X_test, axis=3)[y_test==n], keepdims=True, axis=0)

                imgpath = f"imgs/{args.dataset}/{args.attack_method}/{BATCH}/"
                if not os.path.exists(imgpath):
                    os.makedirs(imgpath)
                # df_cm = pd.DataFrame(heatmap)
                fig, ax = plt.subplots(figsize=(20, 18))
                plt.xticks([])
                plt.yticks([])
                
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(8)
                plt.imshow(X[0], aspect='auto')
                plt.colorbar()
                plt.savefig(imgpath + f"spectrogram{n}.png")
                plt.close()

                cam = GradCAM(model, n)
                heatmap = cam.compute_heatmap(X)
                fig, ax = plt.subplots(figsize=(20, 18))
                plt.axis('on')
                plt.xticks([])
                plt.yticks([])
                
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(8)
                plt.imshow(heatmap, aspect='auto', cmap='plasma')
                plt.colorbar()

                plt.savefig(imgpath + f"spectrogramCAM{n}.png")
                plt.close()
                
        if target is None:
            ATTACK = f"{args.attack_method}"
        else:
            ATTACK = f"{args.attack_method}{target}"

        if args.dataset == "cifar10":
            label_path = f"../data/em-adversarial-data/y_{ATTACK}.npy"
        elif not args.robust:
            label_path = f"../data/data/{args.dataset}/{args.attack_method}/{target}/y_adv.npy"
        else:
            if not args.target:
                # label_path = f"../fpga-fashion-mnist/data/robust/{args.attack_method}/y_adv.npy"
                label_path = f"../data/data/{args.dataset}/{ATTACK}/y_adv.npy"
            else:
                label_path = f"../fpga-fashion-mnist/data/robust/{args.attack_method}/{target}/y_adv.npy"
                
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
        truth = labels
        # adv_labels
        adv_labels = myloader.label
        adv_labels = np.squeeze(adv_labels)

        # Zxxs = Zxxs[truth!=target]
        # adv_labels = adv_labels[truth!=target]
        # truth = truth[truth!=target]
        if target:
            Zxxs = Zxxs[adv_labels!=truth]
            adv_labels = adv_labels[adv_labels!=truth]
            

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
        adv_logits_lst.append(adv_logits)
        cal_accuracy(adv_logits, y_adv)

    exit()
    Xorg = np.concatenate(benign_logits_lst, axis=1)
    Xadv = np.concatenate(adv_logits_lst, axis=1)

    saving_path = f"data/{args.dataset}-org-{ATTACK}/"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    logging.info(f"Saving the benign logits and adversarial logits to... {saving_path}")
    np.save(f"{saving_path}Xorg.npy", Xorg)
    np.save(f"{saving_path}Xadv.npy", Xadv)
    np.save(f"{saving_path}yorg.npy", benign_labels)
    np.save(f"{saving_path}yadv.npy", y_adv)

def main():
    args = parse_args()
    logging.basicConfig(filename=f'log/{time.strftime("%Y%m%d-%H%M%S")}.log', encoding='utf-8', level=logging.INFO)
    train_derived_model(args)

if __name__ == "__main__":
    main()