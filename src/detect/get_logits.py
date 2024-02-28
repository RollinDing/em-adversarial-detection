"""
Get Benign and Adversarial Logits from the derived models
"""
import sys
import tensorflow as tf
from foolbox.attacks.base import Attack
from tensorflow.python.ops.gen_math_ops import Log
sys.path.append("./src/")
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.classify import classifyVGG
from utils.loader import Loader
from sklearn.metrics import classification_report, confusion_matrix


def cal_accuracy(y_pred, y_true):
    # convert the predicted from binary to integer 
    y_pred = np.argmax(y_pred , axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred))
    # plt.figure()
    # plt.imshow(confusion_matrix(y_true, y_pred))
    # plt.savefig("test.png")
    # plt.close()
    print('Classification Report')
    print(classification_report(y_true, y_pred))

def save_benign_logits():
    TRACE_PATH = "../data/data/fm_lenet-20211129"
    benign_logits_lst = []

    # Get Benign Logits
    for BATCH in range(28):
        ATTACK = "train"
        RATE  = 1
        trace_num = 60_000
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
        myloader.stft(nperseg=256)
        
        Zxxs = myloader.Zxxs
        Zxxs = Zxxs[:, :22, :]
        labels = myloader.label
        v_min = Zxxs.min(axis=(0, 1), keepdims=True)
        v_max = Zxxs.max(axis=(0, 1), keepdims=True)
        Zxxs = (Zxxs - v_min)/(v_max - v_min)
        X_train, X_test, y_train, y_test = train_test_split(Zxxs, labels, test_size=0.2, random_state=42)
        model = classifyVGG(X_train, y_train, X_test, y_test, trainable=True, output_directory=f"saved_models/vgg/{ATTACK}/{BATCH}/")
        benign_labels = np.concatenate([y_train, y_test], axis=0)
        benign = np.concatenate([X_train, X_test], axis=0)
        benign_logits = model.predict(benign)
        benign_logits_lst.append(benign_logits)

    Xorg = np.concatenate(benign_logits_lst, axis=1)
    np.save(f"data/train/X{ATTACK}.npy", Xorg)
    np.save(f"data/train/y{ATTACK}.npy", benign_labels)

def save_adversarial_logits():
    TRACE_PATH = "../data/data/fm_lenet-20220111"
    ATTACK = f"{METHOD}{target}"
    trace_num = 10_000
    RATE = 1
    adv_logits_lst = []
    for BATCH in range(18):
        if BATCH <= 3:
            trace_len = 36_000
        elif BATCH <= 5 :
            trace_len = 33_000
        elif BATCH <= 17:
            trace_len = 10_000
        else:
            trace_len = 6_000
        label_path = f"../data/data/fm/targetDeepFool/{target}/y_adv.npy"
        trace_path = f"{TRACE_PATH}/{ATTACK}/{str(BATCH)}"
        output_path = f"meta/raw/{ATTACK}/{str(BATCH)}"
        myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
        
        myloader.stft(nperseg=256)
        Zxxs = myloader.Zxxs
        Zxxs = Zxxs[:, :22, :]

        output_directory=f"saved_models/vgg/train/{BATCH}/"
        path = output_directory+"best_model.hdf5"
        model = tf.keras.models.load_model(path)
        # the ground_truth is labels    

        truth = np.load("../data/data/fm/org/y_test.npy")
        # adv_labels
        adv_labels = myloader.label
        adv_labels = np.squeeze(adv_labels)

        Zxxs = Zxxs[truth!=target]
        adv_labels = adv_labels[truth!=target]
        
        y_truth = truth[truth!=target]
        y_truth = y_truth[adv_labels==target]

        Zxxs = Zxxs[adv_labels==target]
        adv_labels = adv_labels[adv_labels==target]

        v_min = Zxxs.min(axis=(0, 1), keepdims=True)
        v_max = Zxxs.max(axis=(0, 1), keepdims=True)
        Zxxs = (Zxxs - v_min)/(v_max - v_min)

        X_adv = Zxxs
        y_adv = np.squeeze(adv_labels)

       
        adv_logits    = model.predict(X_adv)
        cal_accuracy(adv_logits, y_adv)
        cal_accuracy(adv_logits, y_truth)
        adv_logits_lst.append(adv_logits)

    Xadv = np.concatenate(adv_logits_lst, axis=1)
    np.save(f"data/train/X{ATTACK}.npy", Xadv)
    np.save(f"data/train/y{ATTACK}.npy", y_adv)


if __name__ == "__main__":
    # save_benign_logits()
    for target in range(0, 10):
        METHOD = "cw"
        save_adversarial_logits()