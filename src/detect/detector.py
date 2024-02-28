"""
THIS FILE IS A GENERAL PROCEDURE FOR OUR ADVERSARIAL DETECTOR
"""
from collections import Counter
import sys
from tensorflow.keras import optimizers

from tensorflow.python.ops.gen_math_ops import Log
sys.path.append("./src/")
import numpy as np 
import tensorflow as tf
import tensorflow.keras as keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from download import preprocessor
from models.classify import classifyVGG
from utils.normalizer import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from activation import ActivationExtracter
from utils.loader import Loader
from sklearn.metrics import f1_score
from sklearn import preprocessing
from scipy.special import softmax
from models.vae import VAE, build_1D_encoder, build_1D_decoder, plot_label_clusters, plot_label_clusters_3D, build_2D_decoder, build_2D_encoder
import argparse
import logging
import time


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='Adversarial Detector: Given the adversarial logits, using the detector (SVM or VAE) to detect the adversarial attacker')


def cal_accuracy(y_pred, y_true):
    # convert the predicted from binary to integer 
    y_pred = np.argmax(y_pred , axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred))
    plt.figure()
    plt.imshow(confusion_matrix(y_true, y_pred))
    plt.savefig("test.png")
    plt.close()
    print('Classification Report')
    print(classification_report(y_true, y_pred))
    keras.backend.clear_session()

def timeit(func, iterations, *args):
    t0 = time.time()
    for _ in range(iterations):
        func(*args)
    print("Time/iter: %.4f sec" % ((time.time() - t0) / iterations))

def one_class_SVM(logits_benign_lst, nu=0.01, kernel="rbf", gamma='scale'):
    """
    conduct one class SVM to distinguish outliers(adversarial examples)
    """
    svm_lst = []
    y_pred_trains = []
    y_pred_tests = []
    y_test_scores = []
    for logits_benign in logits_benign_lst:

        X_train = logits_benign[:int(logits_benign.shape[0]*0.75), :]
        X_test = logits_benign[int(logits_benign.shape[0]*0.75):, :]
        clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        y_pred_trains.append(y_pred_train)
        y_pred_tests.append(y_pred_test)
        y_test_scores.append(clf.score_samples(X_test))
        svm_lst.append(clf)
        print("> Train accuracy ", compute_logits_accuracy([y_pred_train]))
        print("> Test accuracy ", compute_logits_accuracy([y_pred_test]))
    return svm_lst, np.asarray(y_test_scores).mean(axis=0)

def compute_logits_accuracy(y_pred_lst):
    """
    Given a y_pred_lst, compute the error rate
    """
    results = np.ones(y_pred_lst[0].shape)
    prediction = np.ones(y_pred_lst[0].shape)
    for y_pred in y_pred_lst:
        prediction = np.logical_and((y_pred==1), prediction)
    return results[prediction].size/results.shape[0]

def one_class_SVM_evaluate(logits_adv_lst, svm_lst):
    """
    Evaluate the performance of one class SVM on adversarial examples
    """
    
    y_preds = []
    y_scores = []
    for logits_adv, clf in zip(logits_adv_lst, svm_lst):
        y_pred = clf.predict(logits_adv)
        
        y_score = clf.score_samples(logits_adv)
        y_scores.append(y_score)
        y_preds.append(y_pred)
        print("> One fail detection rate ", 1-compute_logits_accuracy([y_pred]))
    np.save("results/ypred.npy", np.asarray(y_preds))
    return np.asarray(y_scores).mean(axis=0)

def one_class_SVM_decision(logits_lst, svm_lst, classIdx, name="adv"):
    """
    Compute the decision of each class
    """
    dists = []
    for logits, clf in zip(logits_lst, svm_lst):
        y_dist = clf.decision_function(logits)
        dists.append(y_dist.reshape(-1, 1))
    dists = np.concatenate(dists, axis=1)
    np.save(f"results/svm_decision/{name}_class_{classIdx}.npy", dists)
    print(name, (dists.sum(axis=1)<0).sum()/dists.shape[0])

def visualize(fig, ax, logits):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2D = pca.fit_transform(logits)
    ax.scatter(X_2D[:, 0], X_2D[:, 1])

def visualize_wrapper(benign, adv):
    fig, ax = plt.subplots()
    visualize(fig, ax, benign)
    visualize(fig, ax, adv)
    plt.show()

def clusterSVM():
    for classIdx in range(10):
        print("Detecting adversarial examples in class ", classIdx)
        # benign_lst = [benign_logits[np.squeeze(y_test)==classIdx] for benign_logits in benign_logits_lst]
        benign_lst = [benign_logits[np.squeeze(np.concatenate([y_test], axis=0))==classIdx] for benign_logits in benign_logits_lst]
        adv_lst    = [adv_logits[np.squeeze(y_adv)==classIdx] for adv_logits in adv_logits_lst]
        
        # benign_lst = [np.concatenate(benign_lst, axis=1)]
        # adv_lst    = [np.concatenate(adv_lst, axis=1)]

        # visualize_wrapper(benign_lst[0], adv_lst[0])

        svm_lst, _ = one_class_SVM(benign_lst)
        one_class_SVM_evaluate(adv_lst, svm_lst)
        # one_class_SVM_decision(benign_lst, svm_lst, classIdx, name="benign")
        # one_class_SVM_decision(adv_lst, svm_lst, classIdx)

def main():
    args = parse_args()

if __name__ == "__main__":
    # TRACE_PATH = "../data/data/fm_lenet-20220321"
    # benign_logits_lst = []
    # adv_logits_lst = []
    # f1_lst = []
    # target = 6
    
    # for BATCH in range(6):
    #     ATTACK = "org"
    #     RATE  = 1
    #     trace_num = 10_000
    #     if BATCH > 3:
    #         trace_len = 33_000
    #     else:
    #         trace_len = 36_000
        
    #     label_path = f"../data/data/fm/{ATTACK}/y_adv.npy"
    #     trace_path = f"{TRACE_PATH}/{ATTACK}/{str(BATCH)}"
    #     output_path = f"meta/raw/robust-{ATTACK}/{str(BATCH)}"
    #     myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
    #     myloader.stft(nperseg=256)
        
    #     Zxxs = myloader.Zxxs
    #     Zxxs = Zxxs[:, :22, :]
    #     labels = myloader.label
    #     v_min = Zxxs.min(axis=(0, 1), keepdims=True)
    #     v_max = Zxxs.max(axis=(0, 1), keepdims=True)
    #     Zxxs = (Zxxs - v_min)/(v_max - v_min)
    #     X_train, X_test, y_train, y_test = train_test_split(Zxxs, labels, test_size=0.2, random_state=42)
    #     model = classifyVGG(X_train, y_train, X_test, y_test, trainable=True, output_directory=f"saved_models/vgg/{ATTACK}/{BATCH}/")
    #     benign_labels = np.concatenate([y_train, y_test], axis=0)
    #     benign = np.concatenate([X_train, X_test], axis=0)
    #     benign_logits = model.predict(benign)
    #     # benign_logits = model.predict(X_test)
    #     # load adversarial examples
        
    #     ATTACK = f"pgd{target}"
    #     label_path = f"../data/data/fm/{ATTACK}/y_adv.npy"
    #     trace_path = f"{TRACE_PATH}/{ATTACK}/{str(BATCH)}"
    #     output_path = f"meta/raw/{ATTACK}/{str(BATCH)}"
    #     myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
    #     myloader.stft(nperseg=256)
    #     Zxxs = myloader.Zxxs
    #     Zxxs = Zxxs[:, :22, :]

    #     # the ground_truth is labels    
    #     truth = labels
    #     # adv_labels
    #     adv_labels = myloader.label
    #     adv_labels = np.squeeze(adv_labels)

    #     # Zxxs = Zxxs[truth!=target]
    #     # adv_labels = adv_labels[truth!=target]
        
    #     Zxxs = Zxxs[adv_labels==target]
    #     adv_labels = adv_labels[adv_labels==target]

    #     v_min = Zxxs.min(axis=(0, 1), keepdims=True)
    #     v_max = Zxxs.max(axis=(0, 1), keepdims=True)
    #     Zxxs = (Zxxs - v_min)/(v_max - v_min)

    #     X_adv = Zxxs
    #     y_adv = np.squeeze(adv_labels)
       
    #     adv_logits    = model.predict(X_adv)
    #     benign_logits_lst.append(benign_logits)
    #     adv_logits_lst.append(adv_logits)
    #     cal_accuracy(adv_logits, y_adv)
    
          
    # Xorg = np.concatenate(benign_logits_lst, axis=1)
    # Xadv = np.concatenate(adv_logits_lst, axis=1)
    # y_test =benign_labels

    # np.save(f"data/Xorg_{target}.npy", Xorg)
    # np.save(f"data/Xadv_{target}.npy", Xadv)
    # np.save(f"data/yorg_{target}.npy", benign_labels)
    # np.save(f"data/yadv_{target}.npy", y_adv)
    
    # VAE
    target = 8
    ATTACK = "targetcw"
    Xorg = np.load(f"data/org-{ATTACK}{target}/Xorg.npy")
    Xadv = np.load(f"data/org-{ATTACK}{target}/Xadv.npy")
    y_test = np.load(f"data/org-{ATTACK}{target}/yorg.npy")
    y_adv  = np.load(f"data/org-{ATTACK}{target}/yadv.npy")

    y_test = np.squeeze(y_test)
    Xtrain, Xtest = train_test_split(Xorg[y_test==target], test_size=0.1, random_state=41)
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    normalizer = preprocessing.Normalizer().fit(Xtrain)
    Xorg_scaled = scaler.transform(Xtest)
    Xorg_normalized = normalizer.transform(Xtest)

    Xadv_scaled = scaler.transform(Xadv)
    Xadv_normalized = normalizer.transform(Xadv)

    Xorg = Xorg_normalized
    Xadv = Xadv_normalized

    EarlyStopping = keras.callbacks.EarlyStopping(monitor='reconstruction_loss', 
            min_delta=0, 
            patience=50, 
            verbose=1,
    		mode='auto', 
            baseline=None, 
            restore_best_weights=True)

    learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True)

    latent_dim = 6
    # vae = VAE(build_2D_encoder(Xorg.shape[1:], latent_dim=latent_dim), build_2D_decoder(Xorg.shape[1:], latent_dim=latent_dim))
    vae = VAE(build_1D_encoder(Xorg.shape[1], latent_dim=latent_dim), build_1D_decoder(Xorg.shape[1], latent_dim=latent_dim))
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
    vae.fit(Xorg, epochs=1000, batch_size=32)
    
    timeit(vae.get_losses, 10, Xadv[:1])
    bl = vae.get_losses(Xtest).numpy()
    al = vae.get_losses(Xadv).numpy()
    threashold = 0.30
    print("False Positive rate", (bl > threashold).sum()/bl.shape[0])
    print("Detection rate", (al > threashold).sum()/al.shape[0])

    data = np.concatenate([bl, al], axis=0)
    lab  = np.concatenate([np.zeros(bl.shape), np.ones(al.shape)], axis=0)
    dataset = pd.DataFrame({'label': lab, 'losses': data}, columns=['label', 'losses'])
    plt.figure()
    sns.histplot(data=dataset, x='losses', hue='label')
    plt.show()
    # plot embedding results of benign examples
    fig, ax = plt.subplots()

    plot_label_clusters(fig, ax, vae, Xorg, y_test)
    plot_label_clusters(fig, ax, vae, Xadv, y_adv, color='pink')


    # plt.show()
    # visualize_wrapper(Xorg[y_test==target], Xadv)
    
    # plot_label_clusters_3D(vae, Xorg, y_test)
    # plot_label_clusters_3D(vae, Xadv, y_adv)


    # classIdx = target
    # print(">SVM : Detecting adversarial examples in class ", classIdx)
    # # benign_lst = [benign_logits[np.squeeze(y_test)==classIdx] for benign_logits in benign_logits_lst]
    # benign_lst = [benign_logits[np.squeeze(y_test)==classIdx] for benign_logits in benign_logits_lst]
    # adv_lst    = [adv_logits[np.squeeze(y_adv)==classIdx] for adv_logits in adv_logits_lst]
    # # benign_lst = [np.concatenate(benign_lst, axis=1)]
    # # adv_lst    = [np.concatenate(adv_lst, axis=1)]

    # # visualize_wrapper(benign_lst[0], adv_lst[0])

    # svm_lst, _ = one_class_SVM(benign_lst)
    # one_class_SVM_evaluate(adv_lst, svm_lst)