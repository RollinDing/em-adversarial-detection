import numpy as np
from pip import main
import tensorflow as tf
import pandas as pd
import seaborn as sns
import argparse
import sys
import os
import logging
sys.path.append("./src/")
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import keras
from models.vae import VAE, build_1D_encoder, build_1D_decoder
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, f1_score
from sklearn.metrics import precision_recall_curve
import time 
import matplotlib
import matplotlib.cm as cm
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('seaborn-paper')

def timeit(func, iterations, *args):
    t0 = time.time()
    for _ in range(iterations):
        func(*args)
    logging.info("Time/iter: %.4f sec" % ((time.time() - t0) / iterations))


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='SkipNet Plotter')
    parser.add_argument('cmd', choices=['compare', 'histogram', 'pca', 'class'], 
        help='compare: compare between VAE, SVM;')
    parser.add_argument('--attack-method', choices=['pgd', 'cw', 'targetcw'], type=str,
        help='choose the adversarial attack method')
    parser.add_argument('--target', default=0, type=int, 
        help='the target victim class to be analyze; choose from 0-9;')
    parser.add_argument('--robust', default=False, type=bool,
        help='determine whether test the robust model or not ')
    parser.add_argument('--dataset', default='fm', choices=['fm', 'cifar10'], type=str,
        help="The dataset of both original cnn model and adversarial attack")
    args = parser.parse_args()
    return args

def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

def VAEdetector(target, Xtrain, Xorg, Xadv, ax, args):
    if not args.robust: 
        output_directory = f"saved_models/vae/{args.dataset}-{target}/"
    else:
        output_directory = f"saved_models/vae-robust/{target}/"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    EarlyStopping = keras.callbacks.EarlyStopping(monitor='reconstruction_loss', 
            min_delta=0, 
            patience=20, 
            verbose=1,
            mode='auto', 
            baseline=None, 
            restore_best_weights=True)

    file_path = output_directory+'last_model.weights'

    # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, 
    #         monitor='val_loss', 
    #         save_weights_only=True)

    learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True)

    latent_dim = 3
    # vae = VAE(build_2D_encoder(Xorg.shape[1:], latent_dim=latent_dim), build_2D_decoder(Xorg.shape[1:], latent_dim=latent_dim))
    vae = VAE(build_1D_encoder(Xorg.shape[1], latent_dim=latent_dim), build_1D_decoder(Xorg.shape[1], latent_dim=latent_dim))

    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

    if not os.path.exists(output_directory+"checkpoint"):    
        logging.info("> Fitting the VAE model------------")
        vae.fit(Xtrain, epochs=1000, batch_size=32, callbacks=[EarlyStopping])
        vae.save_weights(file_path)
    else:
        vae.load_weights(file_path)

    bl = vae.get_losses(Xorg).numpy()
    al = vae.get_losses(Xadv).numpy()
    
    logging.info(f"Timing the vae inference time of {bl.shape[0]} benign examples ")
    timeit(vae.get_losses, 10, Xorg)
    logging.info(f"Timing the vae inference time of {al.shape[0]} adversarial examples ")
    timeit(vae.get_losses, 10, Xadv)

    if target == 0:
        threshold = 0.38
    elif target==1:
        threshold = 0.40
    elif target==2:
        threshold = 0.35
    elif target==3:
        threshold = 0.35
    elif target==4:
        threshold = 0.35
    elif target==5:
        threshold = 0.35
    elif target==6:
        threshold = 0.28
    elif target==7:
        threshold = 0.30
    elif target==8:
        threshold = 0.455
    elif target==9:
        threshold = 0.30


    # select the same number of al, bl
    al = np.random.choice(al, bl.shape[0])
    y_true = np.concatenate([np.ones(bl.shape), np.zeros(al.shape)], axis=0)
    y_pred = np.concatenate([bl<threshold, al<threshold], axis=0)
    y_scores = np.concatenate([-bl, -al], axis=0)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = precision[0:-1:20]
    recall = recall[0:-1:20]
    logging.info(thresholds)

    # plot PR curve
    markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    colors = cm.tab10(np.linspace(0, 1, 10)) 

    ax.plot(recall, precision, marker=markers[target], markersize=10, label=f"{target}", lw=1, c=colors[target])

    _, _, auc_score = compute_roc(y_true, y_pred, plot=False)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    fscore = f1_score(y_true, y_pred)
    
    print('Detector ROC-AUC score: %0.4f, f1score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (auc_score, fscore, acc, precision, recall))

    print("> F1 score: ", f1_score(y_true, y_pred))
    print("> False Positive Rate:", (bl > threshold).sum()/bl.shape[0])
    print("> Detection Rate:", (al > threshold).sum()/al.shape[0])

    data = np.concatenate([bl, al], axis=0)
    lab  = np.concatenate([np.zeros(bl.shape), np.ones(al.shape)], axis=0)
    dataset = pd.DataFrame({'label': lab, 'losses': data}, columns=['label', 'losses'])

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(data=dataset, x='losses', hue='label', bins=50, palette='tab10')
    plt.axvline(x=threshold, color="red", ls='--')
    
    ax.set_xlabel("VAE Loss", fontsize=40)
    ax.set_ylabel("Sample Number", fontsize=40)
    plt.title("Anomaly Detector Loss", fontsize=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['Threshold', 'Adversarial Samples', 'Benign Samples'], fontsize=30)
    plt.savefig(f"imgs/{args.attack_method}/{args.target}.pdf")
    print(plt.gca().get_legend_handles_labels())

    plt.close()
    return auc_score

def SVMdetector(target, Xtrain, Xorg, Xadv, ax):
    nu=0.2
    kernel="rbf"
    gamma='scale'
    from sklearn import svm
    clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    clf.fit(Xtrain)
    bl = clf.predict(Xorg)
    al = clf.predict(Xadv)

    # al = np.random.choice(al, bl.shape[0])
    y_true = np.concatenate([np.ones(bl.shape), -np.ones(al.shape)], axis=0)
    y_pred = np.concatenate([bl, al], axis=0)

    bl = clf.score_samples(Xorg)
    al = clf.score_samples(Xadv)
    print(bl.shape, al.shape)
    y_scores = np.concatenate([bl, al], axis=0)
    print(y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ax.plot(recall, precision, marker='x', label=f"EM+OC-SVM")

    print("> F1 score: ", f1_score(y_true, y_pred))

def pca(XtrainA, XorgA):
    target = 1
    METHOD = "pgd"
    # VAE
    VICTIM = "train"
    ATTACK = f"{METHOD}{target}"
    index = np.concatenate([np.arange(0,100), np.arange(110,130), np.arange(170,180)])
    Xorg = np.load(f"data/train/X{VICTIM}.npy")[:, :60]
    Xadv = np.load(f"data/train/X{ATTACK}.npy")[:, :60]

    y_test = np.load(f"data/train/y{VICTIM}.npy")
    y_adv  = np.load(f"data/train/y{ATTACK}.npy")

    y_test = np.squeeze(y_test)
    Xtrain, Xtest = train_test_split(Xorg[y_test==target], test_size=0.2, random_state=42)
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    normalizer = preprocessing.Normalizer().fit(Xtrain)

    Xtrain_scaled = scaler.transform(Xtrain)
    Xtrain_normalized = normalizer.transform(Xtrain)
    Xorg_scaled = scaler.transform(Xtest)
    Xorg_normalized = normalizer.transform(Xtest)
    Xadv_scaled = scaler.transform(Xadv)
    Xadv_normalized = normalizer.transform(Xadv)

    Xtrain = Xtrain_normalized
    Xorg = Xorg_normalized
    Xadv = Xadv_normalized
    from sklearn import decomposition
    from mpl_toolkits.mplot3d import Axes3D

    pca = decomposition.PCA(n_components=3)
    pca.fit(np.concatenate([XtrainA, Xtrain], axis=0))
    Xorg = pca.transform(Xorg)
    Xadv = pca.transform(Xadv)
    XorgA = pca.transform(XorgA)

    fig = plt.figure(1, figsize=(12, 10))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    number = 200
    Xorg = Xorg[:number]
    Xadv = Xadv[:number]
    XorgA = XorgA[:number]
    # colors = ['#2095DF', '#81DF20', '#DF2020']
    colors = cm.tab10(np.linspace(0, 1, 10)) 
    ax.scatter(Xorg[:, 0], Xorg[:, 1], Xorg[:, 2], c=colors[0], edgecolor="k", label="Benign: Class 1", marker='o', s=80)
    ax.scatter(XorgA[:, 0], XorgA[:, 1], XorgA[:, 2], c=colors[1], edgecolor="k", label="Benign: Class 2", marker='<', s=80)
    ax.scatter(Xadv[:, 0], Xadv[:, 1], Xadv[:, 2], c=colors[3], edgecolor="k", label="Advesarial: Class 2->1", marker='^', s=80)
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])
    ax.set_title("Embedding of adversarial samples", fontsize=28)
    leg = plt.legend(title='Samples Embeddings', title_fontsize=40, loc='upper right', fontsize=40, markerscale=2)
    # for text, color in zip(leg.get_texts(), colors):
    #     plt.setp(text, color = color)
    plt.savefig("imgs/3D.pdf")

def load_data(target, VICTIM, ATTACK, args):
    index = np.concatenate([np.arange(0,100), np.arange(110,130), np.arange(170,180)])

    if args.dataset=="cifar10":
        Xorg = np.load(f"data/{args.dataset}-{VICTIM}-{ATTACK}/Xorg.npy")[:, :40]
        Xadv = np.load(f"data/{args.dataset}-{VICTIM}-{ATTACK}/Xadv.npy")[:, :40]

        y_test = np.load(f"data/{args.dataset}-{VICTIM}-{ATTACK}/yorg.npy")
        y_adv  = np.load(f"data/{args.dataset}-{VICTIM}-{ATTACK}/yadv.npy")
    else:
        # Xorg = np.load(f"data/{args.dataset}-{VICTIM}-{ATTACK}{target}/Xorg.npy")[:, :60]
        # Xadv = np.load(f"data/{args.dataset}-{VICTIM}-{ATTACK}{target}/Xadv.npy")[:, :60]

        # y_test = np.load(f"data/{args.dataset}-{VICTIM}-{ATTACK}{target}/yorg.npy")
        # y_adv  = np.load(f"data/{args.dataset}-{VICTIM}-{ATTACK}{target}/yadv.npy")
        Xorg = np.load(f"data/train/X{VICTIM}.npy")[:, :60]
        Xadv = np.load(f"data/train/X{ATTACK}.npy")[:, :60]

        y_test = np.load(f"data/train/y{VICTIM}.npy")
        y_adv  = np.load(f"data/train/y{ATTACK}.npy")

    y_test = np.squeeze(y_test)
    Xtrain, Xtest = train_test_split(Xorg[y_test==target], test_size=0.2, random_state=42)
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    normalizer = preprocessing.Normalizer().fit(Xtrain)

    Xtrain_scaled = scaler.transform(Xtrain)
    Xtrain_normalized = normalizer.transform(Xtrain)
    Xorg_scaled = scaler.transform(Xtest)
    Xorg_normalized = normalizer.transform(Xtest)
    Xadv_scaled = scaler.transform(Xadv)
    Xadv_normalized = normalizer.transform(Xadv)

    Xtrain = Xtrain_normalized
    Xorg = Xorg_normalized
    Xadv = Xadv_normalized
    XtrainA = Xtrain
    XorgA = Xorg
    return Xtrain, Xorg, Xadv

def draw_compare(args):
    fig, ax = plt.subplots()
    target= args.target
    scores = []
    METHOD = "pgd"
    # VAE
    VICTIM = "train"
    ATTACK = f"{METHOD}{target}"

    Xtrain, Xorg, Xadv = load_data(target, VICTIM, ATTACK, args)

    recall = np.load("data/recall_vae.npy")
    precision = np.load("data/precision_vae.npy")
    ax.plot(recall, precision, marker='*', label=f"NIC+OC-SVM")

    recall = np.load("data/recall_svm.npy")
    precision = np.load("data/precision_svm.npy")
    ax.plot(recall, precision, marker='^', label=f"NIC+VAE")
    

    SVMdetector(target, Xtrain, Xorg, Xadv, ax)
    score = VAEdetector(target, Xtrain, Xorg, Xadv, ax)
    ax.set_xlabel("Recall", fontsize=20)
    ax.set_ylabel("Precision", fontsize=20)
    ax.legend(fontsize=20)
    ax.grid()
    ax.set_title("Precision-Recall Graph", fontsize=20)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.5, 1.0)
    scores.append(score)
    fig.tight_layout()
    plt.savefig("imgs/targetedPGD/PRcurve-compare.pdf")
    print(scores)

def draw_pca(args):
    target = 2
    METHOD = "pgd"
    # VAE
    VICTIM = "train"
    ATTACK = f"{METHOD}{target}"
    Xtrain, Xorg, _ = load_data(target, VICTIM, ATTACK, args)
    pca(Xtrain, Xorg)

def draw_classes(args):
    fig, ax = plt.subplots(figsize=(13, 10))
    
    scores = []
    METHOD = "pgd"
    # VAE
    VICTIM = "train"
    
    for target in range(10):
        ATTACK = f"{METHOD}{target}"
        Xtrain, Xorg, Xadv = load_data(target, VICTIM, ATTACK, args)
        score = VAEdetector(target, Xtrain, Xorg, Xadv, ax, args)
    
    ax.set_xlabel("Recall", fontsize=40)
    ax.set_ylabel("Precision", fontsize=40)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # ax.legend(fontsize=28)
    legend = ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1, fontsize=28, title="Target", markerscale=2)
    plt.setp(legend.get_title(),fontsize=40)
    ax.grid()
    ax.set_title("Precision-Recall Graph", fontsize=40)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.5, 1.0)
    scores.append(score)
    fig.tight_layout()
    plt.savefig("imgs/targetedPGD/PRcurve.pdf")
    print(scores)

def draw_histogram(args):
    fig, ax = plt.subplots(figsize=(10, 10))
    target = args.target
    scores = []
    METHOD = args.attack_method
    VICTIM = "train"

    ATTACK = f"{METHOD}{target}"
    Xtrain, Xorg, Xadv = load_data(target, VICTIM, ATTACK, args)
    score = VAEdetector(target, Xtrain, Xorg, Xadv, ax, args)
    # score = SVMdetector(target, Xtrain, Xorg, Xadv, ax)
    ax.set_xlabel("Recall", fontsize=40)
    ax.set_ylabel("Precision", fontsize=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(fontsize=20)
    ax.grid()
    ax.set_title("Precision-Recall Graph", fontsize=40)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    scores.append(score)
    fig.tight_layout()
    plt.savefig(f"imgs/{args.dataset}-{METHOD}{target}.png")
    plt.close()
    print(scores)

def main():
    args = parse_args()
    logging.basicConfig(filename=f'log/{time.strftime("%Y%m%d-%H%M%S")}.log', encoding='utf-8', level=logging.INFO)
    if args.cmd == "compare":
        draw_compare(args)
    elif args.cmd == "histogram":
        draw_histogram(args)
    elif args.cmd == "pca":
        draw_pca(args)
    elif args.cmd == 'class':
        draw_classes(args)

if __name__ == "__main__":
    main()