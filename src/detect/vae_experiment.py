import numpy as np
from pip import main
import tensorflow as tf
import pandas as pd
import seaborn as sns
import sys
import os
sys.path.append("./src/")
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import keras
from models.vae import VAE, build_1D_encoder, build_1D_decoder
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, f1_score
from sklearn.metrics import precision_recall_curve
import matplotlib
import matplotlib.cm as cm

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('seaborn-paper')

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

def VAEdetector(target, Xtrain, Xorg, Xadv, ax, latent, targetadv):
    output_directory = f"saved_models/latent/{latent}/"

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

    learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True)

    latent_dim = latent
    # vae = VAE(build_2D_encoder(Xorg.shape[1:], latent_dim=latent_dim), build_2D_decoder(Xorg.shape[1:], latent_dim=latent_dim))
    vae = VAE(build_1D_encoder(Xorg.shape[1], latent_dim=latent_dim), build_1D_decoder(Xorg.shape[1], latent_dim=latent_dim))

    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

    if not os.path.exists(output_directory+"checkpoint"):    
        print("> Fitting the VAE model------------")
        vae.fit(Xtrain, epochs=1000, batch_size=32, callbacks=[EarlyStopping])
        vae.save_weights(file_path)
    else:
        path = output_directory
        vae.load_weights(file_path)



    bl = vae.get_losses(Xorg).numpy()
    al = vae.get_losses(Xadv).numpy()

    if target == 0:
        threshold = 0.26
    elif target==1:
        threshold = 0.30
    elif target==2:
        threshold = 0.30
    elif target==3:
        threshold = 0.30
    elif target==4:
        threshold = 0.28
    elif target==5:
        threshold = 0.30
    elif target==6:
        threshold = 0.25
    elif target==7:
        threshold = 0.30
    elif target==8:
        threshold = 0.23
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
    print(thresholds)
    print(thresholds)
    markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    colors = cm.tab10(np.linspace(0, 1, 10)) 
    # plot PR curve
    ax.plot(recall, precision, marker=markers[latent], label=f"{latent}", lw=0.5, c=colors[latent-2])


    _, _, auc_score = compute_roc(y_true, y_pred, plot=False)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    fscore = f1_score(y_true, y_pred)
    
    print('Detector ROC-AUC score: %0.4f, f1score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (auc_score, fscore, acc, precision, recall))

    print("> F1 score: ", f1_score(y_true, y_pred))
    print("> Benign rate:", (bl < threshold).sum()/bl.shape[0])
    print("> Advesarial rate:", (al < threshold).sum()/al.shape[0])

    data = np.concatenate([bl, al], axis=0)
    lab  = np.concatenate([np.zeros(bl.shape), np.ones(al.shape)], axis=0)
    dataset = pd.DataFrame({'label': lab, 'losses': data}, columns=['label', 'losses'])
    
    return auc_score

def load_data(target, VICTIM, ATTACK):
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
    XtrainA = Xtrain
    XorgA = Xorg
    return Xtrain, Xorg, Xadv

def main():
    
    fig, ax = plt.subplots(figsize=(14, 10))
    scores = []

    targetadv = 8
    METHOD = "pgd"
    # VAE
    VICTIM = "train"
    ATTACK = f"{METHOD}{targetadv}"


    _, _, Xadv = load_data(targetadv, VICTIM, ATTACK)
    target = 8
    VICTIM = "train"
    ATTACK = f"{METHOD}{targetadv}"
    Xtrain, Xorg, _ = load_data(target, VICTIM, ATTACK)    
    # pca(XtrainA, XorgA)
    for latent in range(2, 12):
        score = VAEdetector(target, Xtrain, Xorg, Xadv, ax, latent, targetadv)
        scores.append(score)

    print(scores)
    ax.set_xlabel("Recall", fontsize=40)
    ax.set_ylabel("Precision", fontsize=40)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.5, 1.0)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # legend = ax.legend(ncol=2, title="Latent Space Size", fontsize=20, loc='lower left')
    legend = ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1, fontsize=28, title="Latent Size", markerscale=2)
    plt.setp(legend.get_title(),fontsize=40)
    ax.grid()
    ax.set_title("PR Curves of Different Latent Sizes", fontsize=40)
    fig.tight_layout()
    plt.savefig("imgs/targetedPGD/PRcurve-latent.pdf")

main()