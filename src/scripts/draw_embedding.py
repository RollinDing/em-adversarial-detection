import sys
sys.path.append("./src/")
import numpy as np
import matplotlib.pyplot as plt
from utils.loader import Loader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from models.classify import classifyVGG
import matplotlib.cm as cm
import tensorflow as tf 
import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('seaborn-paper')

BATCH = 1
names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker","Bag","Ankle boot"]
def load_train():
    TRACE_PATH = f"../data/data/fm_lenet-20211129/train/"
    ATTACK = "train"
    RATE  = 1

    trace_num = 60_000
    trace_len = 36_000

    label_path = f"../data/data/fm/{ATTACK}/y_adv.npy"
    trace_path = f"{TRACE_PATH}/{str(BATCH)}"
    output_path = f"meta/fm/{ATTACK}/{str(BATCH)}"
    myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
    myloader.stft(fs=2e10, nperseg=256, n_channel=22)
    Zxxs = myloader.Zxxs
    v_min = Zxxs.min(axis=(0, 1), keepdims=True)
    v_max = Zxxs.max(axis=(0, 1), keepdims=True)
    Zxxs = (Zxxs - v_min)/(v_max - v_min + 1e-4)
    y = myloader.label
    return Zxxs, y

def embedding(X):
    print("Do PCA")
    pca = TSNE(n_components=2)
    return pca

def display(Xtransformed, labels):
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.tick_params(
        axis='both',        
        which='both', 
        left=False,   
        labelleft=False, 
        bottom=False,      
        top=False,         
        labelbottom=False)
    markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    colors = cm.tab10(np.linspace(0, 1, 10)) 
    ax.set_title("Feature Space Embedding", fontsize=40)
    for i in range(10):
        Xp = Xtransformed[labels==i]
        c  = colors[i]
        name = names[i]
        plt.scatter(Xp[:, 0], Xp[:, 1], c=c, marker=markers[i], s=10, label=name)
    # plt.legend(fontsize=24, loc=2, ncol=1, bbox_to_anchor=(1.05, 1))
    legend = ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1, fontsize=28, title="Class Name", markerscale=6)
    plt.setp(legend.get_title(),fontsize=40)
    plt.tight_layout()
    plt.savefig("imgs/feature-space-representation.pdf")
    plt.close()

def main():
    X, y = load_train()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    model_path = f"saved_models/fm/train/{BATCH}/"
    model = classifyVGG(Xtrain, ytrain, Xtest, ytest, trainable=True, output_directory=model_path)
    
    aux_model = tf.keras.Model(inputs=model.inputs,
                           outputs=model.layers[-5].output)
    
    output = aux_model.predict(Xtest[:2000]).reshape(2000, -1)
    # output = Xtest
    # print(output.shape)
    tsne = embedding(output)
    Xtransformed = tsne.fit_transform(output)
    display(Xtransformed, ytest[:2000])


if __name__ == '__main__':
    main()
    