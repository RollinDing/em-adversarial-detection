from tokenize import PlainToken
import sys
sys.path.append("./src/")
import numpy as np

from utils.GradCAM import GradCAM
from models.classify import classifyCNN
from scripts.configure import *
from download import preprocessor
from utils.save import *

X_train, y_train, X_test, y_test = preprocessor.do()
target = 1
X_adv, y_adv =np.load(f"../data/data/fm/targetcw/{target}/x_adv.npy"),  np.load(f"../data/data/fm/targetcw/{target}/y_adv.npy")
model = classifyCNN(X_train, y_train, X_test, y_test, trainable=True, output_directory=f"saved_models/{name}/temp/")
model.summary()

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('seaborn-paper')
# X = np.mean(np.expand_dims(X_test, axis=3)[y_test==n], keepdims=True, axis=0)\
a = 8
Xnorm = np.expand_dims(X_test[a:a+1], axis=3)
ynorm = y_test[a:a+1]
Xadv  = X_adv[a:a+1]
yadv  = y_adv[a:a+1]

print(ynorm, yadv)

layer = "conv2d_1"
cam = GradCAM(model, ynorm, layerName=layer)
heatmap = cam.compute_heatmap(Xnorm)

layer2 = "conv2d"
cam2 = GradCAM(model, ynorm, layerName=layer2)
heatmap2 = cam.compute_heatmap(Xnorm)

fig = plt.figure(figsize=(8, 8))
plt.tick_params(
    axis='both',        
    which='both', 
    left=False,   
    labelleft=False, 
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.tight_layout()
fig.set_edgecolor("black")
plt.imshow(heatmap2*1, aspect='auto', cmap='viridis')
plt.savefig(f"imgs/GradCAM_norm_{ynorm}_{layer}_{a}.pdf")
plt.close()

fig = plt.figure(figsize=(8, 8))
plt.tick_params(
    axis='both',        
    which='both', 
    left=False,   
    labelleft=False, 
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.tight_layout()
fig.set_edgecolor("black")
plt.imshow(Xnorm[0], aspect='auto', cmap='gist_gray')
plt.savefig(f"imgs/Orginal_norm.pdf")
plt.close()

cam = GradCAM(model, ynorm, layerName=layer)
heatmap = cam.compute_heatmap(Xadv)
plt.figure(figsize=(8, 8))
plt.tick_params(
    axis='both',        
    which='both', 
    left=False,   
    labelleft=False, 
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.tight_layout()
plt.imshow(heatmap, aspect='auto', cmap='viridis')
plt.savefig(f"imgs/GradCAM_adv_{ynorm}_{layer}_{a}.pdf")
plt.close()

fig = plt.figure(figsize=(8, 8))
plt.tick_params(
    axis='both',        
    which='both', 
    left=False,   
    labelleft=False, 
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.tight_layout()
fig.set_edgecolor("black")
plt.imshow(Xadv[0], aspect='auto', cmap='gist_gray')
plt.savefig(f"imgs/Orginal_adv.pdf")
plt.close()