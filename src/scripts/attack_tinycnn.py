import sys
sys.path.append("./src/")
from matplotlib.pyplot import quiverkey

from download import preprocessor
from models.classify import classifyTinyCNN
from sklearn.metrics import confusion_matrix
import numpy as np 
import tensorflow as tf
from utils.save import *
from sklearn.preprocessing import OneHotEncoder
from attack.attack import generate_adversarial_example

name = "tinycnn-2layers"
X_train, y_train, X_test, y_test = preprocessor.tiny_sample()
model = classifyTinyCNN(X_train, y_train, X_test, y_test, output_directory=f"saved_models/{name}/")

output_path = "data/tinycnn"
attack_name = "CW"
eps = 128/255
eps_iter = 1/255
n_iter   = 1000
verbose = True

generate_adversarial_example(model, X_test, y_test, output_path, attack_name, eps, eps_iter, n_iter, attack_mode=2)