import numpy as np
import tensorflow as tf
import matplotlib as plt
# from tensorflow_model_optimization.python.core.keras import metrics
from Preprocessor import Preprocessor
from models.classify import classifyCNN, classifyTinyCNN
from attack.attack import generate_adversarial_example

# attack configure
name = "cnn"
output_path = "data"
attack_name = "targetPGD"
eps = 128/255
eps_iter = 1/255
n_iter   = 1000
verbose = True

# data configure
