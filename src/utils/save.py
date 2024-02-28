import numpy as np
import tensorflow as tf

def save_array(data, path):
    np.save(path, data)

def save_weights(model, path):
    model.save_weights(path)
