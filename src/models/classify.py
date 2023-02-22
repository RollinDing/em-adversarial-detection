"""
Code for classfication
"""
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from models.vgg import VGG as vgg
from sklearn.preprocessing import OneHotEncoder

def classifyVGG(X_train, y_train, X_test, y_test, trainable=True, save=False, output_directory=f"results/"):
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    input_shape = X_train.shape[1:]
    y_true = y_test
    print(y_test)
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    enc = OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()


    classifier = vgg(
            output_directory, input_shape, nb_classes, verbose=True)
    classifier.train(X_train, y_train, X_test, y_true)    
    # classifier.predict(X_test, y_true)
    if not os.path.exists(output_directory+"best_model.hdf5"):
        print("> Training a new model")
        classifier = vgg(
                output_directory, input_shape, nb_classes, verbose=True)
        classifier.train(X_train, y_train, X_test, y_true)
        if save:
            np.save(f"{output_directory}x_test.npy", X_test)
            np.save(f"{output_directory}y_test.npy", y_test)
            np.save(f"{output_directory}y_true.npy", y_true)
        model = classifier.model
    else:
        print("> Restoring a pretrained model")
        path = output_directory+"best_model.hdf5"
        model = tf.keras.models.load_model(path)
    return model