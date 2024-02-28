"""
In this script, we are trying to embed a 2-D image into a 1-D vector with VAE
"""
import numpy as np
from pip import main
import tensorflow as tf
import pandas as pd
import seaborn as sns
import sys
import os
sys.path.append("./src/")
from sklearn.model_selection import train_test_split
from tensorflow import keras
from models.vae import VAE, build_2D_encoder, build_2D_decoder
from utils.loader import Loader
from sklearn import preprocessing

feature_num = 2048

def build_vae(Zxxs, batch):
    output_directory = f"saved_models/vae-embedding/{batch}/"

    EarlyStopping = keras.callbacks.EarlyStopping(monitor='loss', 
            min_delta=0, 
            patience=10, 
            verbose=1,
            mode='auto', 
            baseline=None, 
            restore_best_weights=True)

    file_path = output_directory+'last_model.weights'

    learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True)

    latent_dim = feature_num
    # vae = VAE(build_2D_encoder(Xorg.shape[1:], latent_dim=latent_dim), build_2D_decoder(Xorg.shape[1:], latent_dim=latent_dim))
    vae = VAE(build_2D_encoder(Zxxs.shape[1:], latent_dim=latent_dim), 
        build_2D_decoder(Zxxs.shape[1:], latent_dim=latent_dim))

    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

    if not os.path.exists(output_directory+"checkpoint"):    
        print("> Fitting the VAE model------------")
        vae.fit(Zxxs, epochs=100, batch_size=32, callbacks=[EarlyStopping])
        vae.save_weights(file_path)
    else:
        path = output_directory
        vae.load_weights(file_path)
    return vae

if __name__ == "__main__":
    batch=3
    trace_num = 10_000
    RATE=1
    if batch <= 3:
        trace_len = 36_000
    elif batch <= 5 :
        trace_len = 33_000
    elif batch <= 17:
        trace_len = 10_000
    else:
        trace_len = 6_000
    TRACE_PATH = "../data/data/fm_lenet-20211130"
    label_path = f"../data/data/fm/org/y_adv.npy"
    trace_path = f"{TRACE_PATH}/{batch}"
    output_path = f"meta/raw/train/{batch}"
    myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
    myloader.stft(nperseg=256)
    Zxxs = myloader.Zxxs                  # Zxxs: input spectrogram [sample_num, image_height, image_length]
    Zxxs = Zxxs[:trace_num, :15, :]
    min, max = Zxxs.min(), Zxxs.max()
    Zxxs_minmax = (Zxxs - min) / (max-min)
    Zxxs_minmax = np.expand_dims(Zxxs_minmax, axis=3)
    build_vae(Zxxs_minmax, batch)