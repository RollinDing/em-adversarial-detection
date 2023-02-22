"""
Variational Auto-Encoder
https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py
"""
from functools import total_ordering
import sys
sys.path.append("./src/")
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from utils.loader import Loader
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Build the encoder
"""
def build_1D_encoder(input_dim, latent_dim=2):
    encoder_inputs = keras.Input(shape=(input_dim))
    
    x = layers.Dense(256, activation='relu')(encoder_inputs)

    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder

def build_2D_encoder(input_dim, latent_dim=2):
    encoder_inputs = keras.Input(shape=input_dim)
    x = layers.Conv2D(32, 3, activation="relu", strides=2,
                    padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder

"""
## Build the decoder
"""
def build_1D_decoder(input_dim, latent_dim=2):
    latent_inputs = keras.Input(shape=(latent_dim))
    x = layers.Dense(16, activation="relu")(latent_inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(input_dim, activation='relu')(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

def build_2D_decoder(input_dim, latent_dim=2):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(input_dim[0] * input_dim[1] * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((input_dim[0], input_dim[1], 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu",
                            strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu",
                            strides=1, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=0
            )
            
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def get_losses(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = keras.losses.binary_crossentropy(data, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var -
                            tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        total_loss = reconstruction_loss + kl_loss
        return total_loss


def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size_1 = 22
    digit_size_2 = 283
    scale = 1.0
    figure = np.zeros((digit_size_1 * n, digit_size_2 * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size_1, digit_size_2)
            figure[
                i * digit_size_1: (i + 1) * digit_size_1,
                j * digit_size_2: (j + 1) * digit_size_2,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size_1 // 2
    end_range = n * digit_size_2 + start_range
    pixel_range = np.arange(start_range, end_range, digit_size_1)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


def plot_label_clusters(fig, ax, vae, data, labels, color='black'):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    X_embedded = PCA(n_components=2).fit_transform(z_mean)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color, s=2, alpha=0.7)
    
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")

def plot_label_clusters_3D(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    X_embedded = PCA(n_components=3).fit_transform(z_mean)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=labels)
    plt.show()



if __name__ == "__main__":
    TRACE_PATH = "../data/data/fm_lenet-20211102"
    ATTACK = "org"
    BATCH = 2
    RATE = 1
    trace_num = 10_000
    if BATCH > 3:
        trace_len = 33_000
    else:
        trace_len = 36_000

    label_path = f"data/{ATTACK}/y_adv.npy"
    trace_path = f"{TRACE_PATH}/{ATTACK}/{str(BATCH)}"
    output_path = f"meta/raw/{ATTACK}/{str(BATCH)}"
    myloader = Loader(trace_path, label_path, trace_num,
                      trace_len, RATE, output_path)
    myloader.stft(nperseg=256)

    Zxxs = myloader.Zxxs
    Zxxs = Zxxs[:, :22, :]
    labels = myloader.label
    v_min = Zxxs.min(axis=(0, 1), keepdims=True)
    v_max = Zxxs.max(axis=(0, 1), keepdims=True)
    Zxxs = (Zxxs - v_min)/(v_max - v_min)
    Zxxs = np.expand_dims(Zxxs, axis=3)
    X_train, X_test, y_train, y_test = train_test_split(
        Zxxs, labels, test_size=0.2, random_state=42)

    
    vae = VAE(build_2D_encoder, build_2D_decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(Zxxs, epochs=30, batch_size=128)

    
    plot_label_clusters(vae, X_train, y_train)
    plot_latent_space(vae)
