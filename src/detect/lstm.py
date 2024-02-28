from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
sys.path.append("./src/")
import numpy as np
from utils.loader import Loader
from tensorflow import keras

# hyperparameters 
BATCH_SIZE = 64
EPOCHS = 100
MAX_SEQ_LENGTH = 4
NUM_FEATURES = 2048
CLASS_NUM = 10

# feature extractor 
def feature_extract(Zxxs, batch):
    from embedding import build_vae
    vae = build_vae(Zxxs, batch)
    return vae.encoder.predict(Zxxs)

# load data/ frame
# Here change batch to focus on different frame 
features = []
for batch in [0, 1, 2, 3]:
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
    output_path = f"meta/raw/org/{batch}"
    myloader = Loader(trace_path, label_path, trace_num, trace_len, RATE, output_path)
    myloader.stft(nperseg=256)
    Zxxs = myloader.Zxxs                  # Zxxs: input spectrogram [sample_num, image_height, image_length]
    Zxxs = Zxxs[:, :15, :]
    labels = myloader.label               # labels: corresponding labels [sample_num, ]
    Zxxs = np.expand_dims(Zxxs, axis=3)
    IMG_HEIGHT = Zxxs.shape[1]
    IMG_LENGTH = Zxxs.shape[2]
    print("finish loading!")
    feature = feature_extract(Zxxs, batch)[2]
    feature = np.array(feature)
    feature = np.expand_dims(feature, axis=1)
    Zxxs = None
    print(feature.shape)
    features.append(feature)

# use the feature extractor to predict and 
data =np.concatenate(features, axis=1)
labels = labels
print(data.shape)
print(labels.shape)
X_train, X_test, y_train, y_test = train_test_split(np.expand_dims(data, axis=3), labels, test_size=0.2, random_state=42)
# The sequential model
# Utility for our sequence model.
def get_sequence_model():
    # frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES, 1))

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    # x = keras.layers.GRU(10, return_sequences=True)(
    #     frame_features_input)
    # x = keras.layers.GRU(8)(x)
    # x = keras.layers.InputLayer(input_shape=(MAX_SEQ_LENGTH, NUM_FEATURES, 1), name='input_data')(frame_features_input)
    # x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(x),
    # x = keras.layers.Dropout(0.4)(x)
    # x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(20, activation="relu")(x)
    # output = keras.layers.Dense(CLASS_NUM, activation="softmax")(x)
    # rnn_model = keras.Model(frame_features_input, output)


    rnn_model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(MAX_SEQ_LENGTH, NUM_FEATURES, 1), name='input_data'),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=100, kernel_initializer='he_uniform', activation='relu'),
        keras.layers.Dense(units=10, activation='softmax', name='output_logits')
    ])
    rnn_model.summary()
    
    learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), 
        metrics=["accuracy"]
    )
    return rnn_model

def run_experiment():
    filepath = "/tmp/rnn-classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()