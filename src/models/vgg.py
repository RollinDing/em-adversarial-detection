import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.python.keras.backend import dropout
import pandas as pd 

def draw_confusion_matrix(array):
    name = ["Top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
    df_cm = pd.DataFrame(array.astype(np.int32), index = [i for i in name],
                  columns = [i for i in name])
    fig = plt.figure(figsize=(14, 12))
    plt.rcParams.update({'font.size': 20})
    ax = sns.heatmap(df_cm, annot=True, fmt="g", cmap='Blues')
    fig.set_edgecolor("black")
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)

    # (orientation="horizontal",fraction=0.046, pad=0.04)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("imgs/confusion-matrix-heatmap.pdf")


class VGG:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):   
        
        self.output_directory = output_directory
        self.weight_decay = 0.05
        
        if build == True:
            self.input_shape = input_shape
            self.nb_classes   = nb_classes
            self.model        = self.build_model()
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

        return

    def build_model(self):
        dropout_rate = 0.4
        weight_decay = self.weight_decay
        input_layer = Input(shape=self.input_shape)
        x = Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(
            weight_decay), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(64, (3, 3),
                kernel_regularizer=regularizers.l2(weight_decay), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(64, (3, 3),
                kernel_regularizer=regularizers.l2(weight_decay), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3),
                kernel_regularizer=regularizers.l2(weight_decay), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(128, (3, 3),
                kernel_regularizer=regularizers.l2(weight_decay), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv2D(128, (3, 3),
                kernel_regularizer=regularizers.l2(weight_decay), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3),
                kernel_regularizer=regularizers.l2(weight_decay), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(256, (3, 3),
                kernel_regularizer=regularizers.l2(weight_decay), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv2D(256, (3, 3),
                kernel_regularizer=regularizers.l2(weight_decay), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2, 2))(x)
        
        x = Flatten()(x)

        x = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

    
        x = Dense(1024, kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

        output_layer = Dense(self.nb_classes, activation='softmax')(x)

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        learning_rate = 0.0001
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=1200,
            decay_rate=0.95,
            staircase=True)

        model.compile(loss='categorical_crossentropy', 
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), 
            metrics=['accuracy'])
        
        callbacks = keras.callbacks.EarlyStopping(monitor='val_loss', 
            min_delta=0, 
            patience=30, 
            verbose=1,
    		mode='auto', 
            baseline=None, 
            restore_best_weights=True)
        
        file_path = self.output_directory+'best_model.hdf5' 
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, 
            monitor='val_loss', 
            save_best_only=True)

        self.callbacks = [callbacks, model_checkpoint]    
        return model
    
    def train(self, x_train, y_train, x_test, y_true, batch_size = 32, nb_epochs=1000):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        
        start_time = time.time() 
        
        if not os.path.isfile(self.output_directory + 'best_model.hdf5'):
            hist = self.model.fit(x_train, y_train, 
                batch_size=batch_size, 
                epochs=nb_epochs,
                verbose=self.verbose, 
                validation_split=0.2, 
                callbacks=self.callbacks)
            duration = time.time() - start_time
            
        elif not os.path.isfile(self.output_directory + 'last_model.hdf5'):
            self.model = keras.models.load_model(self.output_directory + 'best_model.hdf5')
            print("> Restoring the Best Model..")
            hist = self.model.fit(x_train, y_train, 
                batch_size=batch_size, 
                epochs=nb_epochs,
                verbose=self.verbose, 
                validation_split=0.2, 
                callbacks=self.callbacks)
            duration = time.time() - start_time
        else:
            pass

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_score = model.predict(x_test)

        # convert the predicted from binary to integer 
        y_pred = np.argmax(y_score, axis=1)
        if self.verbose:
            print('Confusion Matrix')
            print(confusion_matrix(y_true, y_pred))
            fig = plt.figure(figsize=(12, 10))
            plt.tick_params(
                axis='both',        
                which='both', 
                left=False,   
                labelleft=False, 
                bottom=False,      
                top=False,         
                labelbottom=False) 
            plt.tight_layout()
            im = plt.imshow(confusion_matrix(y_true, y_pred), cmap='Blues')
            fig.set_edgecolor("black")
            plt.colorbar(im, orientation="horizontal",fraction=0.046, pad=0.04)
            plt.savefig("confusion-matrix-heatmap.pdf")
            plt.close()
            print('Classification Report')
            print(classification_report(y_true, y_pred))
            print("Top 2 accuracy", top_k_accuracy_score(y_true, y_score, k=2))

            draw_confusion_matrix(confusion_matrix(y_true, y_pred))
        keras.backend.clear_session()

    def predict(self, x_test, y_true):
        y_pred = self.model.predict(x_test)
        
        # convert the predicted from binary to integer 
        y_pred = np.argmax(y_pred, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(y_true, y_pred))
        plt.figure()
        sns.heatmap(confusion_matrix(y_true, y_pred))
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("test.png")
        plt.close()
        print('Classification Report')
        print(classification_report(y_true, y_pred))
        print("Top 2 accuracy", top_k_accuracy_score(y_true, y_pred, k=2))