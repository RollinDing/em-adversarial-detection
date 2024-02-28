from collections import Counter
import sys

from tensorflow.python.ops.gen_math_ops import Log
sys.path.append("./src/")
import numpy as np 
import tensorflow as tf
import tensorflow.keras as keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from download import preprocessor
from models.classify import classifyCNN, classifyTinyCNN

import matplotlib.pyplot as plt

class ActivationExtracter:
    def __init__(self, model, X_train, y_train, X_test, y_test) -> None:
        self.input_train = X_train
        self.output_train = y_train
        self.input_test = X_train
        self.output_test = y_train

        self.nb_samples = self.input_test.shape[0]
        self.nb_classes = len(np.unique(self.output_test, axis=0))
        self.model = model
        self.features = []
        pass

    def show_layer_name(self):
        print([layer.name for layer in self.model.layers])

    def get_layer_output(self, layer_name):
        """
        x: test sample tensor
        """
        output_layer = [layer for layer in self.model.layers if layer.name==layer_name][0]
        feature = tf.keras.models.Model(inputs=self.model.inputs, outputs=output_layer.output)
        self.features.append(feature)
        return feature(self.input_test)

    def split_train_test(self, X, y, random=True):
        """
        Split the training and test data set ahead
        """
        if random:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
        else:
            n_sample = X.shape[0]
            X_train, X_test = X[:int(n_sample*0.8), :, :], X[int(n_sample*0.8):, :, :]
            y_train, y_test = y[:int(n_sample*0.8),], y[int(n_sample*0.8):,]
        return X_train, X_test, y_train, y_test

    def build_derived_model(self, X_train, X_test, y_train, y_test):
        """
        activations: The input of the derived model, which is the output of target layers 
        """
        derived_model = tf.keras.Sequential()
        derived_model.add(tf.keras.layers.Dense(self.nb_classes, activation="softmax")) # output layer

        learning_rate = 0.001
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True)

        derived_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

        callbacks = keras.callbacks.EarlyStopping(monitor='val_loss', 
            min_delta=0, 
            patience=10, 
            verbose=1,
    		mode='auto', 
            baseline=None, 
            restore_best_weights=True)

        derived_model.fit(X_train, y_train, 
            validation_split=0.2, 
            batch_size=32, 
            epochs=1000, 
            callbacks=callbacks)

        y_pred = derived_model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print('Classification Report')
        print(classification_report(y_true, y_pred))
        keras.backend.clear_session()
        return derived_model

    def do_split_train_test(self, layer_output):
        enc = OneHotEncoder(categories='auto')
        enc.fit(self.output_test.reshape(-1, 1))
        y = enc.transform(self.output_test.reshape(-1, 1)).toarray()
        
        X_train, X_test, y_train, y_test = self.split_train_test(layer_output, y)
        return X_train, X_test, y_train, y_test

    def to_activation_pattern(self, layer_output, rate=1):
        """
        Transform activation values to activation patterns
        """
        activation_pattern = (layer_output>0).astype(np.int8)
        # activation_pattern = layer_output.astype(np.float8)
        activation_pattern = activation_pattern.reshape(self.nb_samples, -1, rate).mean(axis=2)
        print(activation_pattern.shape)
        return activation_pattern

    def do_activation_analysis(self, layer_names, derived=True):
        """
        Conduct the anaylsis of output given layername
        """
        self.derived_models = []
        self.layer_outputs = []
        self.X_tests = []
        self.y_tests =[]
        for layer_name in layer_names:
            layer_output = self.get_layer_output(layer_name).numpy()
            layer_output = layer_output.reshape(self.nb_samples, -1)
            # layer_output = self.to_activation_pattern(layer_output)
            if derived == True:
                X_train, X_test, y_train, y_test = self.do_split_train_test(layer_output)
                derived_model = self.build_derived_model(X_train, X_test, y_train, y_test)
                self.X_tests.append(X_test)
                self.y_tests.append(y_test)
                self.derived_models.append(derived_model)
            else:
                self.layer_outputs.append(layer_output)
    def get_logits(self, model, X, labels, classIdx):
        """
        Get the model logits for given class Idx 
        """
        logits = model.predict(X)
        keras.backend.clear_session()

        labels = np.argmax(labels, axis=1)
        mask = np.squeeze(labels==classIdx)
        return logits[mask, :]

    def do_get_benign_logits(self, classIdx):
        """
        """
        logits_lst = []
        for derived_model, X_test, y_test in zip(self.derived_models, self.X_tests, self.y_tests):
            logits = self.get_logits(derived_model, X_test, y_test, classIdx)
            logits_lst.append(logits)
        return logits_lst

    def one_class_SVM(self, classIdx, logits_benign_lst=None, nu=0.05, kernel="rbf", gamma='scale'):
        """
        conduct one class SVM to distinguish outliers(adversarial examples)
        """
        self.classIdx = classIdx
        if logits_benign_lst is None:
            logits_benign_lst = self.do_get_benign_logits(classIdx)
        else:
            logits_benign_lst = [logits_benign[y_test==classIdx] for logits_benign in logits_benign_lst]
        svm_lst = []
        y_pred_trains = []
        y_pred_tests = []
        y_test_scores = []
        logits_benign_lst = [np.concatenate(logits_benign_lst, axis=1)]
        for logits_benign in logits_benign_lst:
            X_train = logits_benign[:int(logits_benign.shape[0]*0.9), :]
            X_test = logits_benign[int(logits_benign.shape[0]*0.9):, :]
            clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
            clf.fit(X_train)
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

            n_error_train = y_pred_train[y_pred_train == -1].size
            n_error_test = y_pred_test[y_pred_test == -1].size
            y_pred_trains.append(y_pred_train)
            y_pred_tests.append(y_pred_test)
            y_test_scores.append(clf.score_samples(X_test))
            svm_lst.append(clf)
            print("> Train accuracy ", self.compute_logits_accuracy(y_pred_trains))
            print("> Test accuracy ", self.compute_logits_accuracy(y_pred_tests))
        return svm_lst, np.asarray(y_test_scores).mean(axis=0)
    
    def compute_score_accuracy(self, y_score, threshold):
        """
        Compute the model accuracy based on score
        """
        print("> Detection Rate based on score is ", np.sum(y_score>threshold)/y_score.shape[0])
        return np.sum(y_score>threshold)/y_score.shape[0]

    def compute_logits_accuracy(self, y_pred_lst):
        """
        Given a y_pred_lst, compute the error rate
        """
        results = np.ones(y_pred_lst[0].shape)
        prediction = np.ones(y_pred_lst[0].shape)
        for y_pred in y_pred_lst:
            prediction = np.logical_and((y_pred==1), prediction)
        print("The value is ", results[prediction].size, results.shape[0])
        return results[prediction].size/results.shape[0]
    
    def load_adversarial_examples(self, name, filepath, labelpath):
        """
        Load adversarial examples
        """
        # filepath = f"data/em-adversarial-data/x_{name}.npy"
        # labelpath = f"data/em-adversarial-data/y_{name}.npy"
        self.X_adv = np.load(filepath)
        self.y_adv = np.load(labelpath)
        self.adv_layer_outputs = [feature(self.X_adv).numpy() for feature in self.features]
        nb_samples = self.X_adv.shape[0]
        self.adv_layer_outputs = [adv_layer_output.reshape(nb_samples, -1) for adv_layer_output in self.adv_layer_outputs] 
        # self.adv_layer_outputs = [self.to_activation_pattern(adv_layer_output) for adv_layer_output in self.adv_layer_outputs]
    
    def do_get_adv_logits(self):
        """
        Get the adversarial logits
        """
        adv_logits = []
        for derived_model, adv_layer_output in zip(self.derived_models, self.adv_layer_outputs):
            enc = OneHotEncoder(categories='auto')
            enc.fit(self.y_adv.reshape(-1, 1))
            y = enc.transform(self.y_adv.reshape(-1, 1)).toarray()
            adv_logit = self.get_logits(derived_model, adv_layer_output, y, self.classIdx)
            adv_logits.append(adv_logit)
        return adv_logits

    def one_class_SVM_evaluate(self, svm_lst, logits_adv_lst=None):
        """
        Evaluate the performance of one class SVM on adversarial examples
        """
        if logits_adv_lst is None:
            logits_adv_lst = self.do_get_adv_logits()
        else:
            logits_adv_lst = [logits_adv[np.squeeze(self.y_adv) == self.classIdx] for logits_adv in logits_adv_lst]
        y_preds = []
        y_scores = []
        logits_adv_lst = [np.concatenate(logits_adv_lst, axis=1)]
        for logits_adv, clf in zip(logits_adv_lst, svm_lst):
            y_pred = clf.predict(logits_adv)
            
            y_dist = clf.decision_function(logits_adv)
            n_error = y_pred[y_pred==-1].size
            y_preds.append(y_pred)
            print("> One fail detection rate ", 1-self.compute_logits_accuracy(y_preds))
        return np.asarray(y_scores).mean(axis=0)


if __name__ == "__main__":
    name = "cnn"
    X_train, y_train, X_test, y_test = preprocessor.do()
    X_train = np.expand_dims(X_train, axis=3)
    X_test  = np.expand_dims(X_test, axis=3)
    model = classifyCNN(X_train, y_train, X_test, y_test, output_directory=f"saved_models/{name}/")
    myAE = ActivationExtracter(model, X_train, np.argmax(model.predict(X_train), axis=1), X_test, y_test)
    
    myAE.show_layer_name()
    # layer_names = ['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'flatten', 'dense', 'dense_1', 'output_logits']
    layer_names = ['max_pooling2d']
    myAE.do_activation_analysis(layer_names)
    
    filepath = "data/cw/x_adv.npy"
    labelpath = "data/cw/y_adv.npy"
    
    myAE.load_adversarial_examples("cw", filepath, labelpath)
    # svms, _ = myAE.one_class_SVM(3, myAE.layer_outputs)
    for classIdx in range(1):
        print("> CLASS: ", classIdx)
        svms, _ = myAE.one_class_SVM(classIdx)

        # myAE.one_class_SVM_evaluate(svms, myAE.adv_layer_outputs)
        myAE.one_class_SVM_evaluate(svms)
        # after get the activation logits, we can build svms based on it