from foolbox.criteria import Misclassification, TargetedMisclassification
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
import matplotlib.pyplot as plt
import foolbox as fb
from foolbox import TensorFlowModel, accuracy, samples, Model

def plot_imgs(my_model, attack_example, orginal_example, orginal_label, attack_name, frame_idx, output_path):
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.array(attack_example[frame_idx]*255, np.int32).reshape(28, 28))
    plt.title(f"{attack_name} attacked label: {np.argmax(my_model.predict(attack_example[frame_idx:frame_idx+1]))}")
    plt.subplot(122)
    plt.imshow(np.array(orginal_example[frame_idx]*255, np.int32).reshape(28, 28))
    plt.title(f"orginal label: {np.argmax(my_model.predict(orginal_example[frame_idx:frame_idx+1]))}, ground truth: {orginal_label[frame_idx]}")
    plt.savefig(output_path + f"/{attack_name}/frame-{frame_idx}.png")
    plt.close()

def generate_adversarial_example(my_model, X_test, y_test, output_path, attack_name, eps, eps_iter, n_iter, attack_mode=np.inf, save_example=True, verbose=False, target=None):
    if attack_name == "BASIC":
        x_adv = basic_iterative_method(my_model, X_test, eps, eps_iter, n_iter, attack_mode)
        score = my_model.evaluate(x_adv, keras.utils.to_categorical(y_test), batch_size=32)
        print("Loss: %f" % score[0])
        print("Accuracy: %f" % score[1])

    if attack_name == "FGM":
        x_adv = fast_gradient_method(my_model, X_test, eps, attack_mode)
        y_pred     = np.argmax(my_model.predict(X_test), axis=1).reshape(-1, 1)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)

        print(np.sum(y_pred==y_pred_adv)/y_pred.shape[0])
       
    if attack_name == "PGD":
        x_adv = projected_gradient_descent(my_model, X_test, eps, eps_iter, n_iter, attack_mode)
        y_pred     = np.argmax(my_model.predict(X_test), axis=1).reshape(-1, 1)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)

        # test_acc   = tf.metrics.SparseCategoricalAccuracy()
        # print(test_acc(y_pred, y_pred_pgd))
        print(np.sum(y_pred==y_pred_adv)/y_pred.shape[0])
    
    if attack_name == "targetPGD":
        print("My target is ", target)
        x = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y = np.argmax(my_model.predict(x), axis=1)
        y_target = tf.convert_to_tensor(np.ones(y.shape[0])*target, dtype=tf.int32)
        y = tf.convert_to_tensor(np.squeeze(y), dtype=tf.int32)
        fmodel = TensorFlowModel(my_model, bounds=(0, 1))
        attack = fb.attacks.projected_gradient_descent.L2ProjectedGradientDescentAttack(steps=n_iter)
        raw_advs, clipped_advs, success= attack(fmodel, x, TargetedMisclassification(y_target), epsilons=eps)
        x_adv = raw_advs
        # y = keras.utils.to_categorical(y)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)
        print((np.squeeze(y_pred_adv)==target).sum()/y_pred_adv.shape[0])
        # score = my_model.evaluate(x_adv, y, batch_size=32)
        # print("Accuracy: %f" % np.sum(y==y_pred_adv)/y.shape[0])
        # print("Loss: %f" % score[0])
        # print("Accuracy: %f" % score[1])

    if attack_name == "targetPGDLinf":   
        print("My target is ", target)
        x = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y = np.argmax(my_model.predict(x), axis=1)
        y_target = tf.convert_to_tensor(np.ones(y.shape[0])*target, dtype=tf.int32)
        y = tf.convert_to_tensor(np.squeeze(y), dtype=tf.int32)
        fmodel = TensorFlowModel(my_model, bounds=(0, 1))
        attack = fb.attacks.projected_gradient_descent.LinfProjectedGradientDescentAttack(steps=n_iter)
        raw_advs, clipped_advs, success= attack(fmodel, x, TargetedMisclassification(y_target), epsilons=eps)
        x_adv = raw_advs
        # y = keras.utils.to_categorical(y)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)
        print((np.squeeze(y_pred_adv)==target).sum()/y_pred_adv.shape[0])
        # score = my_model.evaluate(x_adv, y, batch_size=32)
        # print("Accuracy: %f" % np.sum(y==y_pred_adv)/y.shape[0])
        # print("Loss: %f" % score[0])
        # print("Accuracy: %f" % score[1])

    if attack_name == "targetPGDL1":
        print("My target is ", target)
        x = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y = np.argmax(my_model.predict(x), axis=1)
        y_target = tf.convert_to_tensor(np.ones(y.shape[0])*target, dtype=tf.int32)
        y = tf.convert_to_tensor(np.squeeze(y), dtype=tf.int32)
        fmodel = TensorFlowModel(my_model, bounds=(0, 1))
        attack = fb.attacks.projected_gradient_descent.L1ProjectedGradientDescentAttack(steps=n_iter)
        raw_advs, clipped_advs, success= attack(fmodel, x, TargetedMisclassification(y_target), epsilons=eps)
        x_adv = raw_advs
        # y = keras.utils.to_categorical(y)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)
        print((np.squeeze(y_pred_adv)==target).sum()/y_pred_adv.shape[0])
        # score = my_model.evaluate(x_adv, y, batch_size=32)
        # print("Accuracy: %f" % np.sum(y==y_pred_adv)/y.shape[0])
        # print("Loss: %f" % score[0])
        # print("Accuracy: %f" % score[1])

    if attack_name == "BBL0":
        x = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y = np.argmax(my_model.predict(x), axis=1)
        y = tf.convert_to_tensor(np.squeeze(y), dtype=tf.int32)
        x = x[100:]
        y = y[100:]
        fmodel = TensorFlowModel(my_model, bounds=(0, 1))
        attack = fb.attacks.brendel_bethge.L0BrendelBethgeAttack()
        raw_advs, clipped_advs, success= attack(fmodel, x, Misclassification(y), epsilons=eps)
        x_adv = raw_advs[0]
        labels = keras.utils.to_categorical(y)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)
        score = my_model.evaluate(x_adv, labels, batch_size=32)
        print("Loss: %f" % score[0])
        print("Accuracy: %f" % score[1])


    if attack_name == "NOISE":
        from skimage.util import random_noise
        x_adv = random_noise(X_test, mode='gaussian', var=0.0)
        y = np.argmax(my_model.predict(X_test), axis=1)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)
        score = my_model.evaluate(x_adv, y, batch_size=32)
        print("Loss: %f" % score[0])
        print("Accuracy: %f" % score[1])

    if attack_name == "CW":
        x = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y = np.argmax(my_model.predict(x), axis=1)
        y = tf.convert_to_tensor(np.squeeze(y), dtype=tf.int32)
        fmodel = TensorFlowModel(my_model, bounds=(0, 1))
        attack = fb.attacks.carlini_wagner.L2CarliniWagnerAttack()
        raw_advs, clipped_advs, success= attack(fmodel, x, Misclassification(y), epsilons=eps)
        x_adv = raw_advs
        y = keras.utils.to_categorical(y)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)
        score = my_model.evaluate(x_adv, y, batch_size=32)
        print("Loss: %f" % score[0])
        print("Accuracy: %f" % score[1])
    
    if attack_name == "targetcw":
        print("My target is ", target)
        x = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y = np.argmax(my_model.predict(x), axis=1)
        y_target = tf.convert_to_tensor(np.ones(y.shape[0])*target, dtype=tf.int32)
        y = tf.convert_to_tensor(np.squeeze(y), dtype=tf.int32)
        fmodel = TensorFlowModel(my_model, bounds=(0, 1))
        attack = fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=n_iter)
        raw_advs, clipped_advs, success= attack(fmodel, x, TargetedMisclassification(y_target), epsilons=eps)
        x_adv = raw_advs
        # y = keras.utils.to_categorical(y)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)
        print((np.squeeze(y_pred_adv)==target).sum()/y_pred_adv.shape[0])
        # score = my_model.evaluate(x_adv, y, batch_size=32)
        # print("Accuracy: %f" % np.sum(y==y_pred_adv)/y.shape[0])
        # print("Loss: %f" % score[0])
        # print("Accuracy: %f" % score[1])


    if attack_name == "DeepFool":
        x = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y = np.argmax(my_model.predict(x), axis=1)
        y = tf.convert_to_tensor(np.squeeze(y), dtype=tf.int32)
        
        fmodel = TensorFlowModel(my_model, bounds=(0, 1))
        attack = fb.attacks.deepfool.LinfDeepFoolAttack()
        raw_advs, clipped_advs, success= attack(fmodel,x, Misclassification(y), epsilons=[0.0001])
        x_adv = raw_advs[0]
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)
        
        score = my_model.evaluate(x_adv, y, batch_size=32)
        print("Loss: %f" % score[0])
        print("Accuracy: %f" % score[1])

    if attack_name == "targetDeepFool":
        print("My target is ", target)
        x = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y = np.argmax(my_model.predict(x), axis=1)
        y_target = tf.convert_to_tensor(np.ones(y.shape[0])*target, dtype=tf.int32)
        y = tf.convert_to_tensor(np.squeeze(y), dtype=tf.int32)
        fmodel = TensorFlowModel(my_model, bounds=(0, 1))

        attack = fb.attacks.GaussianBlurAttack(steps=n_iter, distance=fb.distances.linf)
        raw_advs, clipped_advs, success= attack(fmodel, x, TargetedMisclassification(y_target), epsilons=eps)
        x_adv = raw_advs
        # y = keras.utils.to_categorical(y)
        y_pred_adv = np.argmax(my_model.predict(x_adv), axis=1).reshape(-1, 1)
        print((np.squeeze(y_pred_adv)==target).sum()/y_pred_adv.shape[0])
        # score = my_model.evaluate(x_adv, y, batch_size=32)
        # print("Accuracy: %f" % np.sum(y==y_pred_adv)/y.shape[0])
        # print("Loss: %f" % score[0])
        # print("Accuracy: %f" % score[1])

    if verbose==True:
        if target is not None:
            for frame_idx in range(10):
                plot_imgs(my_model, x_adv, X_test, y_test, attack_name, frame_idx, output_path=output_path + f"/{attack_name}/{target}/")
        else:
            for frame_idx in range(10):
                plot_imgs(my_model, x_adv, X_test, y_test, attack_name, frame_idx, output_path=output_path + f"/{attack_name}/")

    if save_example==True:
        # Path = output_path + f"/{attack_name}/x_{attack_name}_{eps}_{eps_iter}_{str(n_iter)}_{attack_mode}.npy"
        if target is not None:
            Pathx = output_path + f"/{attack_name}/{target}/x_adv.npy"
            np.save(Pathx, x_adv)
            Pathy = output_path + f"/{attack_name}/{target}/y_adv.npy"
            np.save(Pathy, y_pred_adv)
        else:
            Pathx = output_path + f"/{attack_name}/x_adv.npy"
            np.save(Pathx, x_adv)
            Pathy = output_path + f"/{attack_name}/y_adv.npy"
            np.save(Pathy, y_pred_adv)