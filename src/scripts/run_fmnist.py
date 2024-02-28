"""
Train a LeNet-5 model on FMNIST,
Attack adversarial models with adversarial attack methods,
Train robust model with adversarial training
"""

import sys
from xmlrpc.client import boolean
sys.path.append("./src/")
from models.classify import classifyCNN
from download import preprocessor
from utils.save import *
from attack.attack import generate_adversarial_example
from tensorflow.keras import models
import matplotlib.pyplot as plt
import argparse
import time

# output_path = "data"
# attack_name = "targetPGD"
# eps = 128/255
# eps_iter = 1/255
# n_iter   = 1000
# verbose = True

def parse_args():
    # hyper-parameters for fmnist model training and adversarial training 
    parser = argparse.ArgumentParser(
        description='Training on fmnist network')
    parser.add_argument('cmd', choices=['train', 'attack', 'advtraining'],
        help='train: training the original LeNet on FMNist; attack: attack the trained model with given adversarial attack methods;advtraining:do adversarial training')
    parser.add_argument('--model-name', default='cnn', choices=['cnn', 'vgg'],
        help='the structure of victim model')
    parser.add_argument('--attack-name', choices=['BASIC', 'FGM', 'PGD', 'targetPGD', 'noise', 'CW', 'targetcw', 'DeepFool'], type=str,
        help='choose the adversarial attack method')
    parser.add_argument('--target', default=0, type=int, 
        help='the target victim class to be analyze; choose from 0-9;')
    parser.add_argument('--output-path', default="data", type=str,
        help='The output path of adversarial samples')
    parser.add_argument('--eps', default=128/255, type=float,
        help='epsilon value of adversarial attack(if required)')
    parser.add_argument('--eps-iter', default=1/255, type=float,
        help='epsilon step size if required')
    parser.add_argument('--n-iter', default=1000, type=int,
        help='number of iteration if required')
    parser.add_argument('--verbose', default=False, type=bool,
        help='verbose of attack process')
    parser.add_argument('--save-example', default=False, type=bool,
        help='save adversarial sample or not')
    args = parser.parse_args()
    return args

def load_pretrained(args):
    """
    Load the training data and model
    """
    X_train, y_train, X_test, y_test = preprocessor.do()
    model = classifyCNN(X_train, y_train, X_test, y_test, trainable=True, output_directory=f"saved_models/{args.model_name}/")
    model.summary()
    return  X_train, y_train, X_test, y_test, model

def eval_pretrained(model, X_train, X_test):
    start_time = time.time()
    model.predict(np.expand_dims(np.concatenate([X_train, X_test], axis=0), axis=3))
    end_time = time.time()
    print("Evaluation time is:", end_time-start_time)

def do_attack(model, X_test, y_test, args):
    X_test = np.expand_dims(X_test, axis=3)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Testing accuracy", (y_pred==y_test).sum()/10000)
    generate_adversarial_example(model, X_test, y_test, output_path=args.output_path, 
                                    attack_name=args.attack_name, 
                                    eps=args.eps, 
                                    eps_iter=args.eps_iter, 
                                    n_iter=args.n_iter,
                                    save_example=args.save_example,
                                    verbose=args.verbose,
                                    attack_mode=2,
                                    target=args.target)
    target = args.target
    # X_adv = np.load(output_path+ f"/{attack_name}/x_adv.npy")
    X_adv = np.load("/home/ruyi/em-adversarial-examples-detection/data/data/fm/CW1/x_adv.npy")
    X_adv = X_adv[y_pred!=target]

    y_adv = np.argmax(model.predict(X_adv), axis=1)
    print(y_adv)
    print("Attack successful rate accuracy", (y_adv==target).sum()/X_adv.shape[0])

    X_adv = X_adv[y_adv==target]
    y_adv = y_adv[y_adv==target]
    print(X_adv.shape, y_adv.shape)
    # save_array(X_adv, f"data/{attack_name}/x_adv_{target}.npy")
    # save_array(y_adv, f"data/{attack_name}/y_adv_{target}.npy")

def main():
    args = parse_args()
    X_train, y_train, X_test, y_test, model = load_pretrained(args)
    if args.cmd == 'train':
        eval_pretrained(model, X_train, X_test)
    if args.cmd == 'attack':
        do_attack(model, X_test, y_test, args)

if __name__ == "__main__":
    main()
        


exit(0)
n=4

y_pred = np.argmax(model.predict(X_test), axis=1)

# save_weights(model, f"saved_models/{name}/weights.h5")

# save_array(X_train, "data/org/X_train.npy")
# save_array(y_train, "data/org/y_train.npy")
# save_array(X_test, "data/org/X_test.npy")
# save_array(y_test, "data/org/y_test.npy")


# plot output for each layers
layer_outputs = [layer.output for layer in model.layers[:4]] 
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 
activations = activation_model.predict(X_test)
activations = [np.sum(activation, axis=0, keepdims=True) for activation in activations]

layer_names = []
for layer in model.layers[:4]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 3

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]

            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig(f"results/{layer_name}_{n}.png")
    plt.close()


# save_array(y_pred, "data/org/y_adv.npy")


# Generate Adversarial examples 
