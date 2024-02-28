import numpy as np

class Normalizer():
    def __init__(self, data):
        self.mean = data.mean(axis=0)
        self.std  = data.std(axis=0)
    
    def do(self, data):
        return (data - self.mean)/self.std

    def do2D(self, data):
        """
        Normalize 2D image dataset
        checking the post: https://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn
        """
        n_sample, n_channel, length = data.shape[0], data.shape[1], data.shape[2]
        data = data.reshape(-1, int(n_channel*length))
        data = (data - data.mean(axis=0))/data.std(axis=0)
        data = data.reshape(n_sample, n_channel, length)
        return data