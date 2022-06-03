# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 22:24:49 2022

@author: kikig
"""
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib import pyplot as plt

class Audio_VGD():
    def __init__(self, test_size=0.10):
        self.test_size=test_size
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_path = dir_path + "/utils/VGD/data/males_females_audio.json"
        self.save_path = dir_path + "/utils/VGD/temp/dummy.png"
        self.data_shape = (28, 28)

    def get_datasets(self):

        data = open(self.data_path)
        data = json.load(data)

        labels = []
        dataset = []


        for audio in data['males']:
            dataset += [self.convert_to_spectrogram(audio)]
            labels += [np.int64(0)]

        for audio in data['females']:
            dataset += [self.convert_to_spectrogram(audio)]
            labels += [np.int64(1)]

        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=self.test_size, random_state=42)

        return (X_train, np.int64(y_train)), (X_test, np.int64(y_test))

    def get_audio_data(self):
        data = open(self.data_path)
        data = json.load(data)

        labels = []
        dataset = []


        for audio in data['males']:
            dataset += [audio]
            labels += [np.int64(0)]

        for audio in data['females']:
            dataset += [audio]
            labels += [np.int64(1)]

        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=self.test_size, random_state=42)

        return (X_train, np.int64(y_train)), (X_test, np.int64(y_test))


    def convert_to_spectrogram(self, audio):
        self.save_image(audio, self.data_shape, self.save_path)

        return self.load_image(self.save_path)




    def save_image(self, data, out_shape, save_path):



        noverlap=16
        cmap='gray_r'

        fig = plt.figure()
        fig.set_size_inches((out_shape[0]/fig.get_dpi(), out_shape[1]/fig.get_dpi()))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.specgram(data, cmap=cmap, Fs=2, noverlap=noverlap, NFFT=128)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def load_image(self, dummy_file_path):
        return matplotlib.pyplot.imread(dummy_file_path)[:,:,0]


if __name__ == '__main__':
    _, (x_test, y_test) = Audio_VGD().get_datasets()
    print(x_test)
    _, (x_test, y_test) = Audio_VGD().get_audio_data()
    print(x_test)