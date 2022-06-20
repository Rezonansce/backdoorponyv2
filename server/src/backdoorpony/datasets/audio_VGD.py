# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 22:24:49 2022

@author: kikig
"""
from tqdm import tqdm
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib import pyplot as plt
import soundfile as sf

class Audio_VGD():
    def __init__(self, test_size=0.10):
        """
        This class generates the Voice Gender Dataset

        Parameters
        ----------
        test_size : float, optional
            Test-train split size. The default is 0.10.

        dir_path: string
            Path to this file
        save_path: string
            Path to where the spectrogrammers are saved
        data_shape: tuple (n, n)
            Resolution of the generated images
        Returns
        -------
        None.

        """
        self.test_size=test_size
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.save_path = os.path.join(self.dir_path, "utils/VGD/spectrogrammer/")
        self.data_shape = (28, 28)

    def get_datasets(self, train_size=0.10, test_size = 0.10):
        """
        Loads the image version of the dataset

        Returns
        -------
        (array, array), (array, array)
            Returns the train test split of the loaded data.

        """



        data_path = os.path.join(self.dir_path, "utils/VGD/spectrogrammer/")


        males = glob.glob(os.path.join(data_path, "m/*.png"))
        females = glob.glob(os.path.join(data_path, "f/*.png"))
        dataset = [None] * (len(males)+len(females))
        labels = np.zeros(len(males)+len(females))

        for i, file in enumerate(tqdm(males)):
            dataset[i] = self.load_image(file)


        offset = len(males)

        for i, file in enumerate(tqdm(females)):
            dataset[i+offset] = self.load_image(file)
            labels[i+offset] = 1


        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, train_size=train_size, test_size=test_size, random_state=42)

        return (X_train, np.int64(y_train)), (X_test, np.int64(y_test))

    def get_audio_data(self, train_size=0.10, test_size = 0.10):
        """
        Loads the audio version of the dataset. Used for poisoning.

        Returns
        -------
        (array, array), (array, array)
            Returns the train test split of the loaded data.

        """



        data_path = os.path.join(self.dir_path, "utils/VGD/data/")

        males = glob.glob(os.path.join(data_path, "m/*.ogg"))
        females = glob.glob(os.path.join(data_path, "f/*.ogg"))

        dataset = [None] * (len(males)+len(females))
        labels = np.zeros(len(males)+len(females))

        for i, file in enumerate(tqdm(males)):
            data, sr = sf.read(file)
            dataset[i] = data

        offset = len(males)

        for i, file in enumerate(tqdm(females)):
            data, sr = sf.read(file)
            dataset[i+offset] = data
            labels[i+offset] = 1

        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, train_size, test_size=test_size, random_state=42)


        return (X_train, np.int64(y_train)), (X_test, np.int64(y_test))



    def create(self):
        """
        This function was used to create the spectrogrammer files.

        Returns
        -------
        None.

        """
        data_path = os.path.join(self.dir_path, "utils/VGD/data/")

        males = glob.glob(os.path.join(data_path, "m/*.ogg"))
        females = glob.glob(os.path.join(data_path, "f/*.ogg"))

        for i, file in enumerate(tqdm(males)):
            data, sr = sf.read(file)
            self.save_image(data, self.data_shape, os.path.join(self.save_path, "m/" + str(i) + ".png"))


        for i, file in enumerate(tqdm(females)):
            data, sr = sf.read(file)
            self.save_image(data, self.data_shape, os.path.join(self.save_path, "f/" + str(i) + ".png"))


    def save_image(self, data, out_shape, save_path):
        """
        Saves data in .png format

        Parameters
        ----------
        data : array
            audio data to be saved.
        out_shape : tuple (n, n)
            resolution of the saved image.
        save_path : string
            save path of the image.

        Returns
        -------
        None.

        """


        noverlap=16
        cmap='gray_r'

        fig = plt.figure()
        fig.set_size_inches((out_shape[0]/fig.get_dpi(), out_shape[1]/fig.get_dpi()))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.specgram(data, cmap=cmap, Fs=2, noverlap=noverlap)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def load_image(self, dummy_file_path):
        """
        Loads gray scaled image, extracts one dimension from it (other dimensions have the same data)

        Parameters
        ----------
        dummy_file_path : string
            Path location of the image.

        Returns
        -------
        array
            Data of the loaded image.

        """
        return matplotlib.pyplot.imread(dummy_file_path)[:,:,0]


if __name__ == '__main__':
    x = Audio_VGD().create()

    x = Audio_VGD().get_datasets()



