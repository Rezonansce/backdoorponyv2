import backdoorpony.datasets.utils.FSDD.utils.spectogramer as SP
import os
from backdoorpony.datasets.utils.FSDD.utils.fsdd import FSDD
from sklearn.model_selection import train_test_split



"""
This file creates the Audio MNIST dataset

:param test_size: The size of the test set. 0.10 which means 10%
:return (x_raw_train, y_raw_test), (x_raw_test, y_raw_test)
"""
class Audio_MNIST(object):
    def __init__(self, test_size = 0.10):
        self.test_size = test_size

    def get_datasets(self):
        '''Generates datapoints if they are missing, then splits the dataset to train and test

        Returns
        ----------
        (X_train, y_train) :
            A tuple with two numpy arrays: one with the sample and one with the corresponding labels
        (X_test, y_test) :
            A tuple with two numpy arrays: one with the sample and one with the corresponding labels
        '''
        dir_path = os.path.dirname(os.path.realpath(__file__))
        audio_dir = dir_path + "/utils/FSDD/recordings/"
        spectrogram_dir = dir_path + "/utils/FSDD/spectrograms/"

        #if data is not processed process it
        if (not os.path.isdir(spectrogram_dir)):
            os.mkdir(spectrogram_dir)
        if len(os.listdir(spectrogram_dir)) == 0:
            SP.dir_to_spectrogram(audio_dir, spectrogram_dir)

        #generate dataset from sound (waw) files
        fsdd = FSDD(spectrogram_dir)
        dataset, labels = fsdd.get_spectrograms()

        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=self.test_size, random_state=42)

        return (X_train, y_train), (X_test, y_test)


"""
if __name__=='__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    audio_dir = dir_path + "/utils/FSDD/recordings/"
    spectrogram_dir = dir_path + "/utils/FSDD/spectrograms/"

    if len(os.listdir(spectrogram_dir)) == 0:
        SP.dir_to_spectrogram(audio_dir, spectrogram_dir)

    fsdd = FSDD(spectrogram_dir)
    dataset, labels = fsdd.get_spectrograms()

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.10, random_state=42)

    print(np.array(X_train).shape)
    """
