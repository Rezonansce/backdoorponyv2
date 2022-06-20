import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from backdoorpony.datasets.audio_VGD import Audio_VGD


class TestDataLoader(TestCase):
    def test_get_data(self):

        datapoints = [[5], [3], [6], [4], [3], [7], [6], [2], [9], [0]]
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        data = [[[0.5568628,  0.5568628,  0.5568628,  1.        ],
         [0.5568628,  0.5568628,  0.5568628,  1.        ],
         [0.54509807, 0.54509807, 0.54509807, 1.        ],
         [0.50980395, 0.50980395, 0.50980395, 1.        ],
         [0.4862745,  0.4862745,  0.4862745,  1.        ],
         [0.48235294, 0.48235294, 0.48235294, 1.        ],
         [0.4745098,  0.4745098,  0.4745098,  1.        ],
         [0.4745098,  0.4745098,  0.4745098,  1.        ],
         [0.4745098,  0.4745098,  0.4745098,  1.        ],
         [0.5019608,  0.5019608,  0.5019608,  1.        ],
         [0.5294118,  0.5294118,  0.5294118,  1.        ],
         [0.5372549,  0.5372549,  0.5372549,  1.        ],
         [0.53333336, 0.53333336, 0.53333336, 1.        ],
         [0.52156866, 0.52156866, 0.52156866, 1.        ],
         [0.5137255,  0.5137255,  0.5137255,  1.        ],
         [0.5372549,  0.5372549,  0.5372549,  1.        ],
         [0.5764706,  0.5764706,  0.5764706,  1.        ],
         [0.60784316, 0.60784316, 0.60784316, 1.        ],
         [0.6,        0.6,        0.6,        1.        ],
         [0.5686275,  0.5686275,  0.5686275,  1.        ],
         [0.54509807, 0.54509807, 0.54509807, 1.        ],
         [0.54901963, 0.54901963, 0.54901963, 1.        ],
         [0.6,        0.6,        0.6,        1.        ],
         [0.654902,   0.654902,   0.654902,   1.        ],
         [0.6666667,  0.6666667,  0.6666667,  1.        ],
         [0.6627451,  0.6627451,  0.6627451,  1.        ],
         [0.65882355, 0.65882355, 0.65882355, 1.        ],
         [0.654902,   0.654902,   0.654902,   1.        ]]]

        data = np.array(data)

        with patch("glob.glob", return_value=[0, 1]) as gl:
            with patch("matplotlib.pyplot.imread", return_value=data) as load:
                audio = Audio_VGD()
                (X_train, y_train), (X_test, y_test) = audio.get_datasets(1, 1)

                self.assertTrue(len(X_train) == 3)
                self.assertTrue(len(y_train) == 3)
                self.assertTrue(len(X_test) == 1)
                self.assertTrue(len(y_test) == 1)

    def test_audio_get_data(self):

        datapoints = [[5], [3], [6], [4], [3], [7], [6], [2], [9], [0]]
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


        with patch("glob.glob", return_value=[0, 1]) as gl:
            with patch("soundfile.read", return_value=([0], 1)) as read:
                audio = Audio_VGD()
                (X_train, y_train), (X_test, y_test) = audio.get_audio_data(1, 1)

                self.assertTrue(len(X_train) == 3)
                self.assertTrue(len(y_train) == 3)
                self.assertTrue(len(X_test) == 1)
                self.assertTrue(len(y_test) == 1)



if __name__ == '__main__':
    data = [[[0.5568628, 0.5568628, 0.5568628, 1.0],
          [0.5568628, 0.5568628, 0.5568628, 1.0],
          [0.54509807, 0.54509807, 0.54509807, 1.0]]]
    print(data[:,:,0])
    unittest.main()
