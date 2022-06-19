from unittest import TestCase
from art.estimators.classification import PyTorchClassifier
from unittest.mock import patch
from unittest.mock import call, MagicMock, Mock

from backdoorpony.defence_helpers.autoencoder_util.Autoencoder import Autoencoder
from backdoorpony.defence_helpers.autoencoder_util.AutoencoderCNN import AutoencoderCNN

class TestAutencoder(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAutencoder, self).__init__(*args, **kwargs)
        self.input_shape = (1, 1, 2)
        self.cnn = AutoencoderCNN(input_shape=self.input_shape, path="invalid_path")
        self.nb_epcohs = 1
        self.model = Autoencoder(self.cnn, nb_epochs=self.nb_epcohs)
        self.flat_input_shape = 2


    def test_init(self):
        self.assertTrue(isinstance(self.model, PyTorchClassifier))

    @patch('torch.save')
    @patch('torch.optim.Adam')
    @patch('torch.nn.MSELoss')
    @patch('torch.utils.data.DataLoader')
    @patch("art.estimators.classification.PyTorchClassifier.model")
    def test_fit(self, mock_model, mock_loader, mock_loss, mock_optim, mock_save):
        # Arrange
        img1 = MagicMock()
        img2 = MagicMock()
        img3 = MagicMock()
        loss = MagicMock()
        optim = MagicMock()
        state_dict = MagicMock()
        mock_optim.return_value = optim
        mock_loss.return_value = loss
        mock_loader.return_value = [img1, img2]
        img1.reshape.return_value = img1
        img2.reshape.return_value = img2
        img1.float.return_value = img1
        img2.float.return_value = img2
        mock_model.return_value = img3
        mock_model.state_dict.return_value = state_dict
        mock_model.get_input_shape.return_value = self.input_shape

        # Act
        self.model.fit(input, input)

        # Assert
        img1.reshape.assert_called_once_with(-1, self.flat_input_shape)
        img2.reshape.assert_called_once_with(-1, self.flat_input_shape)
        loss_calls = [call(img3, img1), call().backward()
            , call(img3, img2), call().backward()]
        loss.assert_has_calls(loss_calls, any_order=False)
        optim_calls = [call.zero_grad(), call.step()
                       , call.zero_grad(), call.step()]
        optim.assert_has_calls(optim_calls, any_order=False)
        mock_save.assert_called_once()