import os.path
import torch.nn
import numpy as np
from art.estimators.classification import PyTorchClassifier


class Autoencoder(PyTorchClassifier):

    def __init__(self, model, lr=0.1, batch_size=32, nb_epochs=10):
        self.lr = lr
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        super().__init__(
            model=model,
            clip_values=(0.0, 255.0),
            loss=model.get_criterion(),
            optimizer=model.get_opti(),
            input_shape=model.get_input_shape(),
            nb_classes=model.get_nb_classes()
        )

    def fit(self, x, y, *args, **kwargs):
        '''
        Fit the autoencoder to the data set x
        :param x: the data set to fit
        :return: None
        '''
        # TODO: Parameterize Weight Decay
        # If there is an autoencoder already trained, load it
        abs_path = os.path.abspath(__file__)
        file_directory = os.path.dirname(abs_path)
        gparent_directory = os.path.dirname(os.path.dirname(os.path.dirname(file_directory)))
        target_path = r'models/image/pre-load'
        final_path = os.path.join(gparent_directory, target_path
                                  , self.model.get_path())
        if os.path.exists(final_path):
            self.model.load_state_dict(torch.load(final_path))
            return
        # Start training
        loader = torch.utils.data.DataLoader(dataset=x, batch_size=self.batch_size, shuffle=True)
        # Validation using MSE Loss function
        loss_function = torch.nn.MSELoss()

        # Using an Adam Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=1e-8)

        epochs = self.nb_epochs
        input_shape = self.model.get_input_shape()
        for epoch in range(epochs):
            for image in loader:
                # Reshaping the image to 1d array
                image = image.reshape(-1, input_shape[0] * input_shape[1] * input_shape[2]).float()

                # Output of Autoencoder
                reconstructed = self.model(image)

                # Calculating the loss function
                loss = loss_function(reconstructed, image)

                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save the model
        torch.save(self.model.state_dict(), final_path)

    def predict(self, x, *args, **kwargs):
        '''
        Predict the output for the given input x
        :param x: The input images
        :return: Predicted returns, should be simillar to the input
        '''
        self.model.eval()
        input_shape = self.model.get_input_shape()
        # Reshape into a 1d array for each image
        x = x.reshape(-1, input_shape[0] * input_shape[1] * input_shape[2])
        # Predict and reshape back to 3d images
        return super().predict(x.astype(np.float32)).reshape(-1, input_shape[0], input_shape[1], input_shape[2])