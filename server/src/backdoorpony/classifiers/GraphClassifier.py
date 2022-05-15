import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from backdoorpony.classifiers.abstract_classifier import AbstractClassifier


class GraphClassifier(PyTorchClassifier, AbstractClassifier):
    def __init__(self, model):
        '''Initiate the classifier

        Parameters
        ----------
        model :
            Model that the classifier should be based on

        Returns
        ----------
        None
        '''
        criterion = nn.CrossEntropyLoss()
        opti = optim.Adam(model.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(opti, step_size=50, gamma=0.1)
        super().__init__(
            model=model,
            clip_values=(0.0, 255.0),
            loss=criterion,
            optimizer=opti,
            input_shape=5,
            nb_classes=2,
        )
        self.batch_size = 32
        self.iters_per_epoch = 50


    def fit(self, x, y, *args, **kwargs):
        '''Fits the classifier to the training data
        First normalises the data and transform it to the format used by PyTorch.

        Parameters
        ----------
        x :
            Data that the classifier will be trained on
        y :
            Labels that the classifier will be trained on

        Returns
        ----------
        None
        '''
        # Training settings
        # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
        # set up seeds and gpu device
        torch.manual_seed(0)
        np.random.seed(0)


        for epoch in range(1, self.iters_per_epoch + 1):
            self.train(self.model, x[0], self.optimizer)
            self.scheduler.step()

    def train(self, model, train_graphs, optimizer):
        model.train()

        loss_accum = 0
        for i in range(self.iters_per_epoch):
            selected_idx = np.random.permutation(len(train_graphs))[:self.batch_size]
            
            batch_graph = [train_graphs[idx] for idx in selected_idx]
            output = model(batch_graph)

            labels = torch.LongTensor([graph.label for graph in batch_graph])

            # compute loss
            loss = self.loss(output, labels)

            # backprop
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = loss.detach().cpu().numpy()
            loss_accum += loss

        average_loss = loss_accum / self.iters_per_epoch

        return average_loss

    def predict(self, x, *args, **kwargs):
        '''Classifies the given input

        Parameters
        ----------
        x :
            The dataset the classifier should classify

        Returns
        ----------
        prediction :
            Return format is a numpy array with the probability for each class
        '''
        self.model.eval()
        output = self.pass_data_iteratively(x)
        pred = output.max(1, keepdim=True)[1]
        return pred

    def pass_data_iteratively(self, graphs, minibatch_size=1):
        self.model.eval()
        output = []
        idx = np.arange(len(graphs))
        for i in range(0, len(graphs), minibatch_size):
            sampled_idx = idx[i:i + minibatch_size]
            if len(sampled_idx) == 0:
                continue
            output.append(self.model([graphs[j] for j in sampled_idx]).detach())
        return torch.cat(output, 0)
