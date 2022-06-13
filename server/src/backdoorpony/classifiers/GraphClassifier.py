import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from backdoorpony.classifiers.abstract_classifier import AbstractClassifier
from tqdm import tqdm


class GraphClassifier(PyTorchClassifier, AbstractClassifier):
    def __init__(self, model, criterion = nn.CrossEntropyLoss(), lr = 0.01, step_size = 50, gamma = 0.1, 
                 batch_size = 32, iters_per_epoch = 50, iters = 50, clip_values = (0.0, 255.0)):
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
        opti = optim.Adam(model.parameters(), lr = lr)
        self.scheduler = optim.lr_scheduler.StepLR(opti, step_size = step_size, gamma = gamma)
        self.batch_size = batch_size
        self.iters_per_epoch = iters_per_epoch
        self.iters = iters
        
        #input_shape & nb_classes can be arbitrary, but need to be initialized
        super().__init__(
            model=model,
            clip_values=clip_values,
            loss=criterion,
            optimizer=opti,
            input_shape=420,
            nb_classes=2,
        )


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
        


        for epoch in tqdm(range(1, self.iters + 1)):
            print(epoch)
            self.train(self.model, x[0], self.optimizer)
            self.scheduler.step()

    def train(self, model, train_graphs, optimizer):
        '''Executes one epoch of training procedure
        Uses mini-batch gradient descent & backpropagation

        Parameters
        ----------
        model :
            model that is used for classification
        train_graphs :
            graphs used to train the model
        optimizer :
            used to speed up the computations

        Returns
        ----------
        The training loss averaged across all iterations (mini-batches)
        '''
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
            Return format is a numpy array with the predicted class for each sample, classified using MLE
        '''
        self.model.eval()
        output = self.pass_data_iteratively(self.model, x)
        pred = output.max(1, keepdim=True)[1]
        return pred

    def pass_data_iteratively(self, model, graphs, minibatch_size=1):
        '''pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)

        Parameters
        ----------
        model :
            model that is used for classification
        graphs :
            graphs to be classified
        minibatch_size :
            size of the minibatch 

        Returns
        ----------
        The training loss averaged across all iterations
        '''
        model.eval()
        output = []
        idx = np.arange(len(graphs))
        for i in range(0, len(graphs), minibatch_size):
            sampled_idx = idx[i:i + minibatch_size]
            if len(sampled_idx) == 0:
                continue
            output.append(model([graphs[j] for j in sampled_idx]).detach())
        return torch.cat(output, 0)
