import sys, os

sys.path.append(os.path.abspath('..'))

import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import torch.nn as nn
from backdoorpony.classifiers.abstract_classifier import AbstractClassifier
from backdoorpony.datasets.utils.gta.datareader import GraphData, DataReader
from tqdm import tqdm


class GraphClassifier(AbstractClassifier):
    def __init__(self, model, step_size=50, gamma=0.1):
        '''Initiate the classifier

        Parameters
        ----------
        model :
            Model that the classifier should be based on

        Returns
        ----------
        None
        '''
        if (model.optim == "Adam"):
            opti = optim.Adam(model.parameters(), lr=model.lr)
        else:
            opti = optim.SGD(model.parameters(), lr=model.lr)
        
        if (model.loss == "CrossEntropy"):
            criterion = F.cross_entropy
        else:
            criterion = F.nll_loss
            
        self.scheduler = optim.lr_scheduler.StepLR(opti, step_size=step_size, gamma=gamma)
        self.epochs = model.epochs

        # input_shape & nb_classes can be arbitrary, but need to be initialized
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss = criterion
        self.optimizer = opti

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

        # training
        model = self.model
        loss_fn = self.loss
        optimizer = self.optimizer

        for iteration in range(self.epochs):
            print(iteration)
            model.train()
            train_loss, n_samples = 0, 0
            for batch_id, data in enumerate(x):
                for i in range(len(data)):
                    # data[i] = data[i].to(cuda)
                    data[i] = data[i].to(self.device)
                # if args.use_cont_node_attr:
                #     data[0] = norm_features(data[0])
                optimizer.zero_grad()
                output = model(data)
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                loss = loss_fn(output, data[4])
                loss.backward()
                optimizer.step()
                self.scheduler.step()

            # if args.train_verbose and (iteration % args.log_every == 0 or iteration == args.train_epochs - 1):
            #    time_iter = time.time() - start
            #    train_loss += loss.item() * len(output)
            #    n_samples += len(output)
            #    print('Train Epoch: %d\tLoss: %.4f (avg: %.4f) \tsec/iter: %.2f' % (
            #        iteration + 1, loss.item(), train_loss / n_samples, time_iter / (batch_id + 1)))

        return model

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

        predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
        model = self.model
        loss_fn = self.loss

        preds = []

        correct = 0
        total = 0

        model.eval()
        for batch_id, data in enumerate(x):
            for i in range(len(data)):
                # data[i] = data[i].to(cuda)
                data[i] = data[i].to(self.device)
            # if args.use_org_node_attr:
            #     data[0] = norm_features(data[0])
            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            loss = loss_fn(output, data[4], reduction='sum')
            pred = predict_fn(output)

            # FOR DEBUGGING
            correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
            total += len(output)

            preds += [o.detach().cpu().numpy() for o in output]

        print("TOTAL ACCURACY: " + str(correct / total))
        return preds

    def class_gradient(self, x, *args, **kwargs):
        '''
        ...
        '''
        pass
