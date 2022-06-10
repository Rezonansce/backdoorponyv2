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
    def __init__(self, model, criterion = F.cross_entropy, lr = 0.01, step_size = 50, gamma = 0.1, 
                 batch_size = 32, iters_per_epoch = 50, iters = 50):
        '''Initiate the classifier

        Parameters
        ----------
        model :
            Model that the classifier should be based on

        Returns
        ----------
        None
        '''
        opti = optim.Adam(model.parameters(), lr = lr)
        self.scheduler = optim.lr_scheduler.StepLR(opti, step_size = step_size, gamma = gamma)
        self.batch_size = batch_size
        self.iters_per_epoch = iters_per_epoch
        self.iters = iters
        
        #input_shape & nb_classes can be arbitrary, but need to be initialized
        self.model=model
        self.loss=criterion
        self.optimizer=opti

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
        print(enumerate(x))
        
        torch.manual_seed(0)
        np.random.seed(0)
        
        #assert torch.cuda.is_available(), 'no GPU available'
        cpu = torch.device('cpu')
        cuda = torch.device('cuda')
    
        # print('\nInitialize model')
        # print(model)
        #train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        # print('N trainable parameters:', np.sum([p.numel() for p in train_params]))
    
        # training
        model = self.model
        loss_fn = self.loss
        optimizer = self.optimizer
        
        #model.to(cuda)
        model.to(cpu)
        for iteration in range(self.iters):
            model.train()
            start = time.time()
            train_loss, n_samples = 0, 0
            for batch_id, data in enumerate(x):
                for i in range(len(data)):
                    #data[i] = data[i].to(cuda)
                    data[i] = data[i].to(cpu)
                # if args.use_cont_node_attr:
                #     data[0] = norm_features(data[0])
                optimizer.zero_grad()
                output = model(data)
                if len(output.shape)==1:
                    output = output.unsqueeze(0)
                loss = loss_fn(output, data[4])
                loss.backward()
                optimizer.step()
                self.scheduler.step()
                
    
            #if args.train_verbose and (iteration % args.log_every == 0 or iteration == args.train_epochs - 1):
            #    time_iter = time.time() - start
            #    train_loss += loss.item() * len(output)
            #    n_samples += len(output)
            #    print('Train Epoch: %d\tLoss: %.4f (avg: %.4f) \tsec/iter: %.2f' % (
            #        iteration + 1, loss.item(), train_loss / n_samples, time_iter / (batch_id + 1)))
        
        model.to(cpu)
    
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
        cpu = torch.device('cpu')
        cuda = torch.device('cuda')
        
        predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
        model = self.model
        loss_fn = self.loss
        
        preds = []
        
        correct = 0
        total = 0
        
        model.eval()
        start = time.time()
        for batch_id, data in enumerate(x):
            for i in range(len(data)):
                #data[i] = data[i].to(cuda)
                data[i] = data[i].to(cpu)
            # if args.use_org_node_attr:
            #     data[0] = norm_features(data[0])
            output = model(data)
            if len(output.shape)==1:
                output = output.unsqueeze(0)
            loss = loss_fn(output, data[4], reduction='sum')
            pred = predict_fn(output)
            preds += pred
            
            #FOR DEBUGGING
            correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
            total += len(output)
            
        print("TOTAL ACCURACY: " + str(correct/total))
        return preds
    
    def class_gradient(self, x, *args, **kwargs):
        '''
        ...
        '''
        pass
