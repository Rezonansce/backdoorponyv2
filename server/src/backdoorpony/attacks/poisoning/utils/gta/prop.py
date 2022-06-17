import sys, os
sys.path.append(os.path.abspath('..'))

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from backdoorpony.datasets.utils.gta.datareader import GraphData, DataReader
from backdoorpony.datasets.utils.gta.batch import collate_batch

# run on CUDA
def forwarding(args, bkd_dr: DataReader, model, gids, criterion):
    #assert torch.cuda.is_available(), "no GPU available"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gdata = GraphData(bkd_dr, gids)
    loader = DataLoader(gdata,
                        batch_size=args.batch_size,
                        shuffle=False,   
                        collate_fn=collate_batch)
    

    model.to(device)
    model.eval()
    all_loss, n_samples = 0.0, 0.0
    for batch_idx, data in enumerate(loader):
#         assert batch_idx == 0, "In AdaptNet Train, we only need one GNN pass, batch-size=len(all trainset)"
        for i in range(len(data)):
            data[i] = data[i].to(device)
        output = model(data)
        
        if len(output.shape)==1:
            output = output.unsqueeze(0)
        
        loss = criterion(output, data[4])  # only calculate once
        all_loss = torch.add(torch.mul(loss, len(output)), all_loss)  # cannot be loss.item()
        n_samples += len(output)
        
    all_loss = torch.div(all_loss, n_samples)
    return all_loss


def train_model(args, dr_train: DataReader, model, pset, nset):
    #assert torch.cuda.is_available(), "no GPU available"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    gids = {'pos': pset, 'neg': nset}
    gdata = {}
    loader = {}
    for key in ['pos', 'neg']:
        gdata[key] = GraphData(dr_train, gids[key])
        loader[key] = DataLoader(gdata[key],
                                batch_size=args.batch_size,
                                shuffle=False,   
                                collate_fn=collate_batch)
    
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
    loss_fn = F.cross_entropy
    
    pos_samples = []

    model.train()
    for epoch in range(args.train_epochs):
        optimizer.zero_grad()
        
        losses = {'pos': 0.0, 'neg': 0.0}
        n_samples = {'pos': 0.0, 'neg': 0.0}
        for key in ['pos', 'neg']:
            for batch_idx, data in enumerate(loader[key]):
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                    #save poisoned train samples for execution_history
                    if key == "pos":
                        pos_samples.append(data[i])
                        
                output = model(data)
                if len(output.shape)==1:
                    output = output.unsqueeze(0)
                losses[key] += loss_fn(output, data[4])*len(output)
                n_samples[key] += len(output)

        
            losses[key] = torch.div(losses[key], n_samples[key])
        loss = losses['pos'] + args.lambd*losses['neg']
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    
    #return positive train samples for execution_history
    return loader["pos"]


    
def evaluate(args, dr_test: DataReader, model, gids):  
    # separate bkd_test/clean_test gids
    softmax = torch.nn.Softmax(dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gdata = GraphData(dr_test, gids)
    loader = DataLoader(gdata,
                        batch_size=args.batch_size,
                        shuffle=False,   
                        collate_fn=collate_batch)
    
    loss_fn = F.cross_entropy
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
    
    pos_samples = []
    model.eval()
    test_loss, correct, n_samples, confidence = 0, 0, 0, 0
    for batch_idx, data in enumerate(loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)
            #save poisoned test samples for execution_history
            pos_samples.append(data[i])
        output = model(data)  # not softmax yet
        if len(output.shape)==1:
            output = output.unsqueeze(0)
        loss = loss_fn(output, data[4], reduction='sum') 
        test_loss += loss.item()
        n_samples += len(output)
        pred = predict_fn(output)
        
        correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
        confidence += torch.sum(torch.max(softmax(output), dim=1)[0]).item()
    acc = 100. * correct / n_samples
    confidence = confidence / n_samples
    
    print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.2f%s), Average Confidence %.4f' % (
        test_loss / n_samples, correct, n_samples, acc, '%', confidence))
    
    #also return positive test samples for execution_history
    return acc, loader
