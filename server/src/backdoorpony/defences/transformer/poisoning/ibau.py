import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import torch
from backdoorpony.defence_helpers import hypergrad as hg
import torch.nn.functional as F
import torch.nn as nn
import random
from backdoorpony.classifiers.ImageClassifier import ImageClassifier

__name__ = 'ibau'
__category__ = 'transformer'
__input_type__ = 'image'
__defaults__ = {
    'batch_size': {
        'pretty_name': 'Batch Size',
        'default_value': [100],
        'info': 'The size of the batch used in the unlearning step.'
    },
    'learning_rate': {
        'pretty_name': 'Learning Rate',
        'default_value': [0.001],
        'info': 'The learning rate used for the unlearning step'
    }
}
__link__ = 'https://arxiv.org/pdf/2110.03735.pdf'
__info__ = 'Info about I-BAU'

def get_eval_data(test_data):
    '''
    Splits the test_data in two equal halves
    One is used for unlearning and the other for the hypergradient computation
    :param test_data: images + labels
    :return: data split in equal sides
    '''
    half_size = int(len(test_data[1]) / 2)
    y_test = np.asarray(test_data[1])
    y_test = torch.Tensor(y_test.reshape((-1,)).astype(np.int))
    x_test = torch.Tensor(test_data[0].astype('float32'))

    test_set = TensorDataset(x_test[half_size:], y_test[half_size:])
    unl_set = TensorDataset(x_test[:half_size], y_test[:half_size])

    return test_set, unl_set

def run(clean_classifier, test_data, execution_history, defence_params):
    '''
    Runns the I-BAU defence
    clean_classifier :
        Classifier that has not been tampered with, i.e. is clean
    test_data :
        Data that the clean classifier will be validated on as a tuple with (inputs, labels)
    execution_history :
        Dictionary with paths of attacks/defences taken to achieve classifiers, if any
    defence_params :
        Dictionary with the parameters for the defence (one value per parameter)
    To fully understand the algorithm, go to https://arxiv.org/pdf/2110.03735.pdf
    Returns
    ----------
    Returns the updated execution history dictionary
    '''
    print('Instantiating an I-BAU defence.')
    def poison_condition(x): return (x == np.zeros(len(x))).all()
    key_index = 0
    new_execution_history = deepcopy(execution_history)
    for entry in execution_history.values():
        for batch_size in defence_params['batch_size']['value']:
            for learning_rate in defence_params['learning_rate']['value']:
                new_entry = deepcopy(entry)
                defence_classifier = run_def(deepcopy(entry['dict_others']['poison_classifier']), deepcopy(test_data)
                                        , clean_classifier.model.get_opti(), batch_size=batch_size
                                        , lr=learning_rate)
                new_entry.update({
                        'defence': __name__,
                        'defenceCategory': __category__,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'dict_others': {
                            'poison_classifier': deepcopy(defence_classifier),
                            'poison_inputs': deepcopy(entry['dict_others']['poison_inputs']),
                            'poison_labels': deepcopy(entry['dict_others']['poison_labels']),
                            'poison_condition': deepcopy(poison_condition)
                        }
                    })
                key_index += 1
                new_execution_history.update({'I-BAU' + str(key_index): new_entry})

    return new_execution_history

def run_def(classifier, data_set, optimizer, batch_size=50
            , lr=0.001, n_rounds=5, k=5):
    '''
    Run the I-BAU defense
    :param classifier: The poisoned classifier
    :param data_set: The data set used for unlearning (it must be clean)
    :param optimizer: Type of outer loop optimizer (TODO: the user should choose it)
    :param batch_size: The size of the batch of the unlearn loader
    :param lr: Learning rate of outer loop optimizer
    :param n_rounds: The maximum number of unlearning rounds
    :param k: The maximum number of fixed point iterations
    :return: The
    '''
    print("=> Setting up I-BAU defence...")
    model = classifier.model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    test_set, unl_set = get_eval_data(data_set)
    # data loader for the unlearning step
    unlloader = DataLoader(unl_set, batch_size=batch_size, shuffle=False)

    model = classifier.to(device)
    outer_opt = optimizer
    outer_opt['lr'] = lr

    ### define the inner loss L2
    def loss_inner(perturb, model_params):
        images = images_list[0].to(device)
        labels = labels_list[0].long().to(device)
        #     per_img = torch.clamp(images+perturb[0],min=0,max=1)
        per_img = images + perturb[0]
        per_logits = model.forward(per_img)
        loss = F.cross_entropy(per_logits, labels, reduction='none')
        loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(perturb[0]), 2)
        return loss_regu

    ### define the outer loss L1
    def loss_outer(perturb, model_params):
        portion = 0.01
        images, labels = images_list[batchnum].to(device), labels_list[batchnum].long().to(device)
        patching = torch.zeros_like(images)
        number = images.shape[0]
        rand_idx = random.sample(list(np.arange(number)), int(number * portion))
        patching[rand_idx] = perturb[0]
        #     unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
        unlearn_imgs = images + patching
        logits = model(unlearn_imgs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return loss

    images_list, labels_list = [], []
    for index, (images, labels) in enumerate(unlloader):
        images_list.append(images)
        labels_list.append(labels)
    inner_opt = hg.GradientDescent(loss_inner, 0.1)

    ### inner loop and optimization by batch computing
    print("=> Conducting Defence..")
    model.eval()

    for round in range(n_rounds):
        batch_pert = torch.zeros_like(test_set.tensors[0][:1], requires_grad=True)
        batch_opt = torch.optim.SGD(params=[batch_pert], lr=10)

        for images, labels in unlloader:
            images = images.to(device)
            ori_lab = torch.argmax(model.forward(images), axis=1).long()
            #         per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
            per_logits = model.forward(images + batch_pert)
            loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
            loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(batch_pert), 2)
            batch_opt.zero_grad()
            loss_regu.backward(retain_graph=True)
            batch_opt.step()

        # l2-ball
        # pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
        pert = batch_pert

        # unlearn step
        for batchnum in range(len(images_list)):
            outer_opt.zero_grad()
            hg.fixed_point(pert, list(model.parameters()), k, inner_opt, loss_outer)
            outer_opt.step()

    return classifier



