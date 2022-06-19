import torch.nn.functional as F
from backdoorpony.models.graph.gta.sage import GraphSAGE

__name__ = "AIDS_sage"
__input_type__ = "graph"
__defaults__ = {
    'hidden_dim': {
        'pretty_name': 'Hidden layer dimensions',
        'default_value': [64, 16],
        'info': 'Dimensions of the respective hidden layers. Please, provide the input as a list of dimensions for each layer.'
    },
    'dropout': {
        'pretty_name': 'Dropout',
        'default_value': [0.2],
        'info': 'Randomly set some of the elements of the input tensor to zero with the given probability. Used for regularization.'
    },
    "activation": {
        "pretty_name": "Activation function (relu/sigmoid)",
        "default_value": ["relu"],
        "info": 'Non-linear activation function. Can be relu (ReLu) or sigmoid (Sigmoid).'
    },
    "aggregator": {
        "pretty_name": "Aggregation scheme (mean/gcn/pool/lstm)",
        "default_value": ["gcn"],
        "info": "Aggregation function used to construct new node embeddings. Can be mean (mean aggregator), gcn (Graph Convolutional Network), pool (max pooling) or lstm (Long-short term memory network). If the input is not valid, gcn will be chosen."
    },
    "optim": {
        "pretty_name": "Optimizer (Adam, SGD)",
        "default_value": ["Adam"],
        "info": 'The optimizer used in the training process. Currently, only "Adam" and "SGD" are available.' +
                'If the input is not valid, Adam optimizer will be chosen.'
    },
    'learning_rate': {
        'pretty_name': 'Learning Rate',
        'default_value': [0.01],
        'info': 'The learning rate of the optimizer.'
    },
    "epochs": {
        "pretty_name": "Number of epochs",
        "default_value": [50],
        "info": "Controls the number of trainining iterations."
    },
    "criterion": {
        "pretty_name": "Loss function (CrossEntropy, NLL)",
        "default_value": ["CrossEntropy"],
        "info": 'The loss function used by the model in the training process. Currently, only "CrossEntroy" (CrossEntropyLoss) and "NLL" (Negative Log Likelihood) are available.' +
                'If the input is not valid, CrossEntropy criterion will be chosen.'
    }

}
__link__ = 'https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html'
__info__ = '''LSTM with a head sigmoid layer'''

#A model instance for the AIDS dataset that inherits from general graphcnn model.
class AIDS_sage(GraphSAGE):
    def __init__(self, model_parameters):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''
        
        if (model_parameters["optim"]["value"][0] != "SGD"):
            optim = "Adam"
        else:
            optim = "SGD"
            
        if (model_parameters["criterion"]["value"][0] != "NLL"):
            loss = "CrossEntropy"
        else:
            loss = "NLL"

        aggregator = model_parameters["aggregator"]["value"][0]
        if (aggregator != "mean" and aggregator != "gcn" and aggregator != "pool" and aggregator != "lstm"):
            aggregator = "gcn"
        
        self.optim = optim
        self.lr = model_parameters["learning_rate"]["value"][0]
        self.epochs = model_parameters["epochs"]["value"][0]
        self.loss = loss
        
        if (model_parameters["activation"]["value"][0] == "relu"):
            activation = F.relu
        else:
            activation = F.sigmoid
        super().__init__(64, 2, model_parameters["hidden_dim"]["value"], model_parameters["dropout"]["value"][0], 
                         activation, aggregator)
