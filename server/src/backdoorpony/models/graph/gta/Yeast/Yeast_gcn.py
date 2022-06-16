import torch.nn.functional as F
from backdoorpony.models.graph.gta.gcn import GCN

#A model instance for the MUTAG dataset that inherits from general graphcnn model.
class Yeast_gcn(GCN):
    def __init__(self, in_dim = 102, out_dim = 2,
                 hidden_dim=[64, 32],  # GNN layers + 1 layer MLP
                 dropout=0.2,
                 activation=F.relu):
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

        super().__init__(in_dim, out_dim, hidden_dim, dropout, activation)

