from backdoorpony.models.graph.zaixizhang.graphcnn import GraphCNN

#A model instance for the MUTAG dataset that inherits from general graphcnn model.
class Gcnn_MUTAG(GraphCNN):
    def __init__(self, num_layers = 5, num_mlp_layers = 2, input_dim = 59, hidden_dim = 64, output_dim = 2, final_dropout = 0.5, learn_eps = False,
                 graph_pooling_type = "sum", neighbor_pooling_type = "sum", device = 0):
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

        super().__init__(num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type,
                         neighbor_pooling_type, device)

    
