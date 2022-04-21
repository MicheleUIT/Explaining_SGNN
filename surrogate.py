from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.nn.inits import reset
from torch_geometric.nn import global_mean_pool



class MGINConv0(MessagePassing):
    r"""
    Masked vesion of GINZero
    """
    def __init__(self, nn: Callable,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x:Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


def make_gin_conv(input_dim: int, out_dim: int) -> MGINConv0:
    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, out_dim),
        torch.nn.BatchNorm1d(out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim))
    return MGINConv0(mlp)


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim*num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(make_gin_conv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(make_gin_conv(hidden_dim, hidden_dim))


        self.readout_head =torch.nn.Sequential(
                torch.nn.Linear(hidden_dim*num_layers, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, out_dim)
            )
        
        self.pool = global_mean_pool


    def forward(self, x, edge_index, batch=None, edge_weight=None):
        if batch is None: 
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.embedding(x, edge_index, edge_weight)
        x = self.pool(x, batch)
        return self.readout_head(x)

    def embedding(self, x, edge_index, edge_weight=None):
        h_list = []
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index, edge_weight)
            if (i != self.num_layers-1):
                x = x.relu()
            x = F.dropout(x, 0.2, training=self.training)
            h_list.append(x)
        return torch.cat(h_list, dim=1)
