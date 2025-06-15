import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.nn import radius_graph, knn_graph
from .common import GaussianSmearing, MLP, batch_hybrid_edge_connection, NONLINEARITIES

class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, update_x=True, act_fn='relu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10.
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = nn.LayerNorm(hidden_dim)
        self.edge_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            # self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
            x_mlp = [nn.Linear(hidden_dim, hidden_dim), NONLINEARITIES[act_fn]]
            layer = nn.Linear(hidden_dim, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            x_mlp.append(layer)
            x_mlp.append(nn.Tanh())
            self.x_mlp = nn.Sequential(*x_mlp)

        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, h, x, edge_index):
        src, dst = edge_index

        hi, hj = h[dst], h[src]
        rel_x = x[dst] - x[src]
        d_sq = torch.sum(rel_x ** 2, -1, keepdim=True)
        mij = self.edge_mlp(torch.cat([hi, hj], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])
        # h update in Eq(6)
        h = self.norm(h + self.node_mlp(torch.cat([mi, h], -1)))
        return h, x


class Protein3D_Representation(nn.Module):
    def __init__(self, num_layers, hidden_dim, edge_feat_dim, k=8, update_x=True, act_fn='relu', norm=False):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.k = k
        self.net = self._build_network()

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)

    # todo: refactor
    def _connect_edge(self, x, prot_batchs):
        edge_index = knn_graph(x, k=self.k, batch=prot_batchs, flow='source_to_target')
        return edge_index

    def forward(self, h, x, prot_batchs, return_all=False):
        all_x = []
        all_h = []
        for l_idx, layer in enumerate(self.net):
            edge_index = self._connect_edge(x, prot_batchs)
            h, x = layer(h, x, edge_index)
            all_x.append(x)
            all_h.append(h)
        # outputs = {'x': x, 'h': h}
        # if return_all:
        #     outputs.update({'all_x': all_x, 'all_h': all_h})
        return h, x, all_h, edge_index
