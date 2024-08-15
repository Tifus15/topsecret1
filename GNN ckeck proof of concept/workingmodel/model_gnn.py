import dgl
import torch.nn as nn
import torch
import torch.nn.functional as F
from model import *
import dgl.function as fn

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim,bias =True):
        super(GATLayer, self).__init__()
        #self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self,g,h):
        # equation (1)
        z = self.fc(h)
        #print(z)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')



    
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats,bias = True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats,bias = bias)
        

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            gcn_msg = fn.copy_u(u="h", out="m")
            gcn_reduce = fn.sum(msg="m", out="h")
            g.ndata["h"] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata["h"]
            return self.linear(h)
    
    
class GNN(nn.Module):
    def __init__(self,g,acts, in_dim=2, out_dim=2, hidden=[64, 64],bias=True,type="gat"):
        super().__init__()
        layers = []
        layer_sizes = [in_dim] + hidden
        self.g = g
        
        for layer_index in range(1, len(layer_sizes)):
            ind = layer_sizes[layer_index-1]
            outd = layer_sizes[layer_index]
            if type == "gat":
                layer = GATLayer(ind,outd,bias=bias)
            elif type =="gcn":
                layer = GCNLayer(ind,outd,bias=bias)
            layers += [layer,act_module(acts[layer_index-1])]
        ind = layer_sizes[-1]
        if type == "gat":
            layer = GATLayer(ind,out_dim,bias=bias)
        elif type =="gcn":
            layer = GCNLayer(ind,out_dim,bias=bias)
        layers += [layer]
        self.layers = nn.ModuleList(layers) # A module list registers a list of modules as submodules (e.g. for parameters)
        self.config = {"act_fn": acts.__class__.__name__, "input_size": in_dim, "num_classes": out_dim, "hidden_sizes": hidden}
    def change_graph(self,g):
        self.g = g
        
    def forward(self,t, x):
        #print(x.is_leaf)
       #print("foward")
        for l in self.layers:
            #print(x.shape)
            #print(l)
            if isinstance(l,GATLayer) or isinstance(l,GCNLayer):
                x = l(self.g, x)
            else:
                x=l(x)
        #print("end")
        #print(x.shape)
        return x