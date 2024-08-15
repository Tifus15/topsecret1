import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import *




class GATLayer(nn.Module):
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2,bias=True):
        """
        Args:
            c_in: Dimensionality of input features
            c_out: Dimensionality of output features
            num_heads: Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads: If True, the output of the different heads is concatenated instead of averaged.
            alpha: Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads,bias = bias)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out))  # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """Forward.

        Args:
            node_feats: Input features of the node. Shape: [batch_size, c_in]
            adj_matrix: Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs: If True, the attention weights are printed during the forward pass
                               (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        # Returns indices where the adjacency matrix is not 0 => edges
        edges = adj_matrix.nonzero(as_tuple=False)
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat(
            [
                torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
                torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0),
            ],
            dim=-1,
        )  # Index select returns a tensor with node_feats_flat being indexed at the desired positions

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum("bhc,hc->bh", a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum("bijh,bjhc->bihc", attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats

class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out,bias = True):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out,bias = bias)

    def forward(self, node_feats, adj_matrix):
        """Forward.

        Args:
            node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats
class GNN(nn.Module):
    def __init__(self,cin,cout,hid=[128],acts=["tanh"],bias=True,type="GAT", **kwargs):
        super().__init__()
        layers = []
        layer_sizes = [cin] + hid
        for layer_index in range(1, len(layer_sizes)):
            ind = layer_sizes[layer_index-1]
            outd = layer_sizes[layer_index]
            if type == "GAT":
                layer = GATLayer(ind,outd,bias=bias, **kwargs)
            elif type == "GCN":
                layer = GCNLayer(ind,outd,bias=bias)
            elif type == "LP":
                layer = LPLayer(ind,outd,bias=bias)
            layers += [layer,function_act(acts[layer_index-1])]
        ind = layer_sizes[-1]
        if type == "GAT":
            layer = GATLayer(ind,cout,bias=bias,**kwargs)
        elif type == "GCN":
            layer = GCNLayer(ind,cout,bias=bias)
        elif type == "LP":
            layer = LPLayer(ind,cout,bias=bias)
        layers += [layer]
        self.layers = nn.ModuleList(layers) # A module list registers a list of modules as submodules (e.g. for parameters)
        self.config = {"act_fn": acts.__class__.__name__, "input_size": cin, "num_classes": cout, "hidden_sizes": hid}
        
    def forward(self,x,adj):
        for l in self.layers:
            if isinstance(l,GCNLayer) or isinstance(l,GATLayer):
                x = l(x, adj)
                
            elif isinstance(l,LPLayer):
                x, lp =l(x)
        
        if isinstance(l,GCNLayer) or isinstance(l,GATLayer):
            return x
        elif isinstance(l,LPLayer):
            return x, lp
        
class LPLayer(nn.Module):
    def __init__(self, c_in, c_out,bias = True,threshold=0.6):
        super().__init__()
        self.val = threshold
        self.projection = nn.Linear(c_in, c_out,bias = bias)  
        #self.adj_projection = nn.Linear(c_in*2,1,bias=bias)
        self.adj_projection = nn.Linear(c_in,c_out,bias=bias)
        self.act1 = nn.Tanh()
        self.act2 = nn.ReLU()
    def forward(self,x):
        batches , nodes = x.shape[0], x.shape[1]
        e_x = self.adj_projection(x)
        #print(e_x.shape)
        # creating src and dst nodes to represent edges
        srcl = []
        dstl = []
        for i in range(nodes):
            src_temp = torch.cat(([e_x[:,i,:].unsqueeze(-1)]*nodes),dim=-1)
            dst_temp = e_x.transpose(1,2)
            srcl.append(src_temp)
            dstl.append(dst_temp)
        src = torch.cat((srcl),dim=-1).transpose(1,2)
        dst = torch.cat((dstl),dim=-1).transpose(1,2)
        
        #e_input = torch.cat((src,dst),dim=-1)
        #print(e_input.shape)
        # creating the values for adj
        #lp_vec = self.adj_projection(e_input)
        # make values in value from 0-1
        src_d = src.unsqueeze(2)
        dst_d = dst.unsqueeze(2).transpose(2,3)
        #print(src_d.shape)
        #print(dst_d.shape)
        lp_vec = src_d @ dst_d
        #print(lp_vec.shape)
        lp_vec_reducted = self.act2(self.act1(lp_vec))
        #print(lp_vec_reducted.shape)
        # all smaller then given value are 0
        #lp_tresh = (lp_vec_reducted > self.val).float()
        # make shape coresponding to the edges and adj
        lp = lp_vec_reducted .reshape(batches,nodes,nodes)
        #lp = lp_vec_reducted
        #print(lp)
        # GCN with created adj
        num_neighbours = nodes
        node_feats = e_x
        node_feats = torch.bmm(lp, node_feats)
        node_feats = node_feats / num_neighbours
        #print("node out {}".format(node_feats.shape))
        return node_feats, lp
         
        
        
        
              
        
if __name__ == "__main__":
    #model = GNN(3,1,[64,64],["tanh",""],type="GAT")
    model = LPLayer(3,1)
    print(model)
    x = torch.rand(5,6,3)
    adj = torch.randint(0,2,(5,6,6)).float()
    #adj = torch.ones(5,6,6)
    #print(adj)
    
    y=model(x)
    #H = y.view(5,-1).sum(dim=1)
    #print(H.shape)
    #H_l=(H.tolist())
    #print(len(H_l))
    #print(H_l[0])