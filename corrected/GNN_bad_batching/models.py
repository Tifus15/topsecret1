import torch.nn as nn
import dgl
import torch
import dgl.function as fn

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()
    def change_graph(self,g):
        self.g = g
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

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
    
class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim):
        super(GAT, self).__init__()
        self.g = g
        self.c_in = in_dim
        self.layer1 = GATLayer(g, in_dim, hidden_dim)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATLayer(g, hidden_dim, hidden_dim)
        self.layer3 = GATLayer(g, hidden_dim, out_dim)
    def change_graph(self,g):
        self.layer1.change_graph(g)
        self.layer2.change_graph(g)
        self.layer3.change_graph(g)
    def forward(self, h):
        h = self.layer1(h)
        #print("first layer done")
        h = torch.tanh(h)
        h = self.layer2(h)
        #print("second layer done")
        h = F.elu(h)
        h = self.layer3(h)
        #print("third layer done")
        return h



class GNNlayer(nn.Module):
    def __init__(self,g, in_dim, out_dim,bias = True):
        super().__init__()
        self.NN = nn.Linear(in_dim,out_dim,bias)
        self.g = g
        nn.init.normal_(self.NN.weight,mean=0.,std=0.1)

    def change_graph(self,g):
        self.g = g

    def forward(self,feat):
        with self.g.local_scope():
            self.g.ndata["h"] = feat
            self.g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            #print(self.g.ndata["h"].shape)
            h = self.NN(self.g.ndata["h"])

            return h
        
        
class GNN(nn.Module):
    def __init__(self,g, in_dim,hid_dim, out_dim):
        super().__init__()
        self.g = g
        self.in_layer = GNNlayer(g,in_dim, hid_dim)
        self.tan = nn.Tanh()
        self.out_layer = GNNlayer(g,hid_dim, out_dim)

    def forward(self,feat):
        h = self.in_layer(feat)
  
        h = self.tan(h)
   
        h = self.out_layer(h)

        return h
    
    def change_graph(self,g):
        self.in_layer.change_graph(g)
        self.out_layer.change_graph(g)
        
        

class rollout_GNN_GRU(nn.Module):
    def __init__(self,g,in_dim,hid,gnn_hid,method="euler"):
        super().__init__()
        self.GRU = nn.GRU(in_dim,hid,1)
        self.GNN = GNN(g,hid,gnn_hid,in_dim)
        self.method = method
        
        #self.NN = nn.Linear(hid,in_dim)
        nn.init.normal_(self.GRU.weight_ih_l0,mean=0.,std=0.1)
        nn.init.normal_(self.GRU.weight_hh_l0,mean=0.,std=0.1)
        nn.init.constant_(self.GRU.bias_ih_l0,val=0)
        nn.init.constant_(self.GRU.bias_hh_l0,val=0)
    
    def change_graph(self,g):
        self.GNN.change_graph(g)
    
    def forward(self,t,x0):
        
        T = len(t)
        output = torch.zeros(T,x0.shape[0],x0.shape[1])
        output[0,:,:] = x0
        xi, hidden = self.GRU(x0.unsqueeze(dim=0))
        #print("x0: {}".format(x0.shape))
        #print("xi: {}".format(xi.unsqueeze(dim=0).shape))
        xii = self.GNN(xi.squeeze())
        #print("xii: {}".format(xii.shape))
        #print("hidden: {}".format(hidden.shape))
        #print(xii.shape)
        if self.method == "euler":
            dt = t[1]-t[0]
            temp = x0 + dt*xii
            output[1,:,:] = temp
        for i in range(2,T):
            #print("xi: {}".format(xi.unsqueeze(dim=0).shape))
            xi, hidden = self.GRU(temp.unsqueeze(dim=0),hidden)
            #print("hidden: {}".format(hidden.shape))
            xii = self.GNN(xi.squeeze())
            #print("xii: {}".format(xii.shape))
            if self.method == "euler":
                dt = t[i]-t[i-1]
                temp = temp+ dt*xii
                output[i,:,:] = temp
        
        return output
    
class odeintGNN(nn.Module):
    def __init__(self,GNN):
        super().__init__()
        self.nn = GNN

    def change_graph(self,g):
        self.nn.change_graph(g)
    
    def forward(self,t,feat):
        return self.nn(feat.float())   
