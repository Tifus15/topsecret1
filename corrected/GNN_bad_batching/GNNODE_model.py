import torch.nn as nn
import dgl
import torch
import dgl.function as fn
import device_util
import torch.nn.functional as F

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
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=-1)
        #print(z2.shape)
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

def function_act(name):
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sin":
        return Sin()
    if name == "softplus":
        return nn.Softplus()
    else:
        return nn.Identity()

"""sin activation function as torch module"""
    
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(1.0 * x)


class GNNlayer(nn.Module):
    def __init__(self,g, in_dim, out_dim,bias = True):
        super().__init__()
        self.NN = nn.Linear(in_dim,out_dim,bias)
        self.g = g
        nn.init.normal_(self.NN.weight,mean=0.,std=0.1)
        nn.init.constant_(self.NN.bias,val=0)

    def forward(self,feat):
        with self.g.local_scope():
            self.g.ndata["h"] = self.NN(feat)
            self.g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            #print(self.g.ndata["h"].shape)
            h = self.g.ndata["h"]

            return h
        
        
class GNN(nn.Module):
    def __init__(self,g, in_dim,hid_dim, out_dim):
        super().__init__()
        self.g = g
        self.layer1 = GNNlayer(g,in_dim, hid_dim)
        self.tan = nn.Tanh()
        self.layer2 = GNNlayer(g,hid_dim, hid_dim)
        self.relu = nn.ReLU()
        self.layer3 = GNNlayer(g,hid_dim, out_dim)

    def forward(self,t,feat):
        h = self.layer1(feat)
  
        h1 = self.tan(h)
   
        h = self.layer2(h1)
        
        h = self.relu(h)
        
        h = self.layer3(h+h1)
        
        h = h + feat

        return h
    
class GNN_maker(nn.Module):
    def __init__(self,g, in_dim,hid_dim, out_dim,acts,bias=True,type="GCN"):
        super().__init__()
        self.g = g
        modules = []
        if type == "GCN":
            modules.append(GNNlayer(g,in_dim,hid_dim))
            modules.append(function_act(acts[0]))
            for i in range(1,len(acts)-1):
                modules.append(GNNlayer(g,hid_dim,hid_dim,bias=bias))
                if acts[i] != "":
                    modules.append(function_act(acts[i]))
            modules.append(GNNlayer(g,hid_dim,out_dim))
        elif type == "GAT":
            modules.append(GATLayer(g,in_dim,hid_dim))
            modules.append(function_act(acts[0]))
            for i in range(1,len(acts)-1):
                modules.append(GATLayer(g,hid_dim,hid_dim,bias=bias))
                if acts[i] != "":
                    modules.append(function_act(acts[i]))
            modules.append(GATLayer(g,hid_dim,out_dim))
            
        self.net = nn.Sequential(*modules)
        if type =="REC":
            self.net = GNN(g,in_dim,hid_dim,out_dim)
        
    def forward(self,t,y):
        return self.net(y.float())

class GNN_rollout(nn.Module):
    def __init__(self,g,input_dim,hidden_dim,output_dim,acts,bias=True,type="GCN"):
        super(GNN_rollout,self).__init__()
        torch.autograd.set_detect_anomaly(True)
        GNN = GNN_maker(g,input_dim,hidden_dim,output_dim,acts,bias,type=type).to(device_util.DEVICE)
        self.net = GNN.net
        self.g =g

    def forward(self,t,y):
        if y.dim() == 3:#batch, more dim dataset [nodes,batches,feat]
           
            self.trajectory = torch.Tensor(self.g.num_nodes(),y.shape[1],len(t),y.shape[-1]).to(device_util.DEVICE)
            
            self.trajectory[:,:,0,:] = y.clone()
        else: # singledataset
           
            self.trajectory = device_util.DEVICETensor(self.g.num_nodes(),len(t),y.shape[-1]).to(device_util.DEVICE)
    
            self.trajectory[:,0,:] = y.clone()
        for i in range(len(t)-1):
            
            if y.dim() == 3:
                ynow= self.trajectory[:,:,i,:].clone()
                self.trajectory[:,:,i+1,:] = self.net(ynow.float())
            else:
                ynow = self.trajectory[:,i,:].clone()
                self.trajectory[:,i+1,:] = self.net(ynow.float())
        return self.trajectory #[nodes,batch,time,feat] or [nodes,time,feat]
