import torch.nn as nn
import dgl
import torch
import dgl.function as fn
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
        #print("GAT g exchanged")
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
    
    def change_graph(self,g):
        #print("GNNlayer g exchanged")
        #print(g)
        self.g = g
    
    def forward(self,feat):
        with self.g.local_scope():
            #print(feat.shape)
            self.g.ndata["h"] = self.NN(feat)
            self.g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            #print(self.g.ndata["h"].shape)
            h = self.g.ndata["h"]
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
    
class GNN_maker_HNN(nn.Module):
    def __init__(self,g, in_dim,hid_dim, out_dim,acts,bias=True,type="GCN"):
        super().__init__()
        self.g = g
        self.g_hnn=g
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
    
    def change_graph(self,g):
        for module in self.net.modules():
            if isinstance(module,GATLayer) or isinstance(module,GNNlayer):
                module.change_graph(g)   
        self.g_hnn = g
    
    def reset_graph(self):
        for module in self.net.modules():
            if isinstance(module,GATLayer) or isinstance(module,GNNlayer):
                module.change_graph(self.g)
        self.g_hnn = self.g
    
    """   
    def H(self,y):
        return torch.sum(self.net(y.float()),dim=0)
    """
    def forward(self,x,list=False):
        H_feat = self.net(x.float())
        #print(H_feat.shape)
        self.g_hnn.ndata["temp"] = H_feat
        gs = dgl.unbatch(self.g_hnn)
        #print("from model")
        #print("unbatched {}".format(len(gs)))
        h_list = []
        for g in gs:
            h = g.ndata["temp"]
            #print("h {}".format(h.shape))
            h_sc = h.sum()
           # print(h_sc)
            h_list.append(h_sc.unsqueeze(0))
        out=torch.cat((h_list),dim=0).unsqueeze(0)
        #print(out)
        return out

        
            
            
    def H_rollout(self,x):
        h = []
        for i in range(x.shape[0]):
            h.append(self.forward(x[i,:,:]).unsqueeze(0))
        return torch.cat((h),dim=0)
    
    
    def dx_rollout(self,x):
        out_l = []
        N = self.g.num_nodes()
        for i in range(x.shape[0]):
            xi = x[i,:,:].clone().detach()
            #print(xi.shape)
            dx = torch.autograd.functional.jacobian(self.forward,xi)
            h_n = dx.shape[0]
            temp = []
            for j in range(h_n):
                #print("jac:{} {}".format(j,dx[j,N*j:N*(j+1),:].shape))
                temp.append(dx[j,N*j:N*(j+1),:])
            temp1 = torch.cat((temp),dim=0)
            out_l.append(temp1.unsqueeze(0))   
        out = torch.cat((out_l),dim=0)     
        return out
    """ 
    def dx_rollout(self,x):
        dx = []
        for i in range(x.shape[0]):
            xi = x[i,:,:].clone().requires_grad_()
            #print("dx roll x_i: {}".format(x_i.shape))
            h=self.forward(xi,list=True)
            #print("dx roll H: {}".format(h))
            dHdx = torch.autograd.grad(h,xi,retain_graph=True,create_graph=True)[0]
            #print("dx roll dHdx: {}".format(dHdx.shape))
            qp = torch.split(dHdx,int(x.shape[-1]/2),dim=-1)
            dx.append(torch.cat((qp[1].unsqueeze(0),-qp[0].unsqueeze(0)),dim=-1).float())
        return torch.cat((dx),dim=0)
    """
def rk4(x,dt,model):
   # print("x {}".format(x.shape))
    k1 = model(x)
   # print("k1 {}".format(k1.shape))
    k2 = model(x + dt*k1/2)
   # print("k2 {}".format(k2.shape))
    k3 = model(x + dt*k2/2)
   # print("k3 {}".format(k3.shape))
    k4 = model(x + dt*k3)
   # print("k4 {}".format(k4.shape))
    
    return (k1 + 2*k2 + 2*k3 + k4)/6
    

def rollout(x0,t,model, method = "rk4"):
    l = []
    l.append(x0.unsqueeze(0).detach().requires_grad_())
    for i in range(len(t)-1):
       # print("rollout {}".format(i))
        dt = t[i+1]-t[i]
        xi = l[-1].squeeze().detach().requires_grad_()
        if method == "rk4":
            xii = xi + dt * rk4(xi.unsqueeze(0),dt,model.dx_rollout)
        l.append(xii.requires_grad_())
        #l.append(xii.detach().requires_grad_())
        #print("l[{}] is leaf {}".format(i,l[i].is_leaf))
    #print("l[{}] is leaf {}".format(len(t)-1,l[len(t)-1].is_leaf))
    return torch.cat((l),dim=0)
            
    
"""    
def rollout_dx(model_hnn,x):
    dx_l = []
    for i in range(x.shape[0]):
        h = 
    
    def dx_rollout(self,x):
        dx = []
        for i in range(x.shape[0]):
            dx.append(self.forward(0,x[i,:,:]).unsqueeze(0))
        return torch.cat((dx),dim=0)

    def forward(self,t,y):
        sc_val = self.H(y)
        H_l = torch.split(sc_val,1,dim=0)
        dhdx = torch.autograd.grad(H_l,y,retain_graph=True)[0]
        qp = torch.split(dhdx,int(y.shape[-1]/2),dim=-1)
        return torch.cat((qp[1],-qp[0]),dim=-1).float()
"""
