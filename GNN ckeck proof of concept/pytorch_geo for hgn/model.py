from torch_geometric.nn import GATConv, Linear
from torch_geometric.utils import unbatch
import torch
import torch.nn as nn
class mlp(nn.Module):
    def __init__(self,acts, in_dim=2, out_dim=2, hidden=[0, 0]):
        super().__init__()
        layers = []
        layer_sizes = [in_dim] + hidden
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index-1], layer_sizes[layer_index]),
                       act_module(acts[layer_index-1])]
        layers += [nn.Linear(layer_sizes[-1], out_dim)]
        self.layers = nn.ModuleList(layers) # A module list registers a list of modules as submodules (e.g. for parameters)
        self.init_layers()
        self.config = {"act_fn": acts.__class__.__name__, "input_size": in_dim, "num_classes": out_dim, "hidden_sizes": hidden}

    def init_layers(self):
        for m in self.layers.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0.,std=0.1)
                nn.init.constant_(m.bias,val=0)
    
    
    def forward(self,t, x):
       #print("foward")
        for l in self.layers:
            #print(x.shape)
            #print(l)
            x = l(x)
        #print("end")
        #print(x.shape)
        return x

class Sin(nn.Module):
    def forward(self,x):
        return torch.sin(x)
    

        
def act_module(name):
    if name == "tanh":
        return nn.Tanh()
    elif name == "sin":
        return Sin()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "relu":
        return nn.ReLU()
    elif name == "softplus":
        return nn.Softplus()
    else:
        return nn.Identity()



class GATGNN(nn.Module):
    def __init__(self,edge_map,acts, in_dim=2, out_dim=2, hidden=[64, 64],bias=True):
        super().__init__()
        layers = []
        layer_sizes = [in_dim] + hidden
        for layer_index in range(1, len(layer_sizes)):
            ind = layer_sizes[layer_index-1]
            outd = layer_sizes[layer_index]
            layer = GATConv(ind,outd,bias=bias)
            layers += [layer,act_module(acts[layer_index-1])]
        ind = layer_sizes[-1]
        layer = GATConv(ind,out_dim,bias=bias)
        layers += [layer]
        self.layers = nn.ModuleList(layers) # A module list registers a list of modules as submodules (e.g. for parameters)
        self.config = {"act_fn": acts.__class__.__name__, "input_size": in_dim, "num_classes": out_dim, "hidden_sizes": hidden}
        self.edge_map = edge_map
    def change_graph(self,e_map):
        self.edge_map = e_map
    def forward(self,t, x):
        #print(x.is_leaf)
       #print("foward")
        for l in self.layers:
            #print(x.shape)
            #print(l.parameters())
            if isinstance(l,GATConv):
                x = l(x,self.edge_map)
            else:
                x=l(x)
        #print("end")
        #print(x.shape)
        return x
    
class HGN(nn.Module):
    def __init__(self,edge_map,acts, in_dim=2, h_dim=2, hidden=[64, 64],nodes=3,bias=True):
        super().__init__()
        self.base = GATGNN(edge_map,acts, in_dim, h_dim, hidden,bias)
        M = torch.eye(in_dim)
        M = torch.cat([M[in_dim//2:], -M[:in_dim//2]])
        self.J = M
        self.edge_map = edge_map
        self.nodes = nodes
        
    def change_graph(self,e_map):
        self.base.change_graph(e_map)
        self.edge_map = e_map
    def change_nodes(self,n):
        self.nodes = n # only if you retrain system with more nodes
        
    
    
    def forward(self,t,x):
        y = self.base(t,x)
        H = []
        N = int(y.shape[0]/self.nodes)
        #print(self.nodes)
       # print(y.shape)
        #print(N)
        for i in range(N):
            H.append(torch.sum(y[i*self.nodes:(i+1)*self.nodes,:]))
        #print("H size {}".format(len(H)))
        gr = torch.autograd.grad(H,x,create_graph=True)[0]
        out = gr @ self.J.transpose(0,1)
        return out

def rolloutdxHGN(model,x):
    l = []
    for i in range(x.shape[0]):
        temp = model(0,x[i,:,:]).unsqueeze(0)
        l.append(temp)
    out = torch.cat((l),dim=0)
    return out 

def rk4(x,dt,model):
   # print("x {}".format(x.shape))
    k1 = model(0,x)
   # print("k1 {}".format(k1.shape))
    k2 = model(0,x + dt*k1/2)
   # print("k2 {}".format(k2.shape))
    k3 = model(0,x + dt*k2/2)
   # print("k3 {}".format(k3.shape))
    k4 = model(0,x + dt*k3)
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
            xii = xi + dt * rk4(xi,dt,model)
        l.append(xii.unsqueeze(0).detach().requires_grad_())
        #print("l[{}] is leaf {}".format(i,l[i].is_leaf))
    #print("l[{}] is leaf {}".format(len(t)-1,l[len(t)-1].is_leaf))
    return torch.cat((l),dim=0)

if __name__ == "__main__":
    src = [0,0,1,1,2,2]
    dst = [1,2,0,2,0,1]
    
    edge_map = torch.tensor([src,dst],dtype=torch.long)
    model = HGN(edge_map,["tanh"],4,5,[4],nodes=3,bias=True)
    print(model)
    for name, param in model.named_parameters():
        print(name,param)
    
    x = torch.rand(6,4).requires_grad_()
    
    y = model(0,x)
    print(y.shape)


