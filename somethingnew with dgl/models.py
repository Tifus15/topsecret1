import torch
import torch.nn as nn
from torch.nn import Linear, Sequential as Seq, Tanh, ReLU
import torch.nn.functional as F
import dgl.function as fn
from device_util import DEVICE

def roll(model,x0, t,method="euler"):
    ### batch : [ N F]
    list = []
    list.append(x0.view(x0.shape[0],x0.shape[1], 1))
    #model.change_graph(edges)
    if method == "rk4":
        for i in range(1,len(t)):
            K1= model(list[-1].squeeze())
            K2= model(K1+ (t[i]-t[i-1])*list[-1].squeeze()/2)
            K3= model(K2+ (t[i]-t[i-1])*list[-1].squeeze()/2)
            K4= model(K3+ (t[i]-t[i-1])*list[-1].squeeze())

            list.append(list[-1] + (t[i]-t[i-1])*(K1 + 2*K2 + 2*K3 + K4).view(x0.shape[0],x0.shape[1],1)/6)
    else:
        for i in range(1,len(t)):
            f = model(list[-1].squeeze())
            list.append(list[-1]+(t[i]-t[i-1])*f.view(x0.shape[0],x0.shape[1],1))
    return torch.cat((list),dim=-1).to(DEVICE)

def H_roll(model,traj):
    list=[]
    for i in range(traj.shape[-1]):
        list.append(model.get_H(traj[:,:,i]))
    return torch.Tensor(list)
        

class Energy_Layer(torch.nn.Module):
    def __init__(self,g,enc_dim, Hout,out_dim,init_w=0.1):
        super(Energy_Layer,self).__init__()
        self.indim=enc_dim
        
        self.Kfun = Seq(Linear(enc_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
        self.Ufun = Seq(Linear(enc_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
        self.encK = Linear(enc_dim,enc_dim)
        self.encP1 = Linear(enc_dim,enc_dim)
        self.encP2 = Linear(enc_dim,enc_dim)
        self.g = g
        #self.grad = Linear(Hout, out_dim)
        nn.init.normal_(self.Kfun[0].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Kfun[2].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Kfun[4].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[0].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[2].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[4].weight,mean=0.,std=init_w)
        
        nn.init.normal_(self.encK.weight,mean=0.,std=init_w)
        nn.init.normal_(self.encP1.weight,mean=0.,std=init_w)
        nn.init.normal_(self.encP2.weight,mean=0.,std=init_w)
        #nn.init.normal_(self.grad.weight,mean=0.,std=init_w)
    def change_graph(self,g):
        self.g = g
        
    def getK(self,x):
        self.g.ndata["h1"] = self.encK(x)
        self.g.update_all(fn.copy_u("h1", "m"), fn.sum("m", "h1"))
        self.g.ndata["E"] = self.Kfun(self.g.ndata["h1"])
        #print(K.shape)
        return self.g.ndata["E"]
    
    def getU(self,x):
        self.g.ndata["h2_1"] = self.encP1(x)
        self.g.ndata["h2_2"] = self.encP2(x)
        self.g.apply_edges(fn.u_add_v('h2_1', 'h2_2', 'out'))
        #print("edges out size : {}".format(self.g.edata["out"].shape))
        self.g.edata["E"] = self.Ufun(self.g.edata["out"])
        
        #print(U.shape)
        return self.g.edata["E"]
    
    def getH(self,x):
        _ = self.getK(x)
        _ = self.getU(x)
        K = self.g.ndata["E"] #* self.g.ndata["a"] # weighted K
        #print("K shape {}".format(self.g.ndata["E"].shape))
        P = self.g.edata["E"] * self.g.edata["e"] # weighted potential on whole graph
        #print("p shape {}".format(self.g.edata["E"].shape))
        self.g.update_all(fn.u_mul_e('E', 'E', 'm'), fn.sum('m', 'E_new'))
        return self.g.ndata["E_new"]

    
    def forward(self,x):
        H_layer = self.getH(x)
        #print(H_layer.shape)
        return torch.sum(H_layer)

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, apply_edges=True):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()
        self.apply_edges = apply_edges

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        
        return {"e": torch.sigmoid(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        #self.g.edata["a"] = alpha
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata["z"] = z
        # equation (2)
        if self.apply_edges:
            self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop("h")


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == "cat":
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))
        
        
class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(g, in_dim, hidden_dim)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATLayer(g, hidden_dim, out_dim)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h
    
class PortHNN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim,H_dim):
        super(PortHNN, self).__init__()
        self.g = g
        self.emb = GATLayer(self.g, in_dim, hidden_dim) # to get weights on g
        self.HNN = Energy_Layer(self.g, hidden_dim,H_dim, in_dim) # energy calculation layer)
        M = torch.eye(in_dim)
        M = torch.cat([M[in_dim//2:], -M[:in_dim//2]]).to(DEVICE)  
        self.J = M
        self.D = nn.Parameter(torch.rand(in_dim,in_dim))
        
    def get_H(self,x):
        h = self.emb(x)
        H_sc = self.HNN(h)
        return H_sc
    def dH(self,x):
        H = self.get_H(x)
        return torch.autograd.grad(H,x)[0]

    def forward(self, h):
        # GAT layer gives us edge attention and alphas which we could implement in  
        dHdx = self.dH(h)
        qp = dHdx @ self.J.transpose(0,1)
        Dx = dHdx @ self.D.transpose(0,1)
        return qp - Dx
    
    # we have ndata"x" and ndata"y"
    def calculate_loss(self,out,H= None,H_real = None,func = nn.MSELoss()):
        if H is None:
            loss = func(out[:,0,:],self.g.ndata["y"][:,0,:])+func(out[:,1,:],self.g.ndata["y"][:,1,:])  
        else:
            loss = func(out[:,0,:],self.g.ndata["y"][:,0,:])+func(out[:,1,:],self.g.ndata["y"][:,1,:])   + func(H,H_real)
        return loss
         
    
    



def load_cora_data():
    data = citegrh.load_cora()
    g = data[0]
    mask = torch.BoolTensor(g.ndata["train_mask"])
    return g, g.ndata["feat"], g.ndata["label"], mask



