import dgl
import torch
import torch.nn as nn
from dgl.nn import GraphConv
from torch.autograd.functional import hessian 
from functools import partial

    

    
class ConvGNN(nn.Module):
    def __init__(self,in_dim,hidden_dim):
        super(ConvGNN,self).__init__()
        self.layer1 =GraphConv(in_dim, hidden_dim, norm='both', weight=True, bias=True)
        self.act = nn.ReLU()
        self.layer2 =GraphConv(hidden_dim,in_dim,norm='both',weight=True,bias=True)
        
    def forward(self,g,x):
        #print("at Conv {}".format(x.shape))
        h = self.layer1(g,x)
        h = self.act(h)
        h1 = self.layer2(g,h)
        return h1+x
    
class ConvGNN_R(nn.Module):
    def __init__(self,in_dim,hidden_dim,out):
        super(ConvGNN,self).__init__()
        self.layer1 =GraphConv(in_dim, hidden_dim, norm='both', weight=True, bias=True)
        self.act = nn.ReLU()
        self.layer2 =GraphConv(hidden_dim,out,norm='both',weight=True,bias=True)
        
    def forward(self,g,x):
        h = self.layer1(g,x)
        h = self.act(h)
        h1 = self.layer2(g,h)
        return h1+x
        

# let assume that we got q and q_dot as always from robots but we will do port hamiltonian 
#
#
#   
class HNN(torch.nn.Module):
    def __init__(self,g_selfloops,g_edges,in_dim,hidden_dim):
        super(HNN,self).__init__()
        self.in_dim=in_dim
        self.no_loop= g_edges
        self.selfloop_graph = g_selfloops
        self.K_fun = ConvGNN(int(in_dim/2),hidden_dim)
        self.P_fun = ConvGNN(int(in_dim/2),hidden_dim)
        self.gravity = nn.Parameter(torch.ones(1,1))
    # v or p    
    def K(self,x):
        #print("at K")
        h = self.K_fun(self.no_loop,x)
        self.selfloop_graph.ndata["h"] = h
        self.selfloop_graph.ndata["x"] = x
        self.selfloop_graph.update_all(self.message_fK, self.reduce_funcK)
        
        K = self.selfloop_graph.ndata.pop('K').squeeze()
        K = 0.5*torch.sum(K)
        return K
    
    def message_fK(self,edges):
        return {'x': edges.src['x'], 'h': edges.src['h']}
            
    def reduce_funcK(self, nodes):
        #print(nodes.mailbox["x"].shape)
        #print(nodes.mailbox["h"].shape)
        
        h = nodes.mailbox["x"] @ nodes.mailbox["h"].transpose(1,2) 
        #print(h.shape)
        return {'K': h}  
     
    def message_M(self,edges):
        return {'t': edges.src['temp']}
    
    def reduce_M(self,nodes):
        M = nodes.mailbox['t']
        for i in range(nodes.mailbox['t'].shape[0]):
            m = torch.relu(torch.diag(nodes.mailbox['t'][i,0,:,:]))+0.0001
            M[i,0,:,:] = torch.diag(m)
        return{"M" : M}
    
    # M matrix needs to be constant in every point of time
    def get_mass(self,x,type="v"):
        M=hessian(self.K,x.requires_grad_())
        M_temp = torch.sum(M,dim=2)
        self.selfloop_graph.ndata["temp"] = M_temp
        self.selfloop_graph.update_all(self.message_M, self.reduce_M)
        
        M = self.selfloop_graph.ndata.pop('M').squeeze()
        if type == "v":
            self.no_loop.ndata["M"] = M
            self.selfloop_graph.ndata["M"] = M 
            return self.no_loop.ndata["M"]
        else:
            self.no_loop.ndata["M"] = 1/M
            self.selfloop_graph.ndata["M"] = 1/M 
            return self.no_loop.ndata["M"]
        
        
#########################################################################################
#### princip dq.T@ M @ dq or p.T @ M^-1 @ p
#### with double diff - M    or M^-1
#####################################################################         
    """
    def message_P(self,edges):
        return {'M': edges.src['M'],'e': edges.data['e']}
    """
    def euclid_mass(self,edges):
        euclid =torch.linalg.vector_norm(edges.src['h']-edges.dst['h'],dim=-1)
        masses = edges.src["M"] * edges.dst["M"]
        list =[]
        #for i in range(self.no_loop.num_nodes()):
        #    list.append(torch.sum(torch.diag(masses[i,:,:])).reshape(1,1,1))
        #masses = torch.cat((list),dim=0)
        #print("dist {}".format(euclid.shape))
        #print("mass {}".format(masses.shape))
        return {'P' : self.gravity*masses/euclid.reshape(-1,1,1)}
    
    """
    def reduce_P(self,nodes):
        M = nodes.mailbox['M'] @ nodes.mailbox['M']
        print(M.shape)
        euclid = nodes.mailbox['e'] * self.gravity
        print(euclid.shape)
        P=[]
        for i in range(self.no_loop.num_nodes()):
            P.append(M[i,0,:,:]*euclid[i])
        print(P)
        pt = torch.sum(torch.cat(P,dim=-1))
        return{"P" : pt}
    """
    def P(self,x):
        #print("in P")
        h = self.P_fun(self.no_loop,x)
        self.no_loop.ndata["h"] = h
        self.no_loop.ndata["x"] = x
        self.no_loop.apply_edges(self.euclid_mass)
        #self.no_loop.update_all(self.message_P, self.reduce_P)
        
        P = self.no_loop.edata['P']
        P = 0.5*torch.sum(P)
        return P
    # p need to go here
    def dHdp(self,x):
        self.selfloop_graph.ndata["x"] = x
        self.selfloop_graph.update_all(self.message_HP, self.reduce_HP)
        res = self.selfloop_graph.ndata.pop('res').squeeze()
        #print("dHdp {}".format(res.shape))
        return res
    
    def message_HP(self,edges):
        return {'t': edges.src['x'], 'M': edges.src["M"]}
    
    def reduce_HP(self,nodes):
        x = nodes.mailbox["t"]
        M = nodes.mailbox["M"].squeeze()
        #print("x {}".format(x.shape))
        #print("M {}".format(M.shape))
        res = x @ torch.linalg.inv(M) #p/M = v
        #print("res {}".format(res.shape))
        return {"res" : res}
    
    
    
    
    
    def dHdq(self,x):
        Pt = self.P(x)
        out = torch.autograd.grad(Pt,x)[0]
        #print(out.shape)
        return out
    def forward(self,q,p):
        
        dHdQ = self.dHdq(q)
        dHdP = self.dHdp(p)
        out = torch.cat((dHdQ,dHdP),dim=-1)
        return out
    
    def dHdx(self,x):
        q = x[:,0:int(self.in_dim/2)]
        p = x[:,int(self.in_dim/2):]
        dH =self.forward(q,p)
        return dH @ self.J().transpose(0,1)
    
    
    def J(self):
        M = torch.eye(self.in_dim)
        M = torch.cat([M[self.in_dim//2:], -M[:self.in_dim//2]])
        return M
         
class PortHNN(nn.Module):
    def __init__(self,g_selfloops,g_edges,in_dim,hidden_dim):
        super(PortHNN,self).__init__()
        self.in_dim = in_dim
        self.R_fun = ConvGNN_R(int(in_dim/2),hidden_dim,1)
        self.HNN = HNN(g_selfloops,g_edges,in_dim,hidden_dim)
        
    def J(self):
        M = torch.eye(self.in_dim)
        M = torch.cat([M[self.in_dim//2:], -M[:self.in_dim//2]])
        return M
    
    def R(self,x):
        h = self.R_fun(x)
        R = torch.diag(h)
        out = R
    
    def forward(self,x):
        dqp = self.HNN(x[:,0:int(self.in_dim/2)],x[:,int(self.in_dim/2):])
        D = self.R(x[:,0:int(self.in_dim/2)])
        J = self.J()
        vp = dqp @ J.transpose(0,1)
        v = vp[:,0:int(self.in_dim/2)]
        p = vp[:,int(self.in_dim/2):] - dqp[:,int(self.in_dim/2):] @ D.transpose(0,1) 
        out = torch.cat((v,p),dim=-1)
        return out
    
def rk4_step(func,dt,x0):
    K1 = func(x0)
    K2 = func(x0 + (dt/2) * K1)
    K3 = func(x0 + (dt/2) * K2)
    K4 = func(x0 + dt* K3)
    
    return x0 + 1/6*dt*(K1 + 2*K2 + 2*K3 + K4)

def rollout_HNN(HNN_model, start_x,t):
    # at first set mass
    hdim = int(start_x.shape[-1]/2)
    M=HNN_model.get_mass(start_x[hdim:])
    #print(M)
    out = []
    out.append(start_x)
    for i in range(i,t.shape[0]):
        out.append(rk4_step(HNN_model.dHdx,t[i]-t[i-1],out[-1]).unsqueeze(0))
    return out.cat((out),dim=0)


        
        
       
        



if __name__ == "__main__":
    
    print(range(3))
    import dgl.data
    import matplotlib.pyplot as plt
    import networkx as nx
    g = dgl.graph(([0,1,2],
                   [1,2,0]))
    
    model = HNN(g,g,6,100)
    x = torch.rand(3,6)
    print(x)
    model.get_mass(x[:,3:6].requires_grad_())
    print(model.no_loop.ndata["M"])
    """
    for i in range(g.num_nodes()):
        list=[]
        list.append(torch.autograd.functional.hessian(K_sc[i],x[i,0:2].requires_grad_()))
        
    print(list)
    """
    y=model.P(x[:,0:3])
    print(y)
    d = model(x[:,0:3].requires_grad_(),x[:,3:6].requires_grad_())
    print(d)
    """
    test_mlp = mlp(2,100,2,1,acts=["tanh",""])
    print(test_mlp)
    g = dgl.graph(([0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]))
    sub = dgl.node_subgraph(g,[0])
    for i in range(1,g.num_nodes()):
        temp = dgl.node_subgraph(g,[i])
        sub.add_nodes(i)
    
    print(g)
    print(sub)
    options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
    }
    G = dgl.to_networkx(g)
    plt.figure(figsize=[15,7])
    nx.draw(G, **options)
    G = dgl.to_networkx(sub)
    plt.figure(figsize=[15,7])
    nx.draw(G, **options)
    plt.show()
    """