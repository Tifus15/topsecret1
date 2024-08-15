import dgl 
import torch
import torch.nn as nn
from torchdiffeq import odeint
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def dst_list(nodes):

    base = list(np.arange(nodes))
    out=[]
    for i in range(nodes):
        out = out + base
    return out

def src_list(nodes):
    out=[]
    for i in range(nodes):
        out = out +list(np.zeros((nodes),dtype=int)+i)
    return out

class G_Nbody(torch.nn.Module):
    def __init__(self,graph,G,eps=1e-13):
        super(G_Nbody,self).__init__()
        self.N = graph.num_nodes()
        self.G = G
        self.g = graph
        self.eps = eps
        self.T=0
        self.U=0
        self.zahler=0
        # graph has "m" on every node!!!
        print("simulator of {} body problem".format(self.N))
        #print(graph.ndata["m"])
        
    def message_func_K(self, edges):
        #print("sending message")
        return {'mi': edges.src['m'],"pi": edges.src['p']}
        
    def create_kin(self, nodes):
        #print("create kin")
        mi = nodes.mailbox['mi']
        #print(mi)
        pi = nodes.mailbox['pi']
        #print("got stuff")
        T= (torch.linalg.vector_norm(pi,dim=-1)**2)/(2*mi)
        #print("got T")
        self.T = torch.sum(T[0,0,:])
        return {'T' : torch.zeros(pi.shape)}
        
    def message_func_P(self,edges):
        return {'qi': edges.src['q'], 'qj': edges.dst['q'],
                'mi': edges.src['m'], 'mj': edges.dst['m']}
        
    def create_pot(self, nodes):
        qi = nodes.mailbox['qi']
        qj = nodes.mailbox['qj']
        mi = nodes.mailbox['mi']
        mj = nodes.mailbox['mj']
        sub = qj - qi
        norm = torch.linalg.vector_norm(sub,dim=-1) + self.eps
        mu = mi*mj*self.G
        Us = mu.squeeze()/norm
        self.U = torch.sum(torch.tril(Us,diagonal=1).flatten())
        return {'P' : torch.zeros(qi.shape)}
    def H(self,x):
        K=self.K(x)
        P=self.P(x)
        
        H = K+P
        return H
    def K(self,x):
        qp = torch.split(x,int(x.shape[-1]/2),dim=-1)
        self.g.ndata["q"] = qp[0]
        self.g.ndata["p"] = qp[1]
        
        self.g.update_all(self.message_func_K, self.create_kin)

        return self.T
    def P(self,x):
        qp = torch.split(x,int(x.shape[-1]/2),dim=-1)
        self.g.ndata["q"] = qp[0]
        self.g.ndata["p"] = qp[1]
        

        self.g.update_all(self.message_func_P, self.create_pot)
        
        
        return self.U
    def forward(self,t,x):
        H = self.H(x)
        #print(H)
        
        dHdx = torch.autograd.grad(H,x)[0]
        
        dqdp = torch.split(dHdx,int(dHdx.shape[-1]/2),dim=-1)
        
        return torch.cat((dqdp[1],-dqdp[0]),dim=-1)

class BODY_solver(torch.nn.Module):
    def __init__(self,G_model):
        super(BODY_solver,self).__init__()
        self.real_model = G_model
        
    
    def H(self,x):
        H=[]
        K=[]
        P=[]
        for i in tqdm(range(x.shape[0])):
            H.append(self.real_model.H(x[i,:,:]).unsqueeze(0))
            K.append(self.real_model.K(x[i,:,:]).unsqueeze(0))
            P.append(self.real_model.P(x[i,:,:]).unsqueeze(0))
        return torch.cat(H,dim=0),torch.cat(K,dim=0),torch.cat(P,dim=0)
    
    def dx(self,x):
        dx = []
        for i in tqdm(range(x.shape[0])):
            dx.append(self.real_model(0,x[i,:,:]).unsqueeze(0))
        return torch.cat(dx,dim=0)
        
    def forward(self,t,x0,method):
        #print(t.shape)
        x = odeint(self.real_model,x0,t,method=method)
        return x
    
    
N=40
R=1.0
dst = dst_list(N)
src = src_list(N)
g = dgl.graph((src,dst))
g.ndata["m"] = torch.ones(N,1)
pos = torch.rand(N , 2)* R
vel = -0.1 + torch.rand(N , 2)*0.2
#vel -= torch.mean(vel)/torch.sum(g.ndata["m"])
x0 = torch.cat((pos,vel),dim=-1).requires_grad_()  

model =G_Nbody(g,1.0,eps=1e-10)
solver = BODY_solver(model)

t = torch.linspace(0,3,301)
#print(t)

x = solver(t,x0,"rk4")
H,K,P = solver.H(x)
plt.figure()
plt.plot(t,H.detach().numpy())
plt.plot(t,K.detach().numpy())
plt.plot(t,P.detach().numpy())
plt.legend(["H","K","P"])
plt.show()
#print(x)
#print(H)
H_mean = torch.mean(H)
H_std = torch.std(H)
#print(H_mean)
#print(H_std)
print("zero mean {}".format(torch.mean(H-H_mean)))
print(H)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


for i in range(len(t)):
    plt.cla()
    for j in range(x.shape[1]):
        point = x[i,j,0:3]
        ax.scatter(point[0].detach().numpy(),point[1].detach().numpy(),point[2].detach().numpy())
        ax.set_xlim(-2*R,2*R)
        ax.set_ylim(-2*R,2*R)
        ax.set_zlim(-2*R,2*R)
    plt.pause(0.0001)
        
        
        
        
        
        
        
        
        

