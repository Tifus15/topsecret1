import dgl 
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import numpy as np

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


class BODY_solver(torch.nn.Module):
    def __init__(self,G_model):
        super(BODY_solver,self).__init__()
        self.real_model = G_model
        
    
    def H(self,x):
        H=[]
        for i in tqdm(range(x.shape[0])):
            H.append(self.real_model.H(x[i,:,:]).unsqueeze(0))
        return torch.cat(H,dim=0)
    
    def dx(self,x):
        dx = []
        for i in tqdm(range(x.shape[0])):
            dx.append(self.real_model(0,x[i,:,:]).unsqueeze(0))
        return torch.cat(dx,dim=0)
        
    def forward(self,t,x0,method):
        #print(t.shape)
        x = odeint(self.real_model,x0,t,method=method)
        return x
    
    
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
        
        
    def message_func(self, edges):
        return {'qi': edges.src['q'], 'qj': edges.dst['q'],
                'mi': edges.src['m'], 'mj': edges.dst['m'],
                "pi": edges.src['p']}
    
    def reduce_func_step(self, nodes):
        qi = nodes.mailbox['qi']
        qj = nodes.mailbox['qj']
        mi = nodes.mailbox['mi']
        mj = nodes.mailbox['mj']
        pi = nodes.mailbox['pi']
        #print("qi {}".format(qi.shape))
        #print("mi {}".format(mi))
        #print("mj {}".format(mj))
        #print(pi)
        dhdp = pi/mi
        dhdp= dhdp[0,:,:]
        #print(dhdp.shape)
        const = self.G *mi*mj
       # print(const)
        sub = qj-qi
        #print(sub.shape)
        denom = (torch.linalg.vector_norm(sub,dim=-1) + self.eps) ** 3
        dhdq = - torch.sum((const * sub)/denom.unsqueeze(-1),dim=0)
        return {'dq' : dhdp , 'dp' : -dhdq}
    
    def reduce_func_H(self, nodes):
        qi = nodes.mailbox['qi']
        qj = nodes.mailbox['qj']
        mi = nodes.mailbox['mi']
        mj = nodes.mailbox['mj']
        pi = nodes.mailbox['pi']
        #print("qi {}".format(qi.shape))
        #print("mi {}".format(mi))
        #print("mj {}".format(mj))
        #print(pi)
        T= (torch.linalg.vector_norm(pi,dim=-1)**2)/(2*mi)
        #print("T {}".format(T.shape))
        self.T=torch.sum(T[0,0,:])
        const = self.G *mi*mj
        #print(const.shape)
       # print(const)
        sub = qj-qi
        #print("sub:{}".format(sub.shape))
        denom = torch.linalg.vector_norm(sub,dim=-1) +self.eps
        #print("denom:{}".format(denom.shape))
        U = const.squeeze()/denom
        #print("U {}".format(U))
        U = -torch.tril(U,diagonal=-1)
        #print("U {}".format(U))
        self.U = torch.sum(U.flatten())
        return {'T' : torch.zeros(sub.shape) , 'U' : torch.zeros(sub.shape)}
    
    def H(self,h):
        #print("H")
        qp = torch.split(h,int(h.shape[-1]/2),dim=-1)
        self.g.ndata["q"] = qp[0]
        self.g.ndata["p"] = qp[1]
        self.g.update_all(self.message_func, self.reduce_func_H)

        
        return self.T +self.U
       
    def forward(self, t, h):
        #print("here")
        #self.zahler+=1
        #print(self.zahler)
        qp = torch.split(h,int(h.shape[-1]/2),dim=-1)
        self.g.ndata["q"] = qp[0]
        self.g.ndata["p"] = qp[1]
            
        self.g.update_all(self.message_func, self.reduce_func_step)
            
        dq = self.g.ndata.pop('dq')
        dp = self.g.ndata.pop('dp')
        #print(dq)
        #print(dp)
            
        return torch.cat((dq,dp),dim=-1)

def dataset_loader(first=["4_x.pt","4_dx.pt","4_h.pt"],second=["5_x.pt","5_dx.pt","5_h.pt"]):
    x1 = torch.load(first[0])
    dx1= torch.load(first[1])
    H1 = torch.load(first[2])
    x2 = torch.load(second[0])
    dx2= torch.load(second[1])
    H2 = torch.load(second[2])
    print(x1.shape)
    print(dx1.shape)
    print(H1.shape)
    print(x2.shape)
    print(dx2.shape)
    print(H2.shape)
            
            
    
if __name__ == "__main__":# 
    #dataset_loader()
    """
    #src =[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    #dst =[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3] 
    src = [0,0,0,1,1,1,2,2,2]
    dst = [0,1,2,0,1,2,0,1,2]   
    g = dgl.graph((src,dst))
    g.ndata["m"]=torch.ones(3,1)
    g.ndata["m"][0,0]=1
    rmod = G_Nbody(g,1)

    x0 = torch.Tensor(3,4)
    x0[0,:] = torch.Tensor([0,1,1,1])
    x0[1,:] = torch.Tensor([-1,0,2,1])
    x0[2,:] = torch.Tensor([1,-1,-1,0])

    x = rmod(0,x0)
    H = rmod.H(x0)
    print(x)
    print(H)
    q21 = x0[1,0:2]-x0[0,0:2]
    q31 = x0[2,0:2]-x0[0,0:2]
    q12 = x0[0,0:2]-x0[1,0:2]
    q32 = x0[2,0:2]-x0[1,0:2]

    dp1 = q21/torch.linalg.vector_norm(q21)**3 + q31/torch.linalg.vector_norm(q31)**3
    dp2 = q12/torch.linalg.vector_norm(q12)**3 + q32/torch.linalg.vector_norm(q32)**3

    print(dp1)
    print(dp2)    
    print(x[0:2,2:4]) 

    U=-(1/torch.linalg.vector_norm(q21) + 1/torch.linalg.vector_norm(q32) + 1/torch.linalg.vector_norm(q31))
    T =  torch.linalg.vector_norm(x0[0,2:4])**2/2 + torch.linalg.vector_norm(x0[1,2:4])**2/2 +torch.linalg.vector_norm(x0[2,2:4])**2/2
    print(H)
    print(T+U)
"""
        