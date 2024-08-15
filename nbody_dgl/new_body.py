import dgl 
import torch
import torch.nn as nn
#from torchdiffeq import odeint
from torch_symplectic_adjoint import odeint
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
        x = odeint(self.real_model,x0,t,method=method)
        return x

class G_Nbody(torch.nn.Module):
    def __init__(self,masses,N=7,G=0.01,eps=1e-6):
        super(G_Nbody,self).__init__()
        if masses != 0:
            self.N = len(masses)
            self.m = masses
        else:
            self.N = N
            self.m = 0
        self.G = G
        self.g = dgl.graph((src_list(self.N),dst_list(self.N)))
        self.eps = eps
        self.T=0
        self.U=0
        self.set_masses()
        # graph has "m" on every node!!!
        print("simulator of {} body problem".format(self.N))
        #print(graph.ndata["m"])
        
    def set_masses(self):
        if self.m != 0:
            self.g.ndata["m"] = torch.Tensor(self.m).reshape(self.N,1)
        else: 
            self.g.ndata["m"] = torch.ones(self.N,1)

    def split_qp(self,x):
        qp = torch.split(x,int(x.shape[-1]/2),dim=-1)
        self.g.ndata["q"] = qp[0]
        self.g.ndata["p"] = qp[1]
        
    def mail_for_dotq(self,edges):
        return {'pi': edges.src['p'],
                'mi': edges.src['m']}
    
    def make_dotq(self,nodes):
        mi = nodes.mailbox["mi"]
        pi = nodes.mailbox["pi"]
        out = pi/mi
        out=out[0,:,:]
        return {'dotq' : out}
        
    def make_K(self,nodes):
        mi = nodes.mailbox["mi"]
        pi = nodes.mailbox["pi"]
        #print(pi[0,:,:])
        #print(mi[0,:,:])
        M_inv = torch.linalg.inv(torch.diag(mi[0,:,:].squeeze()))
        p = torch.linalg.vector_norm(pi[0,:,:],dim=-1).unsqueeze(-1)
        #print(p.shape)
        #print(M_inv.shape)
        K = 0.5 * p.transpose(0,1) @ M_inv @ p
        self.T = K.squeeze()       
        return {'nope' : pi}
    
    def make_P(self,nodes):
        qi = nodes.mailbox["qi"]
        qj = nodes.mailbox["qj"]
        mi = nodes.mailbox["mi"]
        mj = nodes.mailbox["mj"]
        const = self.G*mi*mj
        #print(const.squeeze())
        sub = qj - qi
        euclid = torch.linalg.vector_norm(sub,dim=-1)+self.eps 
        #print(euclid)
        out = const.squeeze() / euclid
        out = torch.tril(out,diagonal=-1)
        self.U = -torch.sum(out.flatten())
        #print(self.U)
        return {'nope' : qi}
    
        
    def mail_for_dotp(self,edges):
        return {'qi': edges.src['q'], 'qj': edges.dst['q'],
                'mi': edges.src['m'], 'mj': edges.dst['m'],}
    def make_dotp(self,nodes):
        qi = nodes.mailbox["qi"]
        qj = nodes.mailbox["qj"]
        mi = nodes.mailbox["mi"]
        mj = nodes.mailbox["mj"]
        const = self.G*mi*mj
        #print(const.shape)
        sub = qj - qi
        euclid = torch.linalg.vector_norm(sub,dim=-1)+self.eps 
        denom = euclid**3
        nom = sub*const
        #print(nom.shape)
        #print(denom.shape)
        out = nom/denom.unsqueeze(-1)
        #print(out)
        #print(out.shape)
        #print(out[:,0,:])
        dotp = torch.sum(out,dim=1)
        #print(dp.shape)
        
        return {'dotp' : dotp}

    def forward(self,t,x):
        self.split_qp(x)
        self.g.update_all(self.mail_for_dotp,self.make_dotp)
        self.g.update_all(self.mail_for_dotq,self.make_dotq)
        dotq = self.g.ndata.pop("dotq")
        dotp = -self.g.ndata.pop("dotp")
        
        out = torch.cat((dotq,dotp),dim=-1)
        #print(out.shape)
        return out
    
    def K(self,x):
        self.split_qp(x)
        self.g.update_all(self.mail_for_dotq,self.make_K)
        return self.T
    
    def P(self,x):
        self.split_qp(x)
        self.g.update_all(self.mail_for_dotp,self.make_P)
        return self.U
    
    def H(self,x):
        self.split_qp(x)
        self.g.update_all(self.mail_for_dotq,self.make_K)
        self.g.update_all(self.mail_for_dotp,self.make_P)
        return (self.U + self.T).squeeze()
        
        
        
    
    
def make_dataset_master(batches=3,R=1.0,p_m=0.5,N=[6,7],dim=2,t=torch.linspace(0,5.11,512)):
    H_list = []
    x_list = []
    dx_list = []
        
    HN_list=[]
    xN_list=[]
    dxN_list=[]
    for i in tqdm(range(batches)):
        
        flag = False
        while not flag:
            print("try {}".format(N[0]))
            q = -R + torch.rand(N[0],dim)*2*R
            p = -p_m +  torch.rand(N[0],dim)*2*p_m
            
            x0N= torch.cat((q,p),dim=-1)
            model = G_Nbody(0,N[0])
            solver = BODY_solver(model)
            x = solver(t,x0N,"rk4")
            H, K, P = solver.H(x)
            H_mean = torch.abs(torch.mean(H))
            #my = torch.abs(H-H_mean)/torch.abs(H_mean)
            H_std = torch.std(H)
            
            crit = H_std/torch.abs(H_mean)
            #crit=torch.max(torch.abs(H-H_mean))/torch.abs(H_mean)
            print(crit)
            
            if crit>0.1:
                continue
            else:
                print("appending")
                flag = True
                dx = solver.dx(x)
                H_list.append(H.unsqueeze(0))
                x_list.append(x.unsqueeze(0))
                dx_list.append(dx.unsqueeze(0))
        
        while flag:
            print("try {}".format(N[1]))
            q_new = -R + torch.rand(int(N[1]-N[0]),dim)*2*R
            p_new = -p_m +  torch.rand(N[1]-N[0],dim)*2*p_m
            print(q_new.shape)
            qN = torch.cat((q,q_new),dim=0)
            pN = torch.cat((p,p_new),dim=0)
            x0N1 = torch.cat((qN,pN),dim=-1)
            model = G_Nbody(0,N[1])
            solver = BODY_solver(model)
            x = solver(t,x0N1,"rk4")
            H, K, P = solver.H(x)
            H_mean = torch.mean(H)
            H_std = torch.std(H)
            
            crit = H_std/torch.abs(H_mean)
            print(crit)
            
            if crit>0.1:
                continue
            else:
                print("appending")
                flag = False
                dx = solver.dx(x)
                HN_list.append(H.unsqueeze(0))
                xN_list.append(x.unsqueeze(0))
                dxN_list.append(dx.unsqueeze(0))
    print(len(H_list))
    HN = torch.cat((H_list),dim=0)
    print(HN.shape)
    HN1 = torch.cat((HN_list),dim=0)
    print(HN1.shape)
    xN = torch.cat((x_list),dim=0)
    print(xN.shape)
    xN1 = torch.cat((xN_list),dim=0)
    print(xN1.shape)    
    dxN = torch.cat((dx_list),dim=0)
    print(dxN.shape)
    dxN1 = torch.cat((dxN_list),dim=0)
    print(dxN1.shape)
    
    torch.save(HN,"H{}b{}.pt".format(N[0],batches))
    torch.save(HN1,"H{}b{}.pt".format(N[1],batches))
    torch.save(xN,"x{}b{}.pt".format(N[0],batches))
    torch.save(xN1,"x{}b{}.pt".format(N[1],batches))
    torch.save(dxN,"dx{}b{}.pt".format(N[0],batches))
    torch.save(dxN1,"dx{}b{}.pt".format(N[1],batches))
       





if __name__ == "__main__":
    #make_dataset_master(batches=1000)
    #test = torch.load("H6b3.pt")
    #print(test.shape)
    
    #src = src_list(7)
    #dst = dst_list(7)
    #print(src)
    #print(dst)
    #mod = G_Nbody(0, 6)
    #mod1 = G_Nbody([1,3,2])
    #print(mod.g.ndata["m"])
    #print(mod1.g.ndata["m"])
    R=1.0
    N=7
    dim=2
    q_test = -R + torch.rand(N,dim)*2*R
    
    """
    q_test[0,:]= torch.Tensor([-0.5,0.5])
    q_test[1,:]= torch.Tensor([0.5,0.5])
    q_test[2,:]= torch.Tensor([0.5,-0.5])
    q_test[3,:]= torch.Tensor([-0.5,-0.5])
    """
    p_test = -0.5 +  torch.rand(N,dim)
    #p_test-=torch.mean(p_test)
    m = 1*torch.ones(N).tolist()
    x0 = torch.cat((q_test,p_test),dim=-1)
    mod_test = G_Nbody(0,N)
    print("model created")
    print(mod_test.g.ndata["m"])
    
    print(mod_test.H(x0))
    
    solver = BODY_solver(mod_test)

t = torch.linspace(0,5.11,512)
#print(t)

x = solver(t,x0,"rk4")
H, K, P = solver.H(x)
print(H.shape)
print(K.shape)
print(P.shape)
plt.figure()
plt.plot(t,H.detach().numpy(),c="r")
plt.plot(t,K.detach().numpy(),c ="g")
plt.plot(t,P.detach().numpy(),c = "b")
plt.legend(["H","K","P"])
plt.show()
#print(x)
#print(H)
H_mean = torch.mean(H)
H_std = torch.std(H)
#print(H_mean)
#print(H_std)
print(torch.mean(H-H_mean))
print(H)


if dim==2:
    fig = plt.figure()
    for i in range(len(t)):
        plt.legend(["time {}".format(t[i]),"H: {}".format(H[i])])
        plt.cla()
        for j in range(x.shape[1]):
            point = x[i,j,0:3]
            plt.scatter(point[0],point[1])
            plt.xlim(-5*R,5*R)
            plt.ylim(-5*R,5*R)
        
        plt.pause(0.0001)
else:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for i in range(len(t)):
        ax.set_title("plot at {}".format(t[i]))
        plt.cla()
        for j in range(x.shape[1]):
            point = x[i,j,0:3]
            ax.scatter(point[0],point[1],point[2])
            ax.set_xlim(-5*R,5*R)
            ax.set_ylim(-5*R,5*R)
            ax.set_zlim(-5*R,5*R)
        plt.pause(0.0001)
   



