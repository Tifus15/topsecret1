import torch
import torch.nn as nn
import math
import src.device_util as device_util
from src.device_util import DEVICETensor
from tqdm import tqdm

from torchdiffeq import odeint_adjoint as odeint

class gradH_rconst(nn.Module):
    def __init__(self,m1,m2,G,r):
        super(gradH_rconst,self).__init__()
        self.A=torch.zeros(6,6).to(device_util.DEVICE)
        mu = G*m1*m2/(r**3)
        self.A[0,4] = m1
        self.A[1,5] = m1
        self.A[2,4] = -1/m2
        self.A[3,5] = -1/m2
        self.A[4,0] = -mu
        self.A[4,2] = mu
        self.A[5,1] = -mu
        self.A[5,3] = mu
    def forward(self,t,y):
        return torch.matmul(y,self.A.T)

    
        


"""class for creating twobody data(binary star movement)"""



class twobody:
    def __init__(self,m1,m2,G):
        self.m1 = m1 # mass 1
        self.m2 = m2 # mass 2
        self.G = G # gravitational constant
    

    """get period"""
    def get_T_const(self,r, r1):
        omega = math.sqrt(self.G*self.m2/r1)/r
        return 2 * math.pi/omega
    """get centrypetal velocity """
    def calculate_v(self,r,r1):
        return (self.m1/r)*torch.sqrt(self.G*r1*self.m2)
    """get period"""
    def getT(self,r1):
        r=r1*(1+self.m1/self.m2)
        return self.get_T_const(r,r1)

    """make initialisation points x_0"""
    def make_inits(self,samples, r1_range=[1,5]):
        r1 = r1_range[0]+ torch.rand(samples,)*(r1_range[1]-r1_range[0])
        r2 = self.m1/self.m2 * r1
        phis = torch.rand(samples,)*2*torch.pi
        r = r1+r2
        p = self.calculate_v(r,r1)
        q1x = r1*torch.cos(phis)
        q1y = r1*torch.sin(phis)
        q2x = r2* torch.cos(phis+torch.pi)
        q2y = r2 *torch.sin(phis+torch.pi)
        p1x = -p *torch.sin(phis)
        p1y = p * torch.cos(phis)
        p2x = - p1x 
        p2y = - p1y
        

        return torch.cat((q1x.reshape(-1,1),q1y.reshape(-1,1),
                          q2x.reshape(-1,1),q2y.reshape(-1,1),
                          p1x.reshape(-1,1),p1y.reshape(-1,1),
                          p2x.reshape(-1,1),p2y.reshape(-1,1)
                          ),dim=-1).unsqueeze(dim=1)
    """calcluate hamiltonian of the dataset"""
    def hamiltonian(self,dataset):
        # dataset type [len(t),batches,1,8], [len(t),1,8], [batches,1,8]
        if dataset.dim() > 3:
            dataset = dataset.squeeze()
        qp = torch.split(dataset,2,dim=-1)
        r = torch.norm(qp[0]-qp[1],dim=-1)
        T1 = torch.norm(qp[2],dim=-1)**2/(2*self.m1) 
        T2 = torch.norm(qp[3],dim=-1)**2/(2*self.m2) 
        U = - self.G*self.m1*self.m2/r
        H = T1 +T2 + U

        return H
    """make one randomised trajectory """
    def make_sample(self,points,r1_range=[1,5]):
        y0=self.make_inits(1, r1_range).reshape(1,-1)
        r = torch.norm(y0[0,0:2]-y0[0,2:4])
        r1 = torch.norm(y0[0,0:2])
        t = torch.linspace(0,self.get_T_const(r,r1),points).to(device_util.DEVICE)
        model= gradH_rconst(self.m1,self.m2,self.G,r).to(device_util.DEVICE)
        y = odeint(model,y0[:,0:6].to(device_util.DEVICE),t,method="rk4")
        y = torch.cat((y,-y[:,:,4:6]),dim=-1)
        return y,t
    """make whole dataset"""
    def make_dataset(self,points,samples,T=0,r1_range=[1,5]):
        dataset = DEVICETensor(points,samples,1,8)
        ddataset = DEVICETensor(points,samples,1,8)
        y0 = self.make_inits(samples,r1_range).reshape(samples,1,8)
        r = torch.norm(y0[:,0,0:2]-y0[:,0,2:4],dim=-1)
        r1_max = r1_range[1] 
        r_max =(1+self.m1/self.m2)*r1_max
        if T==0:
            T = self.get_T_const(r_max,r1_max) # max T
        t = torch.linspace(0,T,points).to(device_util.DEVICE)
        
        for i in tqdm(range(samples)):
            model= gradH_rconst(self.m1,self.m2,self.G,r[i]).to(device_util.DEVICE)
            temp= odeint(model,y0[i,:,0:6].to(device_util.DEVICE),t,method="rk4")
            dtemp = model(0,temp)
            dataset[:,i,:,:] = torch.cat((temp,-temp[:,:,4:6]),dim=-1)
            ddataset[:,i,:,:] = torch.cat((dtemp,-dtemp[:,:,4:6]),dim=-1)
        H = self.hamiltonian(dataset)    
        return dataset,ddataset, t ,H
            
        

        



if __name__ == "__main__":
    maker = twobody(1,2,1)
    a,da,t,H  = maker.make_dataset(256,10)
    H = maker.hamiltonian(a)
    print(a.shape)
    print(da.shape)
    print(H.shape)
    print(H)

    
    




    