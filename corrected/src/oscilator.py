import torch
import torch.nn as nn
import math
import src.device_util as device_util
import matplotlib.pyplot as plt



from torchdiffeq import odeint_adjoint as odeint

class oscigradH(nn.Module):
    def __init__(self,k,m):
        super(oscigradH,self).__init__()
        self.A=torch.zeros(2,2).to(device_util.DEVICE)
        self.A[0,1]=1/m
        self.A[1,0]=-k
    def forward(self,t,y):
        return torch.matmul(y,self.A.T)
    
class oscilator:
    def __init__(self,m,k):
        self.k = k
        self.m = m
        self.F = oscigradH(k,m).to(device_util.DEVICE)


    def make_inits(self,samples,H_span=[1,5]):
        Hs = H_span[0]*torch.rand(samples,)*(H_span[1]-H_span[0])
        a = torch.sqrt(2*self.m*Hs)
        b = torch.sqrt(2*Hs/self.k)
        phis = torch.rand(samples,)*2*torch.pi
        
        p = a * torch.cos(phis)
        p = torch.unsqueeze(p,dim=-1)
   
        x = b * torch.sin(phis)
        x = torch.unsqueeze(x,dim=-1)

        return torch.cat((x,p),dim=-1)
    
    def hamiltonian(self,dataset):
        # dataset type [len(t),batches,1,2], [len(t),1,2], [batches,1,2]
        if dataset.dim() > 3:
            dataset = dataset.squeeze()
        if dataset.dim() < 3:
            dataset = torch.unsqueeze(dataset,dim=1)
        x = dataset[:,:,0]
        p = dataset[:,:,1]

        T = p.square()/(2*self.m)
        U = self.k*x.square()/2

        H = T+U
        return H


    def make_one(self,points, H_span=[1,5]):
        omega = math.sqrt(self.k/self.m)
        T = 2*math.pi/omega

        t = torch.linspace(0,T,points).to(device_util.DEVICE)
        y=self.make_inits(1,H_span=H_span).to(device_util.DEVICE)
        data = odeint(self.F,y[0,:],t,method="rk4")
        
        return data.unsqueeze(dim=1), t
    
    def make_dataset(self,points,samples,H_span=[1,5]):
        omega = math.sqrt(self.k/self.m)
        T = 2*math.pi/omega
        t = torch.linspace(0,T,points).to(device_util.DEVICE)
        y=self.make_inits(samples,H_span=H_span).to(device_util.DEVICE)
        y=torch.unsqueeze(y,dim=1)
        #print(y.shape)
        data = odeint(self.F,y,t,method="rk4")
        ddata = self.F(0,data)
        H= self.hamiltonian(data)

        return data,ddata, t, H

    

if __name__ == "__main__":
    creator = oscilator(4,0.5)
    data = creator.make_dataset(100,10)
    print(creator.hamiltonian(data)[:,0])



    

    
