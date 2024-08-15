import torch
import torch.nn as nn
from device_util import DEVICETensor
from torch.autograd.functional import hessian
from torch.autograd import grad
from time import time
from tqdm import tqdm
from torchdiffeq import odeint
def solver_fields(y,model, dt, method):
    if method == "euler": 
        return model.forward(0,y)
    if method == "rk4":
        k1 = model.forward(0,y)
        k2 = model.forward(0,y + (dt/2)*k1)
        k3 = model.forward(0,y + (dt/2)*k2)
        k4 = model.forward(0,y + dt*k3)
        return (k1+2*k2+2*k3+k4)/6
    else:
        return DEVICETensor()

class pendelum2dof(nn.Module):
    def __init__(self,m1,m2,l1,l2,g):
        super(pendelum2dof,self).__init__()
        self.m1 = m1
        self.l1 = l1
        self.m2 = m2
        self.l2 = l2
        self.g = g
        
    def cart_traj(self,theta_dtheta):
        x1 = self.l1*torch.sin(theta_dtheta[:,0:1])
        x2 = x1 + self.l2*torch.sin(theta_dtheta[:,1:2]) 
        y1 = -self.l1*torch.cos(theta_dtheta[:,0:1])
        y2 = y1 - self.l2*torch.cos(theta_dtheta[:,1:2])
        return torch.cat((x1,x2,y1,y2),dim=-1)
    
    def cart_der(self,theta_dtheta):
        dx1 = theta_dtheta[:,2:3]*self.l1*torch.cos(theta_dtheta[:,0:1])
        dy1 = theta_dtheta[:,2:3]*self.l1*torch.sin(theta_dtheta[:,0:1])
        dx2 = dx1 + theta_dtheta[:,3:4]*self.l2*torch.cos(theta_dtheta[:,1:2])
        dy2 = dy1 + theta_dtheta[:,3:4]*self.l2*torch.sin(theta_dtheta[:,1:2])
        return torch.cat((dx1,dx2,dy1,dy2),dim=-1)
    
    def cart_sq_der(self,theta_dtheta):
        vec = self.cart_der(theta_dtheta)
        vs1 = vec[:,0:1].pow(2)+ vec[:,2:3].pow(2)
        vs2 = vec[:,1:2].pow(2)+ vec[:,3:4].pow(2)
        return torch.cat((vs1,vs2),dim=-1)

    def kin(self,theta_dtheta):
        vs = self.cart_sq_der(theta_dtheta)
        return 0.5* self.m1 * vs[0,0:1] + 0.5* self.m2 * vs[0,1:2]

    def pot(self,theta_ptheta):
        traj = self.cart_traj(theta_ptheta)
        return self.m1*self.g*(traj[0,2:3]+self.l1) + self.m2*self.g*(traj[0,3:4]+self.l1+self.l2)

    def trans_matrix(self,theta_dtheta):
        Hess = hessian(self.kin,theta_dtheta,create_graph=True)
        Hess=Hess.reshape(4,4) 
        return Hess[2:4,2:4]
    
    def d_theta_2_p_theta(self,theta_dtheta):
        Hess = self.trans_matrix(theta_dtheta)
        theta_ptheta = theta_dtheta.clone()
        theta_ptheta[:,2:4] =  theta_dtheta[:,2:4] @ Hess 
        return theta_ptheta

    def p_theta_2_d_theta(self,theta_ptheta):
        Hess = self.trans_matrix(theta_ptheta)
        theta_dtheta = theta_ptheta.clone()
        theta_dtheta[:,2:4] = theta_ptheta[:,2:4] @ torch.linalg.inv(Hess)
        return theta_dtheta


    def H(self,theta_ptheta):
        K = self.kin(self.p_theta_2_d_theta(theta_ptheta))
        U = self.pot(theta_ptheta)
        return K+U

    def dH_dth_dpth(self,theta_ptheta):
        Ene = self.H(theta_ptheta)
        dEne= torch.autograd.grad(Ene,theta_ptheta)
        return dEne[0].detach() 
    
    def forward(self,t,y):
        dHdx = self.dH_dth_dpth(y.requires_grad_())
        dq = dHdx[:,2:4]
        dp = -dHdx[:,0:2]
        return torch.cat((dq,dp),dim=-1).detach() 
    
    
def dof2_hamiltonian_eval(model,trajectory):
    H = DEVICETensor(trajectory.shape[0],1)
    for i in range(trajectory.shape[0]):
        H[i,:] = model.H(trajectory[i,:,:])
    return H.detach()

def dof2_grads(model,traj):
    list = []
    for i in range(traj.shape[0]):
        x0 = traj[i,:,:].view(1,-1)
       #print(x0.shape)
        dx = model.forward(0,x0)
        list.append(dx.unsqueeze(0))
    return torch.cat((list),dim=0)  
    
def dof2_trajectory(model,t,y0,method="rk4"):
    # y0 -[1,4]
    out = odeint(model,y0,t,method=method)
    return out
def dof2_dataset(model,t,y0,method="rk4"):
    list=[]
    for i in tqdm(range(y0.shape[0])):
        x = dof2_trajectory(model,t,y0[i,:,:],method).unsqueeze(1)
        dx = dof2_grads(model,x).unsqueeze(1)
        #print(dx.shape)
        #print(x.shape)
        traj = torch.cat((x,dx),dim=-1)
        list.append(traj)
    
    return torch.cat((list),dim=1)
def dof2_symplectic_trajectory(model,t,y0,method="euler"):
    out = DEVICETensor(len(t),y0.shape[0],y0.shape[1])
    out[0,:,:] = y0
    for i in range(1,len(t)):
        #print(out[i-1,:,:])
        dt = t[i]-t[i-1]
        temp = out[i-1,:,:]
        if method =="euler":
            dp = model.forward(0,temp)[:,2:4]
            temp.requires_grad = False
            temp[:,2:4] = temp[:,2:4] + dt*dp
            dq = model.forward(0,temp)[:,0:2]
            temp.requires_grad = False
        
            temp[:,0:2] = temp[:,0:2] + dt*dq
        out[i,:,:] = temp
    return out.detach()
"""    
model = pendelum2dof(1,1,1,1,10)
theta_ptheta = DEVICETensor([[0.0,0.0,0.0,-0.0]])
t = torch.linspace(0,1,100)
traj=dof2_trajectory(model,t,theta_ptheta,method="rk4")
#traj=dof1_symplectic_trajectory(model,t,theta_ptheta,method="euler")
print(dof2_hamiltonian_eval(model,traj))
#print(odeint(model,theta_ptheta,t,method="rk4"))
"""

    