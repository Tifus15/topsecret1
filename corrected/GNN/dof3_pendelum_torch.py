import torch
import torch.nn as nn
from src.device_util import DEVICETensor
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
    if method == "dopri5":
        k1 = model.forward(0,y)
        k2 = model.forward(0,y + (dt/5)*k1)
        k3 = model.forward(0,y + (3*dt/40)*k1 + (9*dt/40)*k2)
        k4 = model.forward(0,y + (44*dt/45)*k1 - (56*dt/15)*k2+ (32*dt/9)*k3)
        k5 = model.forward(0,y + (19372*dt/6561)*k1 - (25360*dt/2187)*k2+ (64448*dt/6561)*k3-(212*dt/729)*k4)
        k6 = model.forward(0,y + (9017*dt/3168)*k1 - (355*dt/33)*k2+ (46732*dt/5247)*k3+(49*dt/176)*k4-(5103*dt/18656)*k5)
        k7 = model.forward(0,y + (35*dt/384)*k1 + (500*dt/1113)*k3+ (125*dt/192)*k4-(2187*dt/6484)*k5 + (11*dt/84)*k6)
        res1 = 5179*k1/57600 + 7571*k3/16695 + 393*k4/640 -92097*k5/339200 + 187*k6/2100 + k7/40
        res2 = 35*k1/384 + 500*k3/1113+125*k4/192 - 2187*k5/6484 +11*k6/84
        return (res1 + res2)/2
        
    else:
        return DEVICETensor()


class pendelum3dof(nn.Module):
    def __init__(self,m1,m2,m3,l1,l2,l3,g):
        super(pendelum3dof,self).__init__()
        self.m1 = m1
        self.l1 = l1
        self.m2 = m2
        self.l2 = l2
        self.m3 = m3
        self.l3 = l3
        self.g = g
        
    def cart_traj(self,theta_dtheta):
        x1 = self.l1*torch.sin(theta_dtheta[:,0:1])
        x2 = x1 + self.l2*torch.sin(theta_dtheta[:,1:2]) 
        x3 = x2 + self.l3*torch.sin(theta_dtheta[:,2:3])
        y1 = -self.l1*torch.cos(theta_dtheta[:,0:1])
        y2 = y1 - self.l2*torch.cos(theta_dtheta[:,1:2])
        y3 = y2 - self.l3*torch.cos(theta_dtheta[:,2:3])
        
        return torch.cat((x1,x2,x3,y1,y2,y3),dim=-1)
    
    def cart_der(self,theta_dtheta):
        dx1 = theta_dtheta[:,3:4]*self.l1*torch.cos(theta_dtheta[:,0:1])
        dy1 = theta_dtheta[:,3:4]*self.l1*torch.sin(theta_dtheta[:,0:1])
        dx2 = dx1 + theta_dtheta[:,4:5]*self.l2*torch.cos(theta_dtheta[:,1:2])
        dy2 = dy1 + theta_dtheta[:,4:5]*self.l2*torch.sin(theta_dtheta[:,1:2])
        dx3 = dx2 + theta_dtheta[:,5:6]*self.l3*torch.cos(theta_dtheta[:,2:3])
        dy3 = dy2 + theta_dtheta[:,5:6]*self.l3*torch.sin(theta_dtheta[:,2:3])
        return torch.cat((dx1,dx2,dx3,dy1,dy2,dy3),dim=-1)
    
    def cart_sq_der(self,theta_dtheta):
        vec = self.cart_der(theta_dtheta)
        vs1 = vec[:,0:1].pow(2)+ vec[:,3:4].pow(2)
        vs2 = vec[:,1:2].pow(2)+ vec[:,4:5].pow(2)
        vs3 = vec[:,2:3].pow(2)+ vec[:,5:6].pow(2)
        return torch.cat((vs1,vs2,vs3),dim=-1)

    def kin(self,theta_dtheta):
        vs = self.cart_sq_der(theta_dtheta)
        return 0.5* self.m1 * vs[0,0:1] + 0.5* self.m2 * vs[0,1:2] + 0.5* self.m3 * vs[0,2:3]

    def pot(self,theta_ptheta):
        traj = self.cart_traj(theta_ptheta)
        return self.m1*self.g*(traj[0,3:4]+self.l1) + self.m2*self.g*(traj[0,4:5]+self.l1+self.l2) + self.m3*self.g*(traj[0,5:6]+self.l1+self.l2+self.l3)

    def trans_matrix(self,theta_dtheta):
        Hess = hessian(self.kin,theta_dtheta,create_graph=True)
        Hess=Hess.reshape(6,6) 
        return Hess[3:6,3:6]
    
    def d_theta_2_p_theta(self,theta_dtheta):
        Hess = self.trans_matrix(theta_dtheta)
        theta_ptheta = theta_dtheta.clone()
        theta_ptheta[:,3:6] =  theta_dtheta[:,3:6] @ Hess 
        return theta_ptheta

    def p_theta_2_d_theta(self,theta_ptheta):
        Hess = self.trans_matrix(theta_ptheta)
        theta_dtheta = theta_ptheta.clone()
        theta_dtheta[:,3:6] = theta_ptheta[:,3:6] @ torch.linalg.inv(Hess)
        return theta_dtheta


    def H(self,theta_ptheta):
        K = self.kin(self.p_theta_2_d_theta(theta_ptheta))
        U = self.pot(theta_ptheta)
        return K+U

    def dH_dth_dpth(self,theta_ptheta):
        Ene = self.H(theta_ptheta)
        dEne= torch.autograd.grad(Ene,theta_ptheta)
        return dEne[0]
    
    def forward(self,t,y):
        dHdx = self.dH_dth_dpth(y.requires_grad_())
        dq = dHdx[:,3:6]
        dp = -dHdx[:,0:3]
        return torch.cat((dq,dp),dim=-1).detach() 
    
    
def dof3_hamiltonian_eval(trajectory):
    model = pendelum3dof(1,1,1,1,1,1,9,81)
    H = DEVICETensor(trajectory.shape[0],1)
    for i in range(trajectory.shape[0]):
        H[i,:] = model.H(trajectory[i,:,:])
    return H.detach()
def dof3_grads(model,traj):
    list = []
    for i in range(traj.shape[0]):
        x0 = traj[i,:,:].view(1,-1)
       #print(x0.shape)
        dx = model.forward(0,x0)
        list.append(dx.unsqueeze(0))
    return torch.cat((list),dim=0)        
    
def dof3_trajectory(model,t,y0,method="rk4"):
    # y0 -[1,6]
    out = odeint(model,y0,t,method=method)
    return out

def dof3_dataset(model,t,y0,method="rk4"):
    list=[]
    for i in tqdm(range(y0.shape[0])):
        x = dof3_trajectory(model,t,y0[i,:,:],method).unsqueeze(1)
        dx = dof3_grads(model,x).unsqueeze(1)
        #print(dx.shape)
        #print(x.shape)
        traj = torch.cat((x,dx),dim=-1)
        list.append(traj)
    return torch.cat((list),dim=1)

def dof3_symplectic_trajectory(model,t,y0,method="euler"):
    out = DEVICETensor(len(t),y0.shape[0],y0.shape[1])
    out[0,:,:] = y0
    for i in range(1,len(t)):
        #print(out[i-1,:,:])
        dt = t[i]-t[i-1]
        temp = out[i-1,:,:]
        if method =="euler":
            dp = model.forward(0,temp)[:,3:6]
            temp.requires_grad = False
            temp[:,3:6] = temp[:,3:6] + dt*dp
            dq = model.forward(0,temp)[:,0:3]
            temp.requires_grad = False
        
            temp[:,0:3] = temp[:,0:3] + dt*dq
        out[i,:,:] = temp
    return out.detach()
"""    
model = pendelum3dof(1,1,1,1,1,1,10)
theta_ptheta = DEVICETensor([[0.0,0.0,0.0,0.0,0.0,0.0]])
t = torch.linspace(0,1,100)
traj=dof3_trajectory(model,t,theta_ptheta,method="rk4")
#traj=dof3_symplectic_trajectory(model,t,theta_ptheta,method="euler")
print(dof3_hamiltonian_eval(model,traj))
#print(odeint(model,theta_ptheta,t,method="rk4"))
"""