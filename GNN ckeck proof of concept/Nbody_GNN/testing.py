import torch
from hnn_model import *
import dgl
"""
def make_grad(sc_val,x):
    H_l = torch.split(sc_val,1,dim=0)
    return torch.autograd.grad(H_l,x,retain_graph=True)[0]
fileH = "data/nbody_4_H.pt"
filet = "data/nbody_4_traj.pt"

x = torch.load(filet)
H = torch.load(fileH)
dt =0.01
T = 4
t = torch.linspace(0,4,401)
print(t[1]-t[0])

print(x.shape)
print(H.shape)
print(torch.std(H[:,0,0]))

src=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
dst=[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
g=dgl.graph((src,dst))

model = GNN_maker_HNN(g,6,128,1,["tanH","relu",""],type="GCN") # H(q,p) func
x0= x[0,0:2,:,:].transpose(0,1).requires_grad_()
x0_s1 = x[0,0,:,:].requires_grad_()
x0_s2 = x[0,1,:,:].requires_grad_()
H = model(0,x0)
H_s1 = model(0,x0_s1)
H_s2 = model(0,x0_s2)
print(H.shape)
print(H_s1)
print(H_s2)
dH = make_grad(H,x0)
dHs_1= make_grad(H_s1,x0_s1)
dHs_2= make_grad(H_s2,x0_s2)
print(dH.shape)
print(dHs_1.shape)
print(dHs_2.shape)
print(torch.sum(dH[:,0,:]-dHs_1.squeeze())) # it should b same and it is
print(torch.sum(dH[:,1,:]-dHs_2.squeeze())) # it should be same
"""

ten = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(ten.shape)
print(ten[2,1])
print(ten.flatten())
print(ten.reshape(4,3))




