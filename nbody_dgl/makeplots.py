import torch
from GRUGNN_model import *
import dgl
from torchdiffeq import odeint
import matplotlib.pyplot as plt


H7 = torch.load("H7b1000.pt")
x7 = torch.load("x7b1000.pt")
dx7 = torch.load("dx7b1000.pt")
H6 = torch.load("H6b1000.pt")
x6 = torch.load("x6b1000.pt")
dx6 = torch.load("dx6b1000.pt")

print(x6.shape)
print(dx6.shape)
print(H6.shape)
print(x7.shape)
print(dx7.shape)
print(H7.shape)
src = [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
dst = [ 1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 5, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4]
g6 = dgl.graph((src,dst))
src = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]
dst = [1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6, 0, 1, 3, 4, 5, 6, 0, 1, 2, 4, 5, 6, 0, 1, 2, 3, 5, 6, 0, 1, 2, 3, 4, 6, 0, 1, 2, 3, 4, 5]

g7 = dgl.graph((src,dst))


model_first = rollout_GNN_GRU(g6,4,128,128,acts=["tanh",""],type="GCN") # 6 after training
model_first.load_state_dict(torch.load("server_6body.pth"))

model_second = rollout_GNN_GRU(g7,4,128,128,acts=["tanh",""],type="GCN") # 7 before training
model_second.load_state_dict(torch.load("server_6body.pth"))

model_third = rollout_GNN_GRU(g7,4,128,128,acts=["tanh",""],type="GCN") # 7 after training
model_third.load_state_dict(torch.load("server_7body_100.pth"))

print(model_first)
print(model_second)

xe6 = x6[-1,:,:,:]
xe7 = x7[-1,:,:,:]

x6_max = torch.max(xe6[:,:,0].flatten())
x6_min = torch.max(xe6[:,:,0].flatten())
y6_max = torch.max(xe6[:,:,1].flatten())
y6_min = torch.max(xe6[:,:,1].flatten())
x7_max = torch.max(xe7[:,:,0].flatten())
x7_min = torch.max(xe7[:,:,0].flatten())
y7_max = torch.max(xe7[:,:,1].flatten())
y7_min = torch.max(xe7[:,:,1].flatten())

dxe6 = dx6[-1,:,:,:]
dxe7 = dx7[-1,:,:,:]

print(xe6.shape)
t = torch.linspace(0,5.11,512)[0:32]
print(xe6[0,:,:].shape)
mod_x6 = model_first(t,xe6[0,:,:]) 
mod_x7 = model_second(t,xe7[0,:,:]) 
mod_x7_2= model_third(t,xe7[0,:,:]) 

fig, axs = plt.subplots(2,3)
for j in range(len(t)):
    axs[0,0].clear()
    axs[0,1].clear()
    axs[1,0].clear()
    axs[1,1].clear()
    axs[1,2].clear()
    for i in range(6):
        axs[0,0].scatter(xe6[j,i,0].detach().numpy(),xe6[j,i,1].detach().numpy())
        axs[0,1].scatter(mod_x6[j,i,0].detach().numpy(),mod_x6[j,i,1].detach().numpy())
    for i in range(7): 
        axs[1,0].scatter(xe7[j,i,0].detach().numpy(),xe7[j,i,1].detach().numpy())
        axs[1,1].scatter(mod_x7[j,i,0].detach().numpy(),mod_x7[j,i,1].detach().numpy())
        axs[1,2].scatter(mod_x7_2[j,i,0].detach().numpy(),mod_x7_2[j,i,1].detach().numpy())
    axs[0,0].set_title("GROUND TRUTH 6 bodies")
    axs[0,1].set_title("AFTER TRAINING 6 bodies")
    axs[1,0].set_title("GROUND TRUTH 7 bodies")
    axs[1,1].set_title("BEFORE TRAINING 7 bodies")
    axs[1,2].set_title("AFTER TRAINING 7 bodies")
    
    plt.pause(0.05)



plt.show()





