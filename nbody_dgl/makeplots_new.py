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


model= rollout_GNN_GRU(g6,4,128,128,acts=["tanh",""],type="GCN") # 6 after training
model.load_state_dict(torch.load("server_6body.pth"))





print(model)

xe6 = x6[-1,:,:,:]
xe7 = x7[-1,:,:,:]

x6_max = torch.max(xe6[:,:,0].flatten()).numpy()
x6_min = torch.max(xe6[:,:,0].flatten()).numpy()
y6_max = torch.max(xe6[:,:,1].flatten()).numpy()
y6_min = torch.max(xe6[:,:,1].flatten()).numpy()
x7_max = torch.max(xe7[:,:,0].flatten()).numpy()
x7_min = torch.max(xe7[:,:,0].flatten()).numpy()
y7_max = torch.max(xe7[:,:,1].flatten()).numpy()
y7_min = torch.max(xe7[:,:,1].flatten()).numpy()

dxe6 = dx6[-1,:,:,:]
dxe7 = dx7[-1,:,:,:]

print(x6_max)
t = torch.linspace(0,5.11,512)[0:256]
plt.plot(t,H6[-1,0:256])
plt.show()
fig = plt.figure()
plt.plot(t,H7[-1,0:256])
plt.show()



print(xe6[0,:,:].shape)
mod_x6 = model(t,xe6[0,:,:])
model.change_graph(g7) 
mod_x7 = model(t,xe7[0,:,:]) 


fig, axs = plt.subplots(2,2)
for j in range(len(t)):
    axs[0,0].clear()
    axs[0,1].clear()
    axs[1,0].clear()
    axs[1,1].clear()
    axs[0,0].set_xlim(-2,2)
    axs[0,0].set_ylim(-2,2)
    axs[0,1].set_xlim(-2,2)
    axs[0,1].set_ylim(-2,2)
    axs[1,0].set_xlim(-2,2)
    axs[1,0].set_ylim(-2,2)
    axs[1,1].set_xlim(-2,2)
    axs[1,1].set_ylim(-2,2)
    for i in range(6):
        axs[0,0].scatter(xe6[j,i,0].detach().numpy(),xe6[j,i,1].detach().numpy())
        axs[0,1].scatter(mod_x6[j,i,0].detach().numpy(),mod_x6[j,i,1].detach().numpy())
    for i in range(7): 
        axs[1,0].scatter(xe7[j,i,0].detach().numpy(),xe7[j,i,1].detach().numpy())
        axs[1,1].scatter(mod_x7[j,i,0].detach().numpy(),mod_x7[j,i,1].detach().numpy())
        
    axs[0,0].set_title("GROUND TRUTH 6 bodies")
    axs[0,1].set_title("AFTER TRAINING 6 bodies")
    axs[1,0].set_title("GROUND TRUTH 7 bodies")
    axs[1,1].set_title("AFTER TRAINING & bodies - 7 bodies")
   
    
    plt.pause(0.05)



plt.show()





