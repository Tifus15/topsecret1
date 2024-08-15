import torch 
from HGN import HNN_maker
from utils import *
import dgl
import os
from torchdiffeq import odeint
import matplotlib.pyplot as plt 
g1 = dgl.graph((src_list(1),dst_list(1))) 


model1 = HNN_maker(g1,2,256,32,type="GAT")
if os.path.isfile("server_1dof.pth"):
    print("loading model")
    model = load_model(model1,"server_1dof.pth")


g2 = dgl.graph((src_list(2),dst_list(2))) 


model2 = HNN_maker(g2,2,256,32,type="GAT")
if os.path.isfile("server_2dof.pth"):
    print("loading model")
    model = load_model(model1,"server_2dof.pth")
    
    
g3 = dgl.graph((src_list(3),dst_list(3))) 


model3 = HNN_maker(g3,2,256,32,type="GAT")
if os.path.isfile("server_3dof.pth"):
    print("loading model")
    model = load_model(model1,"server_3dof.pth")
    
    
g4 = dgl.graph((src_list(4),dst_list(4))) 
t = torch.linspace(0,1.27,128)

model4 = HNN_maker(g4,2,256,32,type="GAT")
if os.path.isfile("server_4dof.pth"):
    print("loading model")
    model = load_model(model1,"server_4dof.pth")

data1 = torch.load("eval_dof1.pt").requires_grad_(False)
data2 = torch.load("eval_dof2.pt").requires_grad_(False)
data3 = torch.load("eval_dof3.pt").requires_grad_(False)
data4 = torch.load("eval_dof4.pt").requires_grad_(False)

print(data4.shape)

g_data1 = create_pend1dof_graph_snapshots([data1],src_list(1),dst_list(1))
g_data2 = create_pend2dof_graph_snapshots([data2],src_list(2),dst_list(2))
g_data3 = create_pend3dof_graph_snapshots([data3],src_list(3),dst_list(3))
g_data4 = create_pend4dof_graph_snapshots([data4],src_list(4),dst_list(4))

x1 = g_data1[0].ndata["xfeat"].transpose(0,1)
x2 = g_data2[0].ndata["xfeat"].transpose(0,1)
x3 = g_data3[0].ndata["xfeat"].transpose(0,1)
x4 = g_data4[0].ndata["xfeat"].transpose(0,1)

H1 = g_data1[0].ndata["hfeat"].transpose(0,1)
H2 = g_data2[0].ndata["hfeat"][0,:,:]
H3 = g_data3[0].ndata["hfeat"][0,:,:]
H4 = g_data4[0].ndata["hfeat"][0,:,:]
print(H1.shape)
print(H2.shape)
print(x1.shape)
model1.set_graph(g1)
pred00 = odeint(model1,x1[0,:,:],t,method="rk4")
H00 = model1.rollH(pred00)
model1.set_graph(g2)
pred01 = odeint(model1,x2[0,:,:],t,method="rk4")
H01 = model1.rollH(pred01)
model1.set_graph(g3)
pred02 = odeint(model1,x3[0,:,:],t,method="rk4")
H02 = model1.rollH(pred02)
model1.set_graph(g4)
pred03 = odeint(model1,x4[0,:,:],t,method="rk4")
H03 = model1.rollH(pred03)

model2.set_graph(g1)
pred10 = odeint(model2,x1[0,:,:],t,method="rk4")
H10 = model2.rollH(pred10)
model2.set_graph(g2)
pred11 = odeint(model2,x2[0,:,:],t,method="rk4")
H11 = model2.rollH(pred11)
model2.set_graph(g3)
pred12 = odeint(model2,x3[0,:,:],t,method="rk4")
H12 = model2.rollH(pred12)
model2.set_graph(g4)
pred13 = odeint(model2,x4[0,:,:],t,method="rk4")
H13 = model2.rollH(pred13)

model3.set_graph(g1)
pred20 = odeint(model3,x1[0,:,:],t,method="rk4")
H20 = model3.rollH(pred20)
model3.set_graph(g2)
pred21 = odeint(model3,x2[0,:,:],t,method="rk4")
H21 = model3.rollH(pred21)
model3.set_graph(g3)
pred22 = odeint(model3,x3[0,:,:],t,method="rk4")
H22 = model3.rollH(pred22)
model3.set_graph(g4)
pred23 = odeint(model3,x4[0,:,:],t,method="rk4")
H23 = model3.rollH(pred23)

model4.set_graph(g1)
pred30 = odeint(model4,x1[0,:,:],t,method="rk4")
H30 = model4.rollH(pred30)
model4.set_graph(g2)
pred31 = odeint(model4,x2[0,:,:],t,method="rk4")
H31 = model4.rollH(pred31)
model4.set_graph(g3)
pred32 = odeint(model4,x3[0,:,:],t,method="rk4")
H32 = model4.rollH(pred32)
model4.set_graph(g4)
pred33 = odeint(model4,x4[0,:,:],t,method="rk4")
H33 = model4.rollH(pred33)

print(H00.shape)
print(H2.shape)
## first
fig,ax =plt.subplots(2,4)
ax[0,0].set_title("model1")
ax[0,0].scatter(x1[:,0,0].detach().numpy(),x1[:,0,1].detach().numpy())
ax[0,0].scatter(pred00[:,0,0].detach().numpy(),pred00[:,0,1].detach().numpy()) 
ax[0,0].legend(["ground truth","prediction"])

ax[0,1].set_title("model2")
ax[0,1].scatter(x1[:,0,0].detach().numpy(),x1[:,0,1].detach().numpy())
ax[0,1].scatter(pred10[:,0,0].detach().numpy(),pred10[:,0,1].detach().numpy())  
ax[0,1].legend(["ground truth","prediction"])

ax[0,2].set_title("model3")
ax[0,2].scatter(x1[:,0,0].detach().numpy(),x1[:,0,1].detach().numpy())
ax[0,2].scatter(pred20[:,0,0].detach().numpy(),pred20[:,0,1].detach().numpy()) 
ax[0,2].legend(["ground truth","prediction"])

ax[0,3].set_title("model4")
ax[0,3].scatter(x1[:,0,0].detach().numpy(),x1[:,0,1].detach().numpy())
ax[0,3].scatter(pred30[:,0,0].detach().numpy(),pred30[:,0,1].detach().numpy()) 
ax[0,3].legend(["ground truth","prediction"])

ax[1,0].set_title("mod1 H")
ax[1,0].plot(t.detach().numpy(),H1[:,0].detach().numpy())
ax[1,0].plot(t.detach().numpy(),H00[:,0].detach().numpy())
ax[1,0].legend(["ground truth","prediction"])

ax[1,1].set_title("mod2 H")
ax[1,1].plot(t.detach().numpy(),H1[:,0].detach().numpy())
ax[1,1].plot(t.detach().numpy(),H10[:,0].detach().numpy())
ax[1,1].legend(["ground truth","prediction"])

ax[1,2].set_title("mod3 H")
ax[1,2].plot(t.detach().numpy(),H1[:,0].detach().numpy())
ax[1,2].plot(t.detach().numpy(),H20[:,0].detach().numpy())
ax[1,2].legend(["ground truth","prediction"])

ax[1,3].set_title("mod4 H")
ax[1,3].plot(t.detach().numpy(),H1[:,0].detach().numpy())
ax[1,3].plot(t.detach().numpy(),H30[:,0].detach().numpy())
ax[1,3].legend(["ground truth","prediction"])

plt.show()

fig,ax =plt.subplots(3,4)
ax[0,0].set_title("model1 b1")
ax[0,0].scatter(x2[:,0,0].detach().numpy(),x2[:,0,1].detach().numpy())
ax[0,0].scatter(pred01[:,0,0].detach().numpy(),pred01[:,0,1].detach().numpy()) 
ax[0,0].legend(["ground truth","prediction"])

ax[1,0].set_title("model1 b2")
ax[1,0].scatter(x2[:,1,0].detach().numpy(),x2[:,1,1].detach().numpy())
ax[1,0].scatter(pred01[:,1,0].detach().numpy(),pred01[:,1,1].detach().numpy()) 
ax[1,0].legend(["ground truth","prediction"])

ax[0,1].set_title("model2 b1")
ax[0,1].scatter(x2[:,0,0].detach().numpy(),x2[:,0,1].detach().numpy())
ax[0,1].scatter(pred11[:,0,0].detach().numpy(),pred11[:,0,1].detach().numpy()) 
ax[0,1].legend(["ground truth","prediction"])

ax[1,1].set_title("model2 b2")
ax[1,1].scatter(x2[:,1,0].detach().numpy(),x2[:,1,1].detach().numpy())
ax[1,1].scatter(pred11[:,1,0].detach().numpy(),pred11[:,1,1].detach().numpy()) 
ax[1,1].legend(["ground truth","prediction"])

ax[0,2].set_title("model3 b1")
ax[0,2].scatter(x2[:,0,0].detach().numpy(),x2[:,0,1].detach().numpy())
ax[0,2].scatter(pred21[:,0,0].detach().numpy(),pred21[:,0,1].detach().numpy()) 
ax[0,2].legend(["ground truth","prediction"])

ax[1,2].set_title("model3 b2")
ax[1,2].scatter(x2[:,1,0].detach().numpy(),x2[:,1,1].detach().numpy())
ax[1,2].scatter(pred21[:,1,0].detach().numpy(),pred21[:,1,1].detach().numpy()) 
ax[1,2].legend(["ground truth","prediction"])

ax[0,3].set_title("model4 b1")
ax[0,3].scatter(x2[:,0,0].detach().numpy(),x2[:,0,1].detach().numpy())
ax[0,3].scatter(pred31[:,0,0].detach().numpy(),pred31[:,0,1].detach().numpy()) 
ax[0,3].legend(["ground truth","prediction"])

ax[1,3].set_title("model4 b2")
ax[1,3].scatter(x2[:,1,0].detach().numpy(),x2[:,1,1].detach().numpy())
ax[1,3].scatter(pred31[:,1,0].detach().numpy(),pred31[:,1,1].detach().numpy()) 
ax[1,3].legend(["ground truth","prediction"])

ax[2,0].set_title("mod1 H")
ax[2,0].plot(t.detach().numpy(),H2[:,0].detach().numpy())
ax[2,0].plot(t.detach().numpy(),H01[:,0].detach().numpy())
ax[2,0].legend(["ground truth","prediction"])

ax[2,1].set_title("mod2 H")
ax[2,1].plot(t.detach().numpy(),H2[:,0].detach().numpy())
ax[2,1].plot(t.detach().numpy(),H11[:,0].detach().numpy())
ax[2,1].legend(["ground truth","prediction"])

ax[2,2].set_title("mod3 H")
ax[2,2].plot(t.detach().numpy(),H2[:,0].detach().numpy())
ax[2,2].plot(t.detach().numpy(),H21[:,0].detach().numpy())
ax[2,2].legend(["ground truth","prediction"])

ax[2,3].set_title("mod4 H")
ax[2,3].plot(t.detach().numpy(),H2[:,0].detach().numpy())
ax[2,3].plot(t.detach().numpy(),H31[:,0].detach().numpy())
ax[2,3].legend(["ground truth","prediction"])

plt.show()


fig,ax =plt.subplots(4,4)
ax[0,0].set_title("model1 b1")
ax[0,0].scatter(x3[:,0,0].detach().numpy(),x3[:,0,1].detach().numpy())
ax[0,0].scatter(pred02[:,0,0].detach().numpy(),pred02[:,0,1].detach().numpy()) 
ax[0,0].legend(["ground truth","prediction"])

ax[1,0].set_title("model1 b2")
ax[1,0].scatter(x3[:,1,0].detach().numpy(),x3[:,1,1].detach().numpy())
ax[1,0].scatter(pred02[:,1,0].detach().numpy(),pred02[:,1,1].detach().numpy()) 
ax[1,0].legend(["ground truth","prediction"])

ax[2,0].set_title("model1 b3")
ax[2,0].scatter(x3[:,2,0].detach().numpy(),x3[:,2,1].detach().numpy())
ax[2,0].scatter(pred02[:,2,0].detach().numpy(),pred02[:,2,1].detach().numpy()) 
ax[2,0].legend(["ground truth","prediction"])

ax[0,1].set_title("model2 b1")
ax[0,1].scatter(x3[:,0,0].detach().numpy(),x3[:,0,1].detach().numpy())
ax[0,1].scatter(pred12[:,0,0].detach().numpy(),pred12[:,0,1].detach().numpy()) 
ax[0,1].legend(["ground truth","prediction"])

ax[1,1].set_title("model2 b2")
ax[1,1].scatter(x3[:,1,0].detach().numpy(),x3[:,1,1].detach().numpy())
ax[1,1].scatter(pred12[:,1,0].detach().numpy(),pred12[:,1,1].detach().numpy()) 
ax[1,1].legend(["ground truth","prediction"])

ax[2,1].set_title("model2 b3")
ax[2,1].scatter(x3[:,2,0].detach().numpy(),x3[:,2,1].detach().numpy())
ax[2,1].scatter(pred12[:,2,0].detach().numpy(),pred12[:,2,1].detach().numpy()) 
ax[2,1].legend(["ground truth","prediction"])

ax[0,2].set_title("model3 b1")
ax[0,2].scatter(x3[:,0,0].detach().numpy(),x3[:,0,1].detach().numpy())
ax[0,2].scatter(pred22[:,0,0].detach().numpy(),pred22[:,0,1].detach().numpy()) 
ax[0,2].legend(["ground truth","prediction"])

ax[1,2].set_title("model3 b2")
ax[1,2].scatter(x3[:,1,0].detach().numpy(),x3[:,1,1].detach().numpy())
ax[1,2].scatter(pred22[:,1,0].detach().numpy(),pred22[:,1,1].detach().numpy()) 
ax[1,2].legend(["ground truth","prediction"])

ax[2,2].set_title("model3 b3")
ax[2,2].scatter(x3[:,2,0].detach().numpy(),x3[:,2,1].detach().numpy())
ax[2,2].scatter(pred22[:,2,0].detach().numpy(),pred22[:,2,1].detach().numpy()) 
ax[2,2].legend(["ground truth","prediction"])


ax[0,3].set_title("model4 b1")
ax[0,3].scatter(x3[:,0,0].detach().numpy(),x3[:,0,1].detach().numpy())
ax[0,3].scatter(pred32[:,0,0].detach().numpy(),pred32[:,0,1].detach().numpy()) 
ax[0,3].legend(["ground truth","prediction"])

ax[1,3].set_title("model4 b2")
ax[1,3].scatter(x3[:,1,0].detach().numpy(),x3[:,1,1].detach().numpy())
ax[1,3].scatter(pred32[:,1,0].detach().numpy(),pred32[:,1,1].detach().numpy()) 
ax[1,3].legend(["ground truth","prediction"])

ax[2,3].set_title("model4 b3")
ax[2,3].scatter(x3[:,2,0].detach().numpy(),x3[:,2,1].detach().numpy())
ax[2,3].scatter(pred32[:,2,0].detach().numpy(),pred32[:,2,1].detach().numpy()) 
ax[2,3].legend(["ground truth","prediction"])

ax[3,0].set_title("mod1 H")
ax[3,0].plot(t.detach().numpy(),H3[:,0].detach().numpy())
ax[3,0].plot(t.detach().numpy(),H02[:,0].detach().numpy())
ax[3,0].legend(["ground truth","prediction"])

ax[3,1].set_title("mod2 H")
ax[3,1].plot(t.detach().numpy(),H3[:,0].detach().numpy())
ax[3,1].plot(t.detach().numpy(),H12[:,0].detach().numpy())
ax[3,1].legend(["ground truth","prediction"])

ax[3,2].set_title("mod3 H")
ax[3,2].plot(t.detach().numpy(),H3[:,0].detach().numpy())
ax[3,2].plot(t.detach().numpy(),H22[:,0].detach().numpy())
ax[3,2].legend(["ground truth","prediction"])

ax[3,3].set_title("mod4 H")
ax[3,3].plot(t.detach().numpy(),H3[:,0].detach().numpy())
ax[3,3].plot(t.detach().numpy(),H32[:,0].detach().numpy())
ax[3,3].legend(["ground truth","prediction"])

plt.show()

fig,ax =plt.subplots(5,4)
ax[0,0].set_title("model1 b1")
ax[0,0].scatter(x4[:,0,0].detach().numpy(),x4[:,0,1].detach().numpy())
ax[0,0].scatter(pred03[:,0,0].detach().numpy(),pred03[:,0,1].detach().numpy()) 
ax[0,0].legend(["ground truth","prediction"])

ax[1,0].set_title("model1 b2")
ax[1,0].scatter(x4[:,1,0].detach().numpy(),x4[:,1,1].detach().numpy())
ax[1,0].scatter(pred03[:,1,0].detach().numpy(),pred03[:,1,1].detach().numpy()) 
ax[1,0].legend(["ground truth","prediction"])

ax[2,0].set_title("model1 b3")
ax[2,0].scatter(x4[:,2,0].detach().numpy(),x4[:,2,1].detach().numpy())
ax[2,0].scatter(pred03[:,2,0].detach().numpy(),pred03[:,2,1].detach().numpy()) 
ax[2,0].legend(["ground truth","prediction"])

ax[3,0].set_title("model1 b4")
ax[3,0].scatter(x4[:,3,0].detach().numpy(),x4[:,3,1].detach().numpy())
ax[3,0].scatter(pred03[:,3,0].detach().numpy(),pred03[:,3,1].detach().numpy()) 
ax[3,0].legend(["ground truth","prediction"])

ax[0,1].set_title("model2 b1")
ax[0,1].scatter(x4[:,0,0].detach().numpy(),x4[:,0,1].detach().numpy())
ax[0,1].scatter(pred13[:,0,0].detach().numpy(),pred13[:,0,1].detach().numpy()) 
ax[0,1].legend(["ground truth","prediction"])

ax[1,1].set_title("model2 b2")
ax[1,1].scatter(x4[:,1,0].detach().numpy(),x4[:,1,1].detach().numpy())
ax[1,1].scatter(pred13[:,1,0].detach().numpy(),pred13[:,1,1].detach().numpy()) 
ax[1,1].legend(["ground truth","prediction"])

ax[2,1].set_title("model2 b3")
ax[2,1].scatter(x4[:,2,0].detach().numpy(),x4[:,2,1].detach().numpy())
ax[2,1].scatter(pred13[:,2,0].detach().numpy(),pred13[:,2,1].detach().numpy()) 
ax[2,1].legend(["ground truth","prediction"])

ax[3,1].set_title("model2 b4")
ax[3,1].scatter(x4[:,3,0].detach().numpy(),x4[:,3,1].detach().numpy())
ax[3,1].scatter(pred13[:,3,0].detach().numpy(),pred13[:,3,1].detach().numpy()) 
ax[3,1].legend(["ground truth","prediction"])

ax[0,2].set_title("model3 b1")
ax[0,2].scatter(x4[:,0,0].detach().numpy(),x4[:,0,1].detach().numpy())
ax[0,2].scatter(pred23[:,0,0].detach().numpy(),pred23[:,0,1].detach().numpy()) 
ax[0,2].legend(["ground truth","prediction"])

ax[1,2].set_title("model3 b2")
ax[1,2].scatter(x4[:,1,0].detach().numpy(),x4[:,1,1].detach().numpy())
ax[1,2].scatter(pred23[:,1,0].detach().numpy(),pred23[:,1,1].detach().numpy()) 
ax[1,2].legend(["ground truth","prediction"])

ax[2,2].set_title("model3 b3")
ax[2,2].scatter(x4[:,2,0].detach().numpy(),x4[:,2,1].detach().numpy())
ax[2,2].scatter(pred23[:,2,0].detach().numpy(),pred23[:,2,1].detach().numpy()) 
ax[2,2].legend(["ground truth","prediction"])

ax[3,2].set_title("model3 b4")
ax[3,2].scatter(x4[:,3,0].detach().numpy(),x4[:,3,1].detach().numpy())
ax[3,2].scatter(pred23[:,3,0].detach().numpy(),pred23[:,3,1].detach().numpy()) 
ax[3,2].legend(["ground truth","prediction"])

ax[0,3].set_title("model4 b1")
ax[0,3].scatter(x4[:,0,0].detach().numpy(),x4[:,0,1].detach().numpy())
ax[0,3].scatter(pred33[:,0,0].detach().numpy(),pred33[:,0,1].detach().numpy()) 
ax[0,3].legend(["ground truth","prediction"])

ax[1,3].set_title("model3 b2")
ax[1,3].scatter(x4[:,1,0].detach().numpy(),x4[:,1,1].detach().numpy())
ax[1,3].scatter(pred33[:,1,0].detach().numpy(),pred33[:,1,1].detach().numpy()) 
ax[1,3].legend(["ground truth","prediction"])

ax[2,3].set_title("model3 b3")
ax[2,3].scatter(x4[:,2,0].detach().numpy(),x4[:,2,1].detach().numpy())
ax[2,3].scatter(pred33[:,2,0].detach().numpy(),pred33[:,2,1].detach().numpy()) 
ax[2,3].legend(["ground truth","prediction"])

ax[3,3].set_title("model3 b4")
ax[3,3].scatter(x4[:,3,0].detach().numpy(),x4[:,3,1].detach().numpy())
ax[3,3].scatter(pred33[:,3,0].detach().numpy(),pred33[:,3,1].detach().numpy()) 
ax[3,3].legend(["ground truth","prediction"])

ax[4,0].set_title("mod1 H")
ax[4,0].plot(t.detach().numpy(),H4[:,0].detach().numpy())
ax[4,0].plot(t.detach().numpy(),H03[:,0].detach().numpy())
ax[4,0].legend(["ground truth","prediction"])

ax[4,1].set_title("mod2 H")
ax[4,1].plot(t.detach().numpy(),H4[:,0].detach().numpy())
ax[4,1].plot(t.detach().numpy(),H13[:,0].detach().numpy())
ax[4,1].legend(["ground truth","prediction"])

ax[4,2].set_title("mod3 H")
ax[4,2].plot(t.detach().numpy(),H4[:,0].detach().numpy())
ax[4,2].plot(t.detach().numpy(),H23[:,0].detach().numpy())
ax[4,2].legend(["ground truth","prediction"])

ax[4,3].set_title("mod4 H")
ax[4,3].plot(t.detach().numpy(),H4[:,0].detach().numpy())
ax[4,3].plot(t.detach().numpy(),H33[:,0].detach().numpy())
ax[4,3].legend(["ground truth","prediction"])

plt.show()



