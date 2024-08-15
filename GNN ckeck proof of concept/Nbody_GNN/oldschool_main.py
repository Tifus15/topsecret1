import torch
import torch.optim as opti
from graph_util import *
from data_func import *
from torch.utils.data import DataLoader
from oldschool_gnnhnn import *
from tqdm import tqdm
import matplotlib.pyplot as plt

src = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
dst = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]

A = edges_to_A(src,dst,no_self_loops=True)

t = torch.linspace(0,1.27,128)
dt = t[1]-t[0]

SOB = [0.0,0.0,1.0]
REG = False
BATCH = 128
EPOCHS = 100
SPLIT = 0.9
MODEL = "GCN"
data1 , H1, _ , _  = dataset_loader()

print(data1.shape)
print(H1.shape)

x = data1[:,:,:,0:6]
dx = data1[:,:,:,6:12]
if REG:
    x_reg,x_max,x_min = minmax(x)
    dx_reg,dx_max,dx_min = minmax(dx)
    H1_reg,H1_max,H1_min = minmax(H1)

    x=x_reg
    dx=dx_reg
    H1=H1_reg
data_bib = graph_pack(A,x,dx,H1)
data_bib.print_keys()

samples = len(data_bib)
tr = int(samples*SPLIT)
ts = int(samples-tr)

n=6

model = GNN(n,1,[128],["relu"],type=MODEL)
optimizer = opti.AdamW(model.parameters(),lr=1e-4)

loss_fn=nn.MSELoss()
J = torch.eye(n)
J = torch.cat([J[n//2:], -J[:n//2]])

train_set, test_set = torch.utils.data.random_split(data_bib,[tr,ts])

print(len(train_set))
print(len(test_set))
loss_container= torch.zeros(EPOCHS,2)
trainloader = DataLoader(train_set,batch_size=BATCH,shuffle=True,drop_last=True)
testloader = DataLoader(test_set,batch_size=BATCH,shuffle=True,drop_last=True)
for epoch in tqdm(range(EPOCHS)):
    model.train()
    for x_tr, dx_tr, H_tr, nx_tr,adj_tr in tqdm(trainloader):
        losstr = 0
        optimizer.zero_grad()
        x_tr = x_tr.requires_grad_()
        H_mpred = model(x_tr,adj_tr)
        #H_mpred, lp = model(x_tr)
        #print(lp) 
        #print(H_mpred.shape)
        H_pred = torch.mean(H_mpred,dim=1) 
        #print(H_pred)
        # mean all parts on nodes
        #print("H prediction shape: {}".format(H_pred.shape))
        #print("H real shape {}".format(H_tr.shape))
        H_l = torch.split(H_pred,1,dim=0)
        dhdx = torch.autograd.grad(H_l,x_tr,retain_graph=True,create_graph=True)[0]
        dx_pred = dhdx @ J.transpose(0,1)
        print(dx_pred)
        #print("dx prediction shape: {}".format(dx_pred))
        #print("dx real shape {}".format(dx_tr))
        if SOB != 0:
            lossH = SOB[0] * loss_fn(H_pred,H_tr)
            losstr += lossH
        lossDX = SOB[1] * loss_fn(dx_pred,dx_tr)
        #print(lossDX.item())
        losstr += lossDX
        if SOB[2]!=0:
            ### RK4 step
            #K1
            
            H_pred1 = torch.mean(model(x_tr,adj_tr) ,dim=1) 
            H_l1 = torch.split(H_pred1,1,dim=0)
            dhdx1 = torch.autograd.grad(H_l1,x_tr,retain_graph=True)[0]
            K1 = dhdx1 @ J.transpose(0,1)
            #K2
            H_pred2 = torch.mean(model(x_tr+ dt*K1/2,adj_tr) ,dim=1) 
            H_l2 = torch.split(H_pred2,1,dim=0)
            dhdx2 = torch.autograd.grad(H_l2,x_tr,retain_graph=True)[0]
            K2 = dhdx2 @ J.transpose(0,1)
            #K3
            H_pred3 = torch.mean(model(x_tr+ dt*K2/2,adj_tr) ,dim=1) 
            H_l3 = torch.split(H_pred3,1,dim=0)
            dhdx3 = torch.autograd.grad(H_l3,x_tr,retain_graph=True)[0]
            K3 = dhdx3 @ J.transpose(0,1)
            #K4
            H_pred4 = torch.mean(model(x_tr+ dt*K3,adj_tr) ,dim=1) 
            H_l4 = torch.split(H_pred4,1,dim=0)
            dhdx4 = torch.autograd.grad(H_l4,x_tr,retain_graph=True)[0]
            K4 = dhdx4 @ J.transpose(0,1)
            
            nx_pred = x_tr + dt*(K1+2*K2+2*K3+K4)/6
            
            #print("nx prediction shape: {}".format(nx_tr.shape))
            #print("nx real shape {}".format(nx_pred.shape))
            lossNX = SOB[2]*loss_fn(nx_pred,nx_tr)
            losstr += lossNX
        loss_container[epoch,0] += losstr.item()
        
        
        losstr.backward()
        dict =  {}
        for name , param in model.named_parameters():
            dict[name+"_grad"] = param.grad
        print(dict) 
        optimizer.step() 
        
    model.eval()
    for x_ts, dx_ts, H_ts, nx_ts,adj_ts in tqdm(testloader):
        lossts = 0
        x_ts = x_ts.requires_grad_()
        H_mpred = model(x_ts,adj_ts) 
        #print(lp)
        H_pred = torch.mean(H_mpred,dim=1) 
        # mean all parts on nodes
        #print("H prediction shape: {}".format(H_pred.shape))
        #print("H real shape {}".format(H_tr.shape))
        H_l = torch.split(H_pred,1,dim=0)
        dhdx = torch.autograd.grad(H_l,x_ts,retain_graph=True)[0]
        dx_pred = dhdx @ J.transpose(0,1)
        #print("dx prediction shape: {}".format(dx_pred.shape))
        #print("dx real shape {}".format(dx_tr.shape))
        lossH = SOB[0] * loss_fn(H_pred,H_ts)
        lossts += lossH
        lossDX = SOB[1] * loss_fn(dx_pred,dx_ts)
        lossts += lossDX
        if SOB[2]!=0:
            ### RK4 step
            #K1
            
            H_pred1 = torch.mean(model(x_ts,adj_ts) ,dim=1) 
            H_l1 = torch.split(H_pred1,1,dim=0)
            dhdx1 = torch.autograd.grad(H_l1,x_ts,retain_graph=True)[0]
            K1 = dhdx1 @ J.transpose(0,1)
            #K2
            H_pred2 = torch.mean(model(x_ts+ dt*K1/2,adj_ts) ,dim=1) 
            H_l2 = torch.split(H_pred2,1,dim=0)
            dhdx2 = torch.autograd.grad(H_l2,x_ts,retain_graph=True)[0]
            K2 = dhdx2 @ J.transpose(0,1)
            #K3
            H_pred3 = torch.mean(model(x_ts + dt*K2/2,adj_ts) ,dim=1) 
            H_l3 = torch.split(H_pred3,1,dim=0)
            dhdx3 = torch.autograd.grad(H_l3,x_ts,retain_graph=True)[0]
            K3 = dhdx3 @ J.transpose(0,1)
            #K4
            H_pred4 = torch.mean(model(x_ts+ dt*K3,adj_ts) ,dim=1) 
            H_l4 = torch.split(H_pred4,1,dim=0)
            dhdx4 = torch.autograd.grad(H_l4,x_ts,retain_graph=True)[0]
            K4 = dhdx4 @ J.transpose(0,1)
            
            nx_pred = x_ts + dt*(K1+2*K2+2*K3+K4)/6
            
            #print("nx prediction shape: {}".format(nx_tr.shape))
            #print("nx real shape {}".format(nx_pred.shape))
            lossNX = SOB[2]*loss_fn(nx_pred,nx_ts)
            lossts += lossNX
        loss_container[epoch,1] += lossts.item()
        
        
        
    print("train loss: {}   test loss: {}".format(loss_container[epoch,0]/len(trainloader),loss_container[epoch,1]/len(testloader))) 
        
plt.figure()
plt.semilogy(torch.linspace(0,EPOCHS+1,EPOCHS),loss_container[:,0]/len(trainloader))    
plt.semilogy(torch.linspace(0,EPOCHS+1,EPOCHS),loss_container[:,1]/len(testloader)) 
plt.show()    
        






