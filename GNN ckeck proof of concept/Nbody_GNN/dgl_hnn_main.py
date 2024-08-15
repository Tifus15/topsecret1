import torch
from tqdm import tqdm
from hnn_model import *
from data_func import *
import dgl
from dgl.dataloading import GraphDataLoader
from torchdiffeq import odeint_adjoint as odeint
import random
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
NO_LOOPS = True
SPLIT = 0.9
TIMESIZE = 1 # 32
BATCHSIZE = 128 # 32
EPOCHS = 100
REG=False
SOB=[1.0,0.01,0.0]
S = 100
src4 = src_list(4)
print(src4)
src5 = src_list(5)
print(src5)
dst4 = dst_list(4)
print(dst4)
dst5 = dst_list(5)
print(dst5)
if NO_LOOPS:
    src4,dst4 = del_loops(src4,dst4)
    src5,dst5 = del_loops(src5,dst5)
    
graph4 = dgl.graph((src4,dst4))
graph5 = dgl.graph((src5,dst5))

print("LOaDING DATA")
data1 , H1, data2 , H2  =dataset_loader()
H1 = torch.cat(([H1.unsqueeze(2).unsqueeze(3)]*4),dim=2)
H2 = torch.cat(([H2.unsqueeze(2).unsqueeze(3)]*5),dim=2)
print("shape of H1: {}".format(H1.shape))
print("shape of H2: {}".format(H2.shape))


if REG:
    print("ADDED REGULARISATION")
    data = torch.cat((data1,data2),dim=2)
    H_comb = torch.cat((H1,H2),dim=2)
    print("data combined shape: {}".format(data.shape))
    print("H combined shape: {}".format(H_comb.shape))
    q_reg, qmaxim, qminim = minmax(data[:,:,:,0:3])
    p_reg, pmaxim, pminim = minmax(data[:,:,:,3:6])
    dq_reg, dqmaxim, dqminim = minmax(data[:,:,:,6:9])
    dp_reg, dpmaxim, dpminim = minmax(data[:,:,:,9:12])
    H_reg , Hmaxim, Hminim = minmax(H_comb)
    print(q_reg.shape)
    print(p_reg.shape)
    print(dq_reg.shape)
    print(dp_reg.shape)
    print(H_reg.shape)


    data_reg4 = torch.cat((q_reg[:,:,0:4,:],p_reg[:,:,0:4,:],dq_reg[:,:,0:4,:],dp_reg[:,:,0:4,:]),dim=-1)
    H_reg4 = H_reg[:,:,0:4,:]

    data_reg5 = torch.cat((q_reg[:,:,4:9,:],p_reg[:,:,4:9,:],dq_reg[:,:,4:9,:],dp_reg[:,:,4:9,:]),dim=-1)
    H_reg5 = H_reg[:,:,4:9,:]

    x1_eval = data_reg4[:,-1,:,0:6]
    H1_eval = H_reg4[:,-1,:,:]
    x2_eval = data_reg5[:,-1,:,0:6]
    H2_eval = H_reg5[:,-1,:,:]
    
    x1 = data_reg4[:,0:S,:,:]
    x2 = data_reg5[:,0:S,:,:]
else:
    x1_eval = data1[:,-1,:,0:6]
    x2_eval = data2[:,-1,:,0:6]
    H1_eval = H1[:,-1,:,:]
    H2_eval = H2[:,-1,:,:]
    #print(reg4.shape)
    x1 = data1[:,0:S,:,:]
    x2 = data2[:,0:S,:,:]

H1 = H1[:,0:S,:,:]
H2 = H2[:,0:S,:,:]
print(H1.shape)
# the first dataset will be used fully as training and test set
t = torch.linspace(0,1.27,128) # dt = 0.01
print(t[1]-t[0])

model = GNN_maker_HNN(graph4,6,128,6,["tanh",""],type="GAT")
print(model)

opti = torch.optim.AdamW(model.parameters(),lr=1e-3)
lossfn = torch.nn.MSELoss()
#########################################
# EXCLUSIVE FOR node4
if TIMESIZE == 1:
    print("TIMESIZE: 1")
    xs, hs = make_simple_snapshots(x1.float(),H1.float())
else:
    xs, hs = make_snapshots(x1.float(),H1,TIMESIZE)
print(xs[0].shape)
border = int(SPLIT*len(xs))
c = list(zip(xs, hs))
random.shuffle(c)
xs, hs = zip(*c)
train = xs[0:border]
test = xs[border:]

h_train = hs[0:border]
h_test = hs[border:]

if TIMESIZE == 1:
    gtrain = transform_simple_dgl(src4,dst4,train,h_train)
    gtest = transform_simple_dgl(src4,dst4,test,h_test)
else:
    gtrain = transform_dgl(src4,dst4,train,h_train)
    gtest = transform_dgl(src4,dst4,test,h_test)

ts = t[0:TIMESIZE]

trainloss1 = []
testloss1 = []

trainset = GraphDataLoader(gtrain,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
it = iter(trainset)
g = next(it)
#model.change_graph(g)
N_train=len(trainset)
print("TRAIN BATCHES : {}".format(N_train))
testset = GraphDataLoader(gtest,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
it = iter(testset)
gt = next(it)

#model.change_graph(g)
N_test=len(testset)
print("TEST BATCHES : {}".format(N_test))
gs=[]
for i in range(TIMESIZE*BATCHSIZE):
    src, dst = make_graph_no_loops(4,0)
    gtemp = dgl.graph((src,dst))
    #print(g.num_nodes())
    gs.append(gtemp)
#print(len(gs))
#print(g.num_nodes())
roll_g4 = dgl.batch(gs)
#print(roll_g4.num_nodes())
loss_container=torch.zeros(EPOCHS,4)
for epoch in tqdm(range(EPOCHS)):
    
    for sample in tqdm(trainset):
        model.change_graph(g)
        lossH=0
        lossDX=0
        loss=0
        opti.zero_grad()
        
        if TIMESIZE ==1:
            x_tr,dx_tr,H_tr = get_simple_elements(sample)
        else:
            x_tr,dx_tr,H_tr = get_d_dx_H(sample)
        #print("samples")    
        #print(x_tr.shape)
        #print(dx_tr.shape)
        #print(H_tr.shape)
        
        if TIMESIZE ==1:
            # H prediction
            x_tr =x_tr.requires_grad_()
            h_pred = model(x_tr)
            #print("h_pred shape {}".format(h_pred.shape))
            #print("h_pred {}".format(h_pred))
            
            lossH = lossfn(h_pred,H_tr)
            
            # dhdx prediction 
            
            H_l = torch.split(h_pred,1,dim=1)
            #print(len(H_l))
            dHdx = torch.autograd.grad(H_l,x_tr,retain_graph=True, create_graph=True)[0]
            #print(dHdx)
            #print(dHdx.shape)
            dqdp_s = torch.split(dHdx,3,dim=-1)
            dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
            #print(dx_pred)
            #print(dx_pred.shape)
            
            lossDX = lossfn(dx_pred[:,0:3],dx_tr[:,0:3])+lossfn(dx_pred[:,3:6],dx_tr[:,3:6])
        
        
        else:
            model.change_graph(roll_g4)
            x_tr = x_tr.requires_grad_()
            x_tr_flat = x_tr.reshape(-1,6)
            #print(x_tr_flat.shape)
            h_pred = model(x_tr_flat)
            #h_rev = h_pred.reshape(TIMESIZE,BATCHSIZE)
            #print("h_pred shape {}".format(h_rev.shape))
            #print("h_pred {}".format(h_rev))
            
            lossH = lossfn(h_pred.flatten(),H_tr.flatten())
            
            # dhdx prediction 
            H_l = torch.split(h_pred,1,dim=1)
            #print(len(H_l))
            dHdx = torch.autograd.grad(H_l,x_tr,retain_graph=True, create_graph=True)[0]
            dqdp_s = torch.split(dHdx,3,dim=-1)
            dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
            #print(dx_pred)
            #print(dx_pred.shape)
            
            lossDX = lossfn(dx_pred[:,:,0:3],dx_tr[:,:,0:3])+lossfn(dx_pred[:,:,3:6],dx_tr[:,:,3:6])
        
        loss+=SOB[0] * lossH
        loss+=SOB[1] * lossDX
        loss.backward()
        
        opti.step()
        loss_container[epoch,0] += loss.item()
    
    model.change_graph(gt)
    for sample in tqdm(testset):
        model.change_graph(gt)
        lossHt=0
        lossDXt=0
        losst=0
        
        if TIMESIZE ==1:
            x_ts,dx_ts,H_ts = get_simple_elements(sample)
        else:
            x_ts,dx_ts,H_ts = get_d_dx_H(sample)
        #print("samples")    
        #print(x_tr.shape)
        #print(dx_tr.shape)
        #print(H_tr.shape)
        
        if TIMESIZE ==1:
            # H prediction
            x_ts =x_ts.requires_grad_()
            h_pred = model(x_ts)
            #print("h_pred shape {}".format(h_pred.shape))
            #print("h_pred {}".format(h_pred))
            
            lossHt = lossfn(h_pred,H_ts)
            
            # dhdx prediction 
            
            H_l = torch.split(h_pred,1,dim=1)
            #print(len(H_l))
            dHdx = torch.autograd.grad(H_l,x_ts,retain_graph=True, create_graph=True)[0]
            #print(dHdx)
            #print(dHdx.shape)
            dqdp_s = torch.split(dHdx,3,dim=-1)
            dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
            #print(dx_pred)
            #print(dx_pred.shape)
            
            lossDXt = lossfn(dx_pred[:,0:3],dx_ts[:,0:3])+lossfn(dx_pred[:,3:6],dx_ts[:,3:6])
        
        
        else:
            model.change_graph(roll_g4)
            x_ts = x_ts.requires_grad_()
            x_ts_flat = x_ts.reshape(-1,6)
            #print(x_tr_flat.shape)
            h_pred = model(x_ts_flat)
            #h_rev = h_pred.reshape(TIMESIZE,BATCHSIZE)
            #print("h_pred shape {}".format(h_rev.shape))
            #print("h_pred {}".format(h_rev))
            
            lossHt = lossfn(h_pred.flatten(),H_ts.flatten())
            
            # dhdx prediction 
            H_l = torch.split(h_pred,1,dim=1)
            #print(len(H_l))
            dHdx = torch.autograd.grad(H_l,x_ts,retain_graph=True, create_graph=True)[0]
            dqdp_s = torch.split(dHdx,3,dim=-1)
            dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
            #print(dx_pred)
            #print(dx_pred.shape)
            
            lossDXt = lossfn(dx_pred[:,:,0:3],dx_ts[:,:,0:3])+lossfn(dx_pred[:,:,3:6],dx_ts[:,:,3:6])
        
        losst+=SOB[0] * lossH
        losst+=SOB[1] * lossDX
        loss_container[epoch,1] += losst.item()
    loss_container[epoch,0]/=N_train
    loss_container[epoch,1]/=N_test
    dict={}
    for name , param in model.named_parameters():
        dict[name+"_grad"] = torch.mean(param.grad).item()
    print(dict)
    
    print("epoch: {}   train loss: {}  test loss: {}".format(epoch+1,loss_container[epoch,0],loss_container[epoch,1]))
plt.figure()
plt.semilogy(torch.linspace(0,EPOCHS+1,EPOCHS),loss_container[:,0])    
plt.semilogy(torch.linspace(0,EPOCHS+1,EPOCHS),loss_container[:,1]) 
plt.show()  
        

