#based on this paper: Alvaro Sanchez-Gonzalez, Victor Bapst, Kyle Cranmer, and Peter Battaglia. Hamiltonian graph
#networks with ode integrators. arXiv e-prints, pp. arXiv–1909, 2019.
import torch
from tqdm import tqdm
from hnn_model import *
from data_func import *
import dgl
from dgl.dataloading import GraphDataLoader
from torchdiffeq import odeint_adjoint as odeint
import random
import matplotlib.pyplot as plt
#torch.autograd.set_detect_anomaly(True)
stamp = str(random.randint(0,1e8))
print(stamp)
NO_LOOPS = True
S = 15
SPLIT = 0.9
TIMESIZE = 32 # 32
BATCHSIZE = 128 # 32
EPOCHS = 200

NORM=False
SOB=[0.30,1.0,0.0] # H , rollout
LR = 1e-4
LOSS_FN = "Huber"
name =""
if NORM:
    name = name+"norm"
if TIMESIZE==1:
    name= name+"_single"
src1 = src_list(12)
print(src1)
src2 = src_list(13)
print(src2)
dst1 = dst_list(12)
print(dst1)
dst2 = dst_list(13)
print(dst1)
if NO_LOOPS:
    src1,dst1 = del_loops(src1,dst1)
    src2,dst2 = del_loops(src2,dst2)
    
graph1 = dgl.graph((src1,dst1))
graph2 = dgl.graph((src2,dst2))

print("LOADING DATA")
data1 , H1, data2 , H2  =dataset_loader()
H1 = torch.cat(([H1.unsqueeze(2).unsqueeze(3)]*12),dim=2).float()
H2 = torch.cat(([H2.unsqueeze(2).unsqueeze(3)]*13),dim=2).float()
print("shape of H1: {}".format(H1.shape))
print("shape of H2: {}".format(H2.shape))


if NORM:
    
    print("ADDED NORM")
    
    data = torch.cat((data1,data2),dim=2)
    print("TEST NORM")
    corr=minimax_test(data[:,:,:,0:3])
    print("TEST: {}".format(corr))
    if corr <= 1e-6:
        print("TEST PASSED")
    H_comb = torch.cat((H1,H2),dim=2)
    print("data combined shape: {}".format(data.shape))
    print("H combined shape: {}".format(H_comb.shape))
    q_norm, qmaxim, qminim = minmax(data[:,:,:,0:3])
    p_norm, pmaxim, pminim = minmax(data[:,:,:,3:6])
    dq_norm, dqmaxim, dqminim = minmax(data[:,:,:,6:9])
    dp_norm, dpmaxim, dpminim = minmax(data[:,:,:,9:12])
    H_norm , Hmaxim, Hminim = minmax(H_comb)
    print(q_norm.shape)
    print(p_norm.shape)
    print(dq_norm.shape)
    print(dp_norm.shape)
    print(H_norm.shape)


    data_norm1 = torch.cat((q_norm[:,:,0:4,:],p_norm[:,:,0:4,:],dq_norm[:,:,0:4,:],dp_norm[:,:,0:4,:]),dim=-1)
    H_norm1 = H_norm[:,:,0:4,:]

    data_norm2 = torch.cat((q_norm[:,:,4:9,:],p_norm[:,:,4:9,:],dq_norm[:,:,4:9,:],dp_norm[:,:,4:9,:]),dim=-1)
    H_norm2 = H_norm[:,:,4:9,:]

    x1_eval = data_norm1[:,-1,:,0:6]
    H1_eval = H_norm1[:,-1,:,:]
    x2_eval = data_norm2[:,-1,:,0:6]
    H2_eval = H_norm2[:,-1,:,:]
    
    x1 = data_norm1[:,0:S,:,:]
    x2 = data_norm2[:,0:S,:,:]
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

model = GNN_maker_HNN(graph1,6,128,6,["tanh",""],type="GCN")
print(model)

opti = torch.optim.AdamW(model.parameters(),lr=1e-3)
lossfn = loss_reader(LOSS_FN)
loss_container=torch.zeros(EPOCHS,4)

#########################################
# EXCLUSIVE FOR node4

xs, hs = make_snapshots(x1.float(),H1.float(),TIMESIZE)
print(xs[0].shape)
border = int(SPLIT*len(xs))
c = list(zip(xs, hs))
random.shuffle(c)
xs, hs = zip(*c)
train = xs[0:border]
test = xs[border:]

h_train = hs[0:border]
h_test = hs[border:]


gtrain = transform_dgl(src1,dst1,train,h_train)
gtest = transform_dgl(src1,dst1,test,h_test)

ts = t[0:TIMESIZE]



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
    src, dst = make_graph_no_loops(12,0)
    gtemp = dgl.graph((src,dst))
    #print(g.num_nodes())
    gs.append(gtemp)
#print(len(gs))
#print(g.num_nodes())
roll_g1 = dgl.batch(gs)
#print(roll_g4.num_nodes())

for epoch in tqdm(range(EPOCHS)):
    
    for sample in tqdm(trainset):
        model.change_graph(g)
        lossH=0
        lossROLL=0
        loss=0
        opti.zero_grad()
        
       
        x_tr,dx_tr,H_tr = get_d_dx_H(sample)
        #print("samples")    
        #print(x_tr.shape)
        #print(dx_tr.shape)
        #print(H_tr.shape)
        
        #print(roll_g1)
        model.change_graph(roll_g1)
        #print(roll_g4.num_nodes)
        #print(model.g)
        x_tr = x_tr.requires_grad_()
        x_tr_flat = x_tr.reshape(-1,6)
        #print(x_tr_flat.shape)
            #print(x_tr_flat.shape)
        h_pred = model(x_tr_flat)
            #h_rev = h_pred.reshape(TIMESIZE,BATCHSIZE)
            #print("h_pred shape {}".format(h_rev.shape))
            #print("h_pred {}".format(h_rev))
            
        lossH = lossfn(h_pred.flatten(),H_tr.flatten())
            
            # trajectory prediction 
        x0 = x_tr[0,:,:].requires_grad_()
        #print("x0 {}".format(x0.shape))
        model.change_graph(sample)
        x_pred = RKroll_for_learning(model,x0,ts)
            
            #print(dx_pred)
            #print(dx_pred.shape)
            
        lossROLL = lossfn(x_pred[:,:,0:3],x_tr[:,:,0:3])+lossfn(x_pred[:,:,3:6],x_tr[:,:,3:6])
        
        loss+=SOB[0] * lossH
        loss+=SOB[1] * lossROLL
        loss.backward()
        
        opti.step()
        loss_container[epoch,0] += loss.item()
    
    model.change_graph(gt)
    for sample in tqdm(testset):
        model.change_graph(gt)
        lossHt=0
        lossROLLt=0
        losst=0
        
        
        x_ts,dx_ts,H_ts = get_d_dx_H(sample)
        
        model.change_graph(roll_g1)
        x_ts = x_ts.requires_grad_()
        x_ts_flat = x_ts.reshape(-1,6)
            #print(x_tr_flat.shape)
        h_pred = model(x_ts_flat)
            #h_rev = h_pred.reshape(TIMESIZE,BATCHSIZE)
            #print("h_pred shape {}".format(h_rev.shape))
            #print("h_pred {}".format(h_rev))
            
        lossHt = lossfn(h_pred.flatten(),H_ts.flatten())
            
            # trajectory prediction 
        x0 = x_ts[0,:,:].requires_grad_()
        #print("x0 {}".format(x0.shape))
        model.change_graph(sample)
        x_pred = RKroll_for_learning(model,x0,ts)
            
        lossROLLt = lossfn(x_pred[:,0:3],x_ts[:,0:3])+lossfn(x_pred[:,3:6],x_ts[:,3:6])
        
        losst+=SOB[0] * lossHt
        losst+=SOB[1] * lossROLLt
        loss_container[epoch,1] += losst.item()
    loss_container[epoch,0]/=N_train
    loss_container[epoch,1]/=N_test
    dict={}
    for namew , param in model.named_parameters():
        dict[namew+"_grad"] = torch.mean(param.grad).item()
    print(dict)
    
    print("epoch: {}   train loss: {}  test loss: {}".format(epoch+1,loss_container[epoch,0],loss_container[epoch,1]))

torch.save(model, stamp+"node12_nbody_model_rollout"+name+".pth")


# EXCLUSIVE FOR node13

xs, hs = make_snapshots(x2.float(),H2,TIMESIZE)
print(xs[0].shape)
border = int(SPLIT*len(xs))
c = list(zip(xs, hs))
random.shuffle(c)
xs, hs = zip(*c)
train = xs[0:border]
test = xs[border:]

h_train = hs[0:border]
h_test = hs[border:]



gtrain = transform_dgl(src2,dst2,train,h_train)
gtest = transform_dgl(src2,dst2,test,h_test)

ts = t[0:TIMESIZE]



trainset = GraphDataLoader(gtrain,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
it = iter(trainset)
g = next(it)
testset = GraphDataLoader(gtest,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
it = iter(testset)
gt = next(it)
#model.change_graph(g)
N_train=len(trainset)
N_test=len(testset)
print("TRAIN BATCHES : {}".format(N_train))
gs=[]
for i in range(TIMESIZE*BATCHSIZE):
    src, dst = make_graph_no_loops(13,0)
    gtemp = dgl.graph((src,dst))
    #print(g.num_nodes())
    gs.append(gtemp)
#print(len(gs))
#print(g.num_nodes())
roll_g2 = dgl.batch(gs)
#print(roll_g4.num_nodes())

for epoch in tqdm(range(EPOCHS)):
    
    for sample in tqdm(trainset):
        model.change_graph(g)
        lossH=0
        lossROLL=0
        loss=0
        opti.zero_grad()
        
        
        x_tr,dx_tr,H_tr = get_d_dx_H(sample)
        #print("samples")    
        #print(x_tr.shape)
        #print(dx_tr.shape)
        #print(H_tr.shape)
        
        
        
    
        model.change_graph(roll_g2)
        #print(roll_g5.num_nodes)
        #print("train x_tr node5 {}".format(x_tr.shape))
        x_tr = x_tr.requires_grad_()
        x_tr_flat = x_tr.reshape(-1,6)
        #print("train x_tr flat node5 {}".format(x_tr_flat.shape))
        h_pred = model(x_tr_flat)
            #h_rev = h_pred.reshape(TIMESIZE,BATCHSIZE)
            #print("h_pred shape {}".format(h_rev.shape))
            #print("h_pred {}".format(h_rev))
            
        lossH = lossfn(h_pred.flatten(),H_tr.flatten())
            
        
        x0 = x_tr[0,:,:].requires_grad_()
        #print("x0 {}".format(x0.shape))
        model.change_graph(sample)
        x_pred = RKroll_for_learning(model,x0,ts)
            
            #print(dx_pred)
            #print(dx_pred.shape)
            
        lossROLL = lossfn(x_pred[:,:,0:3],x_tr[:,:,0:3])+lossfn(x_pred[:,:,3:6],x_tr[:,:,3:6])
        
        loss+=SOB[0] * lossH
        loss+=SOB[1] * lossROLL
        
        
        
        loss.backward()
        
        opti.step()
        loss_container[epoch,2] += loss.item()
    
    model.change_graph(gt)
    print("TEST")
    for sample in tqdm(testset):
        model.change_graph(gt)
        lossHt=0
        lossROLLt=0
        losst=0
        
        
        
        x_ts,dx_ts,H_ts = get_d_dx_H(sample)
        #print("samples")    
        #print(x_tr.shape)
        #print(dx_tr.shape)
        #print(H_tr.shape)
        
        
        
        
        
        model.change_graph(roll_g2)
        #print("test x_tr node5 {}".format(x_ts.shape))
        x_ts = x_ts.requires_grad_()
        x_ts_flat = x_ts.reshape(-1,6)
        #print(roll_g5.num_nodes)
        #print("test x_tr_flat node5 {}".format(x_ts_flat.shape))
        h_pred = model(x_ts_flat)
            #h_rev = h_pred.reshape(TIMESIZE,BATCHSIZE)
            #print("h_pred shape {}".format(h_rev.shape))
            #print("h_pred {}".format(h_rev))
        
        lossHt = lossfn(h_pred.flatten(),H_ts.flatten())
            
        x0 = x_ts[0,:,:].requires_grad_()
        
        model.change_graph(sample)
        x_pred = RKroll_for_learning(model,x0,ts)
            
        lossROLLt = lossfn(x_pred[:,0:3],x_ts[:,0:3])+lossfn(x_pred[:,3:6],x_ts[:,3:6])
        
        losst+=SOB[0] * lossHt
        losst+=SOB[1] * lossROLLt
        
        loss_container[epoch,3] += losst.item()
    loss_container[epoch,2]/=N_train
    loss_container[epoch,3]/=N_test
    dict={}
    for namew , param in model.named_parameters():
        dict[namew+"_grad"] = torch.mean(param.grad).item()
    print(dict)
    
    print("epoch: {}   train loss: {}  test loss: {}".format(epoch+1,loss_container[epoch,2],loss_container[epoch,3]))
    
torch.save(model, stamp+"node13_nbody_model_rollout"+name+".pth")
torch.save(loss_container,stamp+"_loss_rollout"+name+".pth")
