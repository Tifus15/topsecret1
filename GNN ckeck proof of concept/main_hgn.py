import torch
import torch.optim as opti
#from oscilator import *
from visualize import *
from utils import *
import random
from samples import *
from tqdm import tqdm
from threebody import *
from hgn_model import *
from dgl.dataloading import GraphDataLoader
from torchdiffeq import odeint_adjoint as odeint
SAMPLES = get_3body_samples()
maker = threebody(SAMPLES,device=torch.device("cpu"))
S = 10
t,x, dx, H = maker.make_dataset("fig8",S,128,[0,torch.pi/8])
print(t.shape)
print(x.shape)
print(dx.shape)
print(H.shape)
#xdata = torch.cat((x,dx),dim=-1)
gdataset = transform_threbody(x)
dgdataset = transform_threbody(dx)
eval = gdataset[:,-1,:,:]
deval = dgdataset[:,-1,:,:]
print(eval.shape)
print(deval.shape)
phasespace_show_threebody(eval)

TIME_SNAP = 2#32
BATCH_SIZE =64#32
EPOCHS = 200
SOB = [1.0,0.5]

#print(gdataset.shape)
#print(dgdataset.shape)
dataset = snap_maker(gdataset[:,0:S-1,:,:],TIME_SNAP)
ddataset = snap_maker(dgdataset[:,0:S-1,:,:],TIME_SNAP)
#print(dataset[0].shape)
#print(ddataset[1].shape)
src = [0,0,1,1,2,2]
dst = [1,2,0,2,0,1]

g = dgl.graph((src,dst))

dataset = transform2dgl(src,dst,dataset,ddataset)
#print("graph_data x: {}".format(dataset[0].ndata["x"].shape))
#print("graph_data dx: {}".format(dataset[0].ndata["dx"].shape))

g_train = dataset[0:int(len(dataset)*0.9)]
g_test = dataset[int(len(dataset)*0.9):]
model = HGNN(g,["tanh"],4,8,[64],True)
print(model)
for name,param in model.named_parameters():
    print("{} : {}".format(name, param.data.shape))
tb = t[0:TIME_SNAP]
optimizer = opti.AdamW(model.parameters(),lr=1e-3)
trainset = GraphDataLoader(g_train,batch_size=BATCH_SIZE,drop_last=True)
testset = GraphDataLoader(g_test,batch_size=BATCH_SIZE,drop_last=True)
it = iter(trainset)
g_batch = next(it)
loss_fn = nn.HuberLoss()
loss_r = nn.L1Loss()
model.change_graph(g_batch)
a=100
for epoch in tqdm(range(EPOCHS)):
    losstr_acc=0
    losstr_g =0
    lossts_acc=0
    lossts_g =0
    model.train()
    print("TRAIN")
    for sample in tqdm(trainset):
        optimizer.zero_grad()
        data = sample.ndata["x"].transpose(0,1).requires_grad_()
        #print("original x {}".format(data.shape))
        ddata = sample.ndata["dx"].transpose(0,1)
        #print("original dx {}".format(ddata.shape))
        x0 = data[0,:,:]
        #print("start")
        #pred = odeint(model,x0,tb,method="rk4")
        pred = rollout(x0,tb,model)
        #print(pred.shape)
        #print(pred.is_leaf) 
        #print("end")
        #print("pred x {}".format(pred.shape))
        dpred = rolloutdxHGN(model,data) # rollout the vectors from correct data
        #print("pred dx {}".format(dpred.shape))
        #print("pred: {}".format(pred.shape))
        #print("data: {}".format(data.shape))
        loss1 = SOB[0]*a*(loss_fn(pred[:,:,0:2],data[:,:,0:2]) + loss_fn(pred[:,:,2:4],data[:,:,2:4]))
        loss2 = SOB[1]*a* (loss_fn(dpred[:,:,0:2],ddata[:,:,0:2]) + loss_fn(dpred[:,:,2:4],ddata[:,:,2:4]))
        
        
        loss = loss1 + loss2
        print(loss)
        losstr_acc+=loss.item()
        losstr_g+=loss2.item()
        loss.backward()
        #for name,param in model.named_parameters():
        #dict[name]=torch.mean(param.data.flatten())
        #    print(name)
        optimizer.step()
        
    
    model.eval()
    print("TEST")
    for tsample in tqdm(testset):
        tdata = tsample.ndata["x"].transpose(0,1).requires_grad_()
        dtdata = tsample.ndata["dx"].transpose(0,1)
        x0t = tdata[0,:,:]#.requires_grad_()
        #predt = odeint(model,x0t,tb,method="rk4")
        predt = rollout(x0t,tb,model)
        dpredt = rolloutdxHGN(model,tdata)
        loss1 = SOB[0]*a*(loss_fn(predt[:,:,0:2],tdata[:,:,0:2]) + loss_fn(predt[:,:,2:4],tdata[:,:,2:4]))
        loss2 = SOB[1]*a* (loss_fn(dpredt[:,:,0:2],dtdata[:,:,0:2]) + loss_fn(dpredt[:,:,2:4],dtdata[:,:,2:4]))
        
        lossts_g += loss2.item() 
        loss = loss1 +loss2
        lossts_acc+=loss.item()
    dict={}
    for name,param in model.named_parameters():
        #dict[name]=torch.mean(param.data.flatten())
        print(name,param.grad)
        #dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()
    #print("#######")
    N_train = len(trainset)
    N_test = len(testset)
    wt = losstr_acc/N_train
    wst = lossts_acc/N_test
    wqt = losstr_g/N_train
    wqst = lossts_g/N_test
    print("epoch: {}, train loss: {}, test_loss: {}, train grad: {}, test_grad {}".format(epoch+1,wt,wst,wqt,wqst))
    print(dict)
model.change_graph(g)    
pred = rollout(eval[0,:,:],t,model)
phasespace_show_threebody_pred(eval,pred)