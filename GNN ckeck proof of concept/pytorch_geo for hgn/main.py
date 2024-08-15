from threebody import *
from model import *
import torch
import torch.optim as opti
import random
from torch_geometric.data import Data
from utils import *
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch


SAMPLES = get_3body_samples()
maker = threebody(SAMPLES,device=torch.device("cpu"))
S = 5
t,x, dx, H = maker.make_dataset("fig8",S,128,[0,torch.pi/4])
print(t.shape)
print(x.shape)
print(dx.shape)
print(H.shape)

SOB = [1.0,1.0]

gdataset = transform_threbody(x)
dgdataset = transform_threbody(dx)

eval = gdataset[:,-1,:,:]
deval = dgdataset[:,-1,:,:]
phasespace_show_threebody(eval)

TIME_SNAP = 2#32
BATCH_SIZE = 1#32
EPOCHS = 1000

src = [0,0,1,1,2,2]
dst = [1,2,0,2,0,1]

dataset = snap_maker(gdataset[:,0:S-1,:,:],TIME_SNAP)
ddataset = snap_maker(dgdataset[:,0:S-1,:,:],TIME_SNAP)
random.shuffle(dataset)
pygDataset = makePyG3dofDataset(src,dst,dataset,ddataset)

print(pygDataset[0])

g = torch.tensor([src,dst],dtype=torch.long)

model = HGN(g,["tanh"], in_dim=4, h_dim=8, hidden=[64],nodes=3,bias=False)
print(model)
split = int(0.9 * len(pygDataset)) 

train = pygDataset[0:split]
test = pygDataset[split:]
tb = t[0:TIME_SNAP]
trainset = DataLoader(train,BATCH_SIZE,True,drop_last = True)
testset = DataLoader(test,BATCH_SIZE,True,drop_last = True)

it = iter(trainset)
g_batch = next(it)
optimizer = opti.Adam(model.parameters(),lr=1e-3)

loss_fn = nn.HuberLoss()
loss_r = nn.L1Loss()
model.change_graph(g_batch.edge_index)
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
        x = sample.x.transpose(0,1).requires_grad_()
        #ei = sample.edge_index
        dx = sample.dx.transpose(0,1)
        x0 = x[0,:,:]
        pred = rollout(x0,tb,model)
        dpred = rolloutdxHGN(model,x)
        
        #loss1 = SOB[0]*(loss_fn(pred[:,:,0:2],x[:,:,0:2]) + loss_fn(pred[:,:,2:4],x[:,:,2:4]))
        loss2 = SOB[1]*(loss_fn(dpred[:,:,0:2],dx[:,:,0:2]) + loss_fn(dpred[:,:,2:4],dx[:,:,2:4]))
        
        loss =   loss2
        
        losstr_acc+=loss.item()
        losstr_g+=loss2.item()
        loss.backward()
        optimizer.step()
    model.eval()
    for tsample in tqdm(testset):
        xt = tsample.x.transpose(0,1).requires_grad_()
        #ei = tsample.edge_index
        dxt = tsample.dx.transpose(0,1)
        
        x0t = xt[0,:,:]
        predt = rollout(x0t,tb,model)
        dpredt = rolloutdxHGN(model,xt)
        
        #loss1 = SOB[0]*a*(loss_fn(predt[:,:,0:2],xt[:,:,0:2]) + loss_fn(predt[:,:,2:4],xt[:,:,2:4]))
        loss2 = SOB[1]*a* (loss_fn(dpredt[:,:,0:2],dxt[:,:,0:2]) + loss_fn(dpredt[:,:,2:4],dxt[:,:,2:4]))
        
        lossts_g += loss2.item() 
        loss = loss2
        lossts_acc+=loss.item()
    
    dict={}
    for name,param in model.named_parameters():
        #dict[name]=torch.mean(param.data.flatten())
        #print(name,param.grad)
        if param.grad is None:
            dict[name+"_grad"] = None
        else:
            dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()

    N_train = len(trainset)
    N_test = len(testset)
    wt = losstr_acc/N_train
    wst = lossts_acc/N_test
    wqt = losstr_g/N_train
    wqst = lossts_g/N_test
    print("epoch: {}, train loss: {}, test_loss: {}, train grad: {}, test_grad {}".format(epoch+1,wt,wst,wqt,wqst))
    print(dict)
model.change_graph(g)    
pred = rollout(eval[0,:,:].requires_grad_(),t,model)
phasespace_show_threebody_pred(eval,pred)