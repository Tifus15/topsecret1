import torch
import torch.optim as opti
from oscilator import *
from visualize import *
from utils import *
import random
from samples import *
from tqdm import tqdm
from threebody import *
from model_gnn import *
from dgl.dataloading import GraphDataLoader

SAMPLES = get_3body_samples()
maker = threebody(SAMPLES,device=torch.device("cpu"))
S = 20
t,x, dx, H = maker.make_dataset("fig8",S,128,[0,torch.pi/8])
print(t.shape)
print(x.shape)
print(dx.shape)
print(H.shape)

gdataset = transform_threbody(x)

eval = gdataset[:,-1,:,:]
phasespace_show_threebody(eval)

TIME_SNAP = 32#32
BATCH_SIZE = 64#32
EPOCHS = 20

dataset = snap_maker(gdataset[:,0:S-1,:,:],TIME_SNAP)
random.shuffle(dataset)

snap_train = dataset[0:int(len(dataset)*0.9)]
snap_test = dataset[int(len(dataset)*0.9):]

src = [0,0,1,1,2,2]
dst = [1,2,0,2,0,1]
g = dgl.graph((src,dst))
g_train = transform2dgl(src,dst,snap_train)
g_test = transform2dgl(src,dst,snap_test)
model = GATGNN(g,["tanh"],4,4,[128])
print(model)
for name,param in model.named_parameters():
    print("{} : {}".format(name, param.data.shape))

tb = t[0:TIME_SNAP]
optimizer = opti.Adam(model.parameters(),lr=1e-2)
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
    lossts_acc=0
    model.train()
    print("TRAIN")
    for sample in tqdm(trainset):
        optimizer.zero_grad()
        data = sample.ndata["x"].transpose(0,1)
        x0 = data[0,:,:]
        pred = odeint(model,x0,tb,method="rk4")
        #print("pred: {}".format(pred.shape))
        #print("data: {}".format(data.shape))
        loss = a*(loss_fn(pred[:,:,0:2],data[:,:,0:2]) + loss_fn(pred[:,:,2:4],data[:,:,2:4]))
        loss.backward()
        optimizer.step()
        losstr_acc+=loss.item()
    
    #model.change_graph(g_test)
    model.eval()
    print("TEST")
    for tsample in tqdm(testset):
        tdata = tsample.ndata["x"].transpose(0,1)
        x0t = tdata[0,:,:]
        #print(x0t.shape)
        predt = odeint(model,x0t,tb,method="rk4")
        loss = a*(loss_fn(predt[:,:,0:2],tdata[:,:,0:2]) + loss_fn(predt[:,:,2:4],tdata[:,:,2:4]))
        lossts_acc+=loss.item()
    dict={}
    for name,param in model.named_parameters():
        #dict[name]=torch.mean(param.data.flatten())
        dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()
    print("epoch: {}, train loss: {}, test_loss: {}".format(epoch+1,losstr_acc/len(trainset),lossts_acc/len(testset)))
    print(dict)
model.change_graph(g)    
pred = odeint(model,eval[0,:,:],t,method="rk4")
phasespace_show_threebody_pred(eval,pred)
