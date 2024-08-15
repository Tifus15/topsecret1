import torch
import torch.optim as opti
from oscilator import *
from visualize import *
from utils import *
from torch.utils.data import DataLoader
from model import *
import random
from tqdm import tqdm
maker = oscilator(1,0.2)
from torchdiffeq import odeint_adjoint as odeint

data, ddata, t, H = maker.make_dataset(128,100)

data=data.cpu()
ddata=ddata.cpu()
t=t.cpu()
H=H.cpu()




eval = data[:,99,:,:]

phasespace_show(eval)
#ham_show(t,H[:,100])



TIME_SNAP = 2
BATCH_SIZE = 32
EPOCHS = 2000
dataset = snap_maker(data[:,0:99,:,:],TIME_SNAP)
random.shuffle(dataset)

snap_train = dataset[0:int(len(dataset)*0.9)]
snap_test = dataset[int(len(dataset)*0.9):]

model = mlp(acts=["tanh",""],in_dim=2,out_dim=2,hidden=[8,8])
print(model)
tb = t[0:TIME_SNAP]
optimizer = opti.RMSprop(model.parameters(),lr=1e-4)

trainset = DataLoader(snap_train,BATCH_SIZE,True)
testset = DataLoader(snap_test)
loss_fn = nn.MSELoss()
loss_r = nn.L1Loss()

for epoch in tqdm(range(EPOCHS)):
    losstr_acc=0
    lossts_acc=0
    for sample in tqdm(trainset):
        optimizer.zero_grad()
        #print("loop")
        sample =sample.transpose(0,1)
        #print(sample.shape)
        y = odeint(model,sample[0,:,:,:],tb,method="rk4",rtol=1e-6,atol=1e-5)
        loss = 100*(loss_fn(y[:,:,:,0],sample[:,:,:,0]) + loss_fn(y[:,:,:,1],sample[:,:,:,1])) +0.01*loss_r(y,sample)
        loss.backward()
        
        optimizer.step()
        losstr_acc+=loss.item()
        
    for tsample in tqdm(testset):
        tsample =tsample.transpose(0,1)
        #print(tsample.shape)
        yt = odeint(model,tsample[0,:,:,:],tb,method="rk4",rtol=1e-6,atol=1e-6)
        loss = 100*(loss_fn(yt[:,:,:,0],tsample[:,:,:,0]) + loss_fn(yt[:,:,:,1],tsample[:,:,:,1]))+0.01*loss_r(yt,tsample)
        lossts_acc+=loss.item()
    dict={}
    for name,param in model.named_parameters():
        #dict[name]=torch.mean(param.data.flatten())
        dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()
    pred = odeint(model,eval[0,:,:],t,method="rk4")   
    a = acc_pos(eval,pred,eps=0.05)
    
    print("epoch: {}, train loss: {}, test_loss: {}, acc: {}".format(epoch+1,losstr_acc/len(trainset),lossts_acc/len(testset),a))
    if a == 0.95:
        break
    torch.save(model.state_dict(),"perfect_model_osci.pth")
        
    print(dict)

model_phasespace_show(eval,pred)

