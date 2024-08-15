import torch
from tqdm import tqdm
from hnn_model import *
from data_func import *
import dgl
from torch.utils.data import DataLoader
from torchdiffeq import odeint
import random
import matplotlib.pyplot as plt
src = src_list(4)
dst = dst_list(4)
graph = dgl.graph((src,dst))

fileH = "data/nbody_4_H.pt"
filet = "data/nbody_4_traj.pt"

x , H =load_dataset(filet,fileH)
eval = x[:,-1,:,:]
evalH = H[:,-1,:]
print(x.shape)
print(H.shape)
dt =0.01
T = 4

t = torch.linspace(0,4,401)
print(t[1]-t[0])
SPLIT = 0.9
TIMESIZE = 32
BATCHSIZE = 128
EPOCHS = 10
model = GNN_maker_HNN(graph,6,128,1,["tanh",""])
opti = torch.optim.AdamW(model.parameters(),lr=1e-4)
lossfn = torch.nn.HuberLoss()


xs, hs = make_snapshots(x[:,:-1,:,:].float(),H[:,:-1,:].float(),TIMESIZE)
print(xs[0].shape)
border = int(SPLIT*len(xs))

train = xs[0:border]
test = xs[border:]

trainH = hs[0:border]
testH = hs[border:]
ts = t[0:TIMESIZE]
a=[1.00,0.5]
trainloss = []
testloss = []


for epoch in tqdm(range(EPOCHS)):
    c = list(zip(train,trainH))
    random.shuffle(c)
    train, trainH = zip(*c)
    trainset = DataLoader(train,batch_size=BATCHSIZE)
    N_train=len(trainset)
    trainsetH = DataLoader(trainH,batch_size=BATCHSIZE)
    print("Training")
    ploss=0
    for sample, Hs in tqdm(zip(trainset,trainsetH)):
        opti.zero_grad()
        sample = sample.transpose(0,2).transpose(0,1)
        Hs = Hs.transpose(0,1)
        x0 = sample[0,:,:,:].requires_grad_()
        #h_hat = model.H(x0)
        x_hat = odeint(model,x0,ts,method="rk4")
        h_hat = model.H_rollout(sample)

        #print("hat {}".format(x_hat.shape))
        #print("sample {}".format(sample.shape))
        #print("hath {}".format(h_hat.shape))
        #print("Hs {}".format(Hs.shape))
        losst = lossfn(x_hat,sample)
        lossh = lossfn(h_hat,Hs)
        loss = a[0]*losst+a[1]*lossh
        p = model.parameters()
        ploss+=loss.item()
        
        loss.backward()
        opti.step()
    ploss/=N_train
    #print(ploss)

    print("TEST")
    testset = DataLoader(test,batch_size=BATCHSIZE)
    N_test=len(testset)
    testsetH = DataLoader(testH,batch_size=BATCHSIZE)
    tloss = 0
    for sample, Hs in tqdm(zip(testset,testsetH)):
        sample = sample.transpose(0,2).transpose(0,1)
        Hs = Hs.transpose(0,1)
        x0 = sample[0,:,:,:].requires_grad_()
        #h_hat = model.H(x0)
        x_hat = odeint(model,x0,ts,method="rk4")
        h_hat = model.H_rollout(sample)

        losst = lossfn(x_hat,sample)
        lossh = lossfn(h_hat,Hs)
        loss = a[0]*losst+a[1]*lossh
        tloss+=loss.item()

    tloss/=N_test
    #print(tloss)
    trainloss.append(ploss)
    testloss.append(tloss)


    print("EPOCH: {} trainloss: {} testloss: {}".format(epoch+1,ploss,tloss))
    
fig = plt.figure()
plt.semilogy(np.linspace(0,EPOCHS,EPOCHS),trainloss)
plt.semilogy(np.linspace(0,EPOCHS,EPOCHS),testloss)
plt.legend(["train loss", "test_loss"])
plt.show()







    














