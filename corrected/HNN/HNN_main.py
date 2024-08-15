import torch 
from tqdm import tqdm
from hnn_model import *
from utils import open_yamls, make_dataset, make_snapshots, visualize_loss, viz_traj,visualize_hamiltonian
from device_util import *
import random
from torch.utils.data import DataLoader
import wandb
from torchdiffeq import odeint
import time

SOB = [True, True]  # sobolev - [gradients training, hamiltonian]
s_alpha = [1.00,1.00]#0.1
alpha = 0.01
OPTI = "adamW" # ["adamW","RMS","SGD"]
LOSS = "Huber" # ["MSE","MAE","Huber"]
REG = "none" #["lasso","ridge","none"]
ACT_FUNC = ["tanh",""] # activations - don't touch this

WANDB = True

MODEL_SIZE =512 # 1024
#DATASETSIZE = 512
SINGLE = False

EPOCHS = 100
BATCH_SIZE = 32
TIME_SIZE = 32
LR = 1e-3
SPLIT = 0.9
DATASET = "threebody"#["osci","twobody","threebody"]

if DATASET=="osci":
    DICT = open_yamls("oscilator.yaml")
if DATASET=="twobody":
    DICT = open_yamls("twobody.yaml")
if DATASET == "threebody":
    DICT = open_yamls("threebody.yaml")

dim = DICT["dim"]

dataset,ddataset, t, Ht,func_ham = make_dataset(DICT)
#print(H)
#time.sleep(10.0)

if SINGLE:
    num = random.randint(0,dataset.shape[1]-1)
    dataset = dataset[:,num,:,:].unsqueeze(1)
    ddataset = ddataset[:,num,:,:].unsqueeze(1)
    eval = dataset.reshape(-1,1,dim)
    H = Ht[:,num]
    data = torch.cat((dataset,ddataset,H.reshape(dataset.shape[0],-1,1,1)),dim=-1)
else:
    #num = random.randint(0,dataset.shape[1]-1)
    eval = dataset[:,-1,:,:]
    H = Ht[:,-1]
    data = torch.cat((dataset[:,:-1,:,:],ddataset[:,:-1,:,:],Ht[:,:-1].reshape(dataset.shape[0],-1,1,1)),dim=-1)
print(data.shape)
snapshots = make_snapshots(data,TIME_SIZE)
ts = t[0:TIME_SIZE]
model = HNN(dim,MODEL_SIZE,ACT_FUNC)
print(model)

if OPTI=="RMS":
    opti = torch.optim.RMSprop(model.parameters(),lr=LR)
if OPTI=="SGD":
    opti = torch.optim.SGD(model.parameters(),lr=LR)
if OPTI == "adamW":
    opti = torch.optim.AdamW(model.parameters(),lr=LR)

if LOSS == "MSE":
    lossfn = nn.MSELoss()
if LOSS == "MAE":
    lossfn = nn.L1Loss()
if LOSS == "Huber":
    lossfn = nn.HuberLoss()
random.shuffle(snapshots)

train = snapshots[0:int(SPLIT*len(snapshots))]
test = snapshots[int(SPLIT*len(snapshots)):]

if WANDB:
        #wandb.login()
        name_wandb = DATASET + "_HNN"
        wandb.init(project=name_wandb,config={
                    "epochs": EPOCHS,
                    "model": "MLP_HNN",
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "time_size":TIME_SIZE,
                    "optimizer": OPTI,
                    "loss": LOSS})
        metrics={"train_loss":0, "train_grad":0,"train_H":0, "test_loss" :0, "test_grad":0,"test_h":0}
        

container = torch.zeros(6,EPOCHS) #["train loss, train grad loss,train ham, test loss, test grad loss,test ham"] -even not used

    
if WANDB:
    wandb.watch(model)
print(container.shape)
for epoch in tqdm(range(EPOCHS)):
    ## TRAINING
    model.train()
    #print("   TRAINING   ")
    train_loader = DataLoader(train,BATCH_SIZE,shuffle=True)
    N_train = len(train_loader)
    for batch in train_loader:
        loss2 = 0
        loss3 = 0
        opti.zero_grad()
        batch = batch.transpose(0,1).requires_grad_()
        y0 = batch[0,:,:,0:dim]
        #print("here")
        model(0,y0)
        y = odeint(model,y0,ts,method="rk4")
        #print(y.shape)
        #print(y.shape)
        loss = lossfn(y[:,:,:,0:int(dim/2)],batch[:,:,:,0:int(dim/2)]) + lossfn(y[:,:,:,int(dim/2):],batch[:,:,:,int(dim/2):dim])
        if REG == "ridge":
            loss+= 0.01 * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= 0.01 * sum(p.abs().sum() for p in model.parameters())
        if SOB[0]:
            dy = rollout_mlp_vec(model,y)
            #print(dy.shape)
            loss2 = lossfn(dy[:,:,:,0:int(dim/2)],batch[:,:,:,dim:int(1.5*dim)]) + lossfn(dy[:,:,:,int(dim/2):],batch[:,:,:,int(1.5*dim):2*dim])
            #print("dx {},{}".format(dy.shape,batch[:,:,:,dim:2*dim].shape))
            loss += s_alpha[0]*loss2
            container[1,epoch] += loss2.item()
            #container[0,epoch] += loss.item()
        if SOB[1]:
            h = model.giveH(batch[:,:,:,0:dim])
            loss3 = lossfn(h,batch[:,:,:,-1]) 
            #print("h {},{}".format(h.shape,batch[:,:,:,-1].shape)) 
            container[2,epoch] += loss3.item()
            loss += s_alpha[1]*loss3
              
        container[0,epoch] += loss.item()
        
        loss.backward()
        opti.step()
        
    
    container[0:3,epoch]/=N_train
    
    model.eval()
    test_loader = DataLoader(test,1,shuffle=False)
    N_test = len(test_loader)
    #print("   TEST    ")
    for batcht in test_loader:
        loss2 = 0
        loss3 = 0
        batcht=batcht.transpose(0,1).requires_grad_()
        y0t = batcht[0,:,:,0:dim]
        yt = odeint(model,y0t,ts,method="rk4")
        loss = lossfn(yt[:,:,:,0:int(dim/2)],batcht[:,:,:,0:int(dim/2)]) + lossfn(yt[:,:,:,int(dim/2):],batcht[:,:,:,int(dim/2):dim])
        if REG == "ridge":
            loss+= alpha * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= alpha * sum(p.abs().sum() for p in model.parameters())
        if SOB[0]:
            dyt = rollout_mlp_vec(model,yt)
            #print("dx {},{}".format(dyt.shape,batcht[:,:,:,dim:2*dim].shape)) 
            loss2 = lossfn(dyt[:,:,:,0:int(dim/2)],batcht[:,:,:,dim:int(1.5*dim)]) + lossfn(dyt[:,:,:,int(dim/2):],batcht[:,:,:,int(1.5*dim):2*dim])
            loss += s_alpha[0]*loss2
            container[4,epoch] += loss2.item()
            #
        if SOB[1]:
            #container[1,epoch]+=loss.item()
            h = model.giveH(yt)
            #print("h {}".format(batcht[:,:,:,-1])) 
            loss3 = lossfn(h,batcht[:,:,:,-1])  
            container[5,epoch] += loss3.item()
            loss += s_alpha[1]*loss3
        container[3,epoch] += loss.item()
    
    container[3:6,epoch]/=N_test
    if WANDB:
            metrics["train_loss"] = container[0,epoch]
            metrics["train_grad"] = container[1,epoch]
            metrics["train_H"] = container[2,epoch]
            metrics["test_loss"] = container[3,epoch]
            metrics["test_grad"] = container[4,epoch]
            metrics["test_H"] = container[5,epoch]
            wandb.log(metrics)  
    print("Epoch: {}\nLOSS: train: {:.6f} grad:{:.6f} h:{:.6f}   |   test: {:.6f} grad: {:.6f} h:{:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch],container[4,epoch],container[5,epoch]))
if WANDB:
    wandb.finish()
if not SINGLE:
    visualize_loss("Loss at "+DATASET,container)
else: 
    visualize_loss("Singular loss at "+DATASET,container)
y0 = eval[0,:,0:dim].requires_grad_()
y_eval = odeint(model,y0,t,method="rk4")
print(y_eval.shape)
viz_traj(eval.squeeze(),y_eval.squeeze(),DATASET)

H_eval = model.giveH(y_eval)

print("H {}".format(H.shape))
print("eval H {}".format(H_eval.shape))

print("H {}".format(H))
print("eval H {}".format(H_eval))

visualize_hamiltonian(H.squeeze(),H_eval.squeeze(),t)