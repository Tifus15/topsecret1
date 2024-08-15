import torch 
from tqdm import tqdm
from mlp import *
from utils import open_yamls, make_dataset, make_snapshots, visualize_loss, viz_traj,visualize_hamiltonian
from device_util import *
import random
from torch.utils.data import DataLoader
import wandb
from torchdiffeq import odeint_adjoint as odeint
import time

SOB = True  # sobolev - gradients training
s_alpha = 0.01
alpha = 0.01
OPTI = "adamW" # ["adamW","RMS","SGD"]
LOSS = "Huber" # ["MSE","MAE","Huber"]
REG = "none" #["lasso","ridge","none"]
ACT_FUNC = ["tanh","relu",""] # activations - don't touch this

WANDB = True

MODEL_SIZE = 256
#DATASETSIZE = 512
SINGLE = False

EPOCHS = 100
BATCH_SIZE = 32
TIME_SIZE = 32
LR = 1e-4
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
model = mlp(dim,MODEL_SIZE,dim, ACT_FUNC)

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
        name_wandb = DATASET + "_mlp_vec"
        wandb.init(project=name_wandb,config={
                    "epochs": EPOCHS,
                    "model": "MLP",
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "time_size":TIME_SIZE,
                    "optimizer": OPTI,
                    "loss": LOSS})
        metrics={"train_loss":0, "train_grad":0, "test_loss" :0, "test_grad":0}
        
if SOB:
    container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]
else:
    container = torch.zeros(2,EPOCHS)
if WANDB:
    wandb.watch(model)
for epoch in tqdm(range(EPOCHS)):
    ## TRAINING
    model.train()
    
    train_loader = DataLoader(train,BATCH_SIZE,shuffle=True)
    N_train = len(train_loader)
    for batch in train_loader:
        opti.zero_grad()
        batch = batch.transpose(0,1)
        y0 = batch[0,:,:,0:dim]
        y = odeint(model,y0,ts,method="rk4")
        #print(y.shape)
        #print(y.shape)
        loss = lossfn(y,batch[:,:,:,0:dim])
        if REG == "ridge":
            loss+= 0.01 * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= 0.01 * sum(p.abs().sum() for p in model.parameters())
        if SOB:
            dy = rollout_mlp_vec(model,y)
            #print(dy.shape)
            loss2 = s_alpha*lossfn(dy,batch[:,:,:,dim:2*dim])
            loss += loss2
            container[1,epoch] += loss2.item()
            container[0,epoch] += loss.item()
        else:    
            container[0,epoch] += loss.item()
        loss.backward()
        opti.step()
    if SOB:
        container[0:2,epoch]/=N_train
    else:
        container[0,epoch] /= N_train
    model.eval()
    test_loader = DataLoader(test,1,shuffle=False)
    N_test = len(test_loader)
    for batch in test_loader:
        batch=batch.transpose(0,1)
        y0t = batch[0,:,:,0:dim]
        yt = odeint(model,y0t,ts,method="rk4")
        loss = lossfn(yt,batch[:,:,:,0:dim])
        if REG == "ridge":
            loss+= alpha * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= alpha * sum(p.abs().sum() for p in model.parameters())
        if SOB:
            dyt = rollout_mlp_vec(model,yt)
            loss2 = s_alpha*lossfn(dyt,batch[:,:,:,dim:2*dim])
            loss += loss2
            container[3,epoch] += loss2.item()
            container[2,epoch] += loss.item()
        else:
            container[1,epoch]+=loss.item()
    if SOB:
        container[2:4,epoch]/=N_test
    else:
        container[1,epoch] /= N_test
    if WANDB:
        if not SOB:
            metrics["train_loss"] = container[0,epoch]
            metrics["test_loss"] = container[1,epoch]
        else:
            metrics["train_loss"] = container[0,epoch]
            metrics["train_grad"] = container[1,epoch]
            metrics["test_loss"] = container[2,epoch]
            metrics["test_grad"] = container[3,epoch]
        wandb.log(metrics)  
    if not SOB:
        print("Epoch: {}\nLOSS: train: {:.6f}    |   test: {:.6f}".format(epoch+1,container[0,epoch],container[2,epoch]))
    else:
        print("Epoch: {}\nLOSS: train: {:.6f} grad:{:.6f}   |   test: {:.6f} grad: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
if WANDB:
    wandb.finish()
if not SINGLE:
    visualize_loss("Loss at "+DATASET,container)
else: 
    visualize_loss("Singular loss at "+DATASET,container)
y0 = eval[0,:,0:dim]
y_eval = odeint(model,y0,t,method="rk4")
print(y_eval.shape)
viz_traj(eval.squeeze(),y_eval.squeeze(),DATASET)

H_eval = func_ham(y_eval.unsqueeze(1))

print("H {}".format(H.shape))
print("eval H {}".format(H_eval.shape))

#print("H {}".format(H))
#print("eval H {}".format(H_eval))

visualize_hamiltonian(H.squeeze(),H_eval.squeeze(),t)