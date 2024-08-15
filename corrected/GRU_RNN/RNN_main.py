import torch 
from tqdm import tqdm
from rnn_model import *
from utils import open_yamls, make_dataset, make_snapshots, visualize_loss, viz_traj,visualize_hamiltonian
from device_util import *
import random
from torch.utils.data import DataLoader
import wandb
from torchdiffeq import odeint_adjoint as odeint
import time

#SOB = True  # sobolev - gradients training
#s_alpha = 0.01
alpha = 0.01
OPTI = "adamW" # ["adamW","RMS","SGD"]
LOSS = "Huber" # ["MSE","MAE","Huber"]
REG = "none" #["lasso","ridge","none"]
#ACT_FUNC = ["tanh","relu",""] # activations - don't touch this

WANDB = False

MODEL_SIZE = 128
#DATASETSIZE = 512
SINGLE = True

EPOCHS = 10
BATCH_SIZE = 16
TIME_SIZE = 64
LR = 1e-3
SPLIT = 0.9
DATASET = "twobody"#["osci","twobody","threebody"]

if DATASET=="osci":
    DICT = open_yamls("oscilator.yaml")
if DATASET=="twobody":
    DICT = open_yamls("twobody.yaml")
if DATASET == "threebody":
    DICT = open_yamls("threebody.yaml")

dim = DICT["dim"]

dataset,ddataset, t, H,func_ham = make_dataset(DICT)
#print(H)
time.sleep(5.0)

if SINGLE:
    num = random.randint(0,dataset.shape[1]-1)
    dataset = dataset[:,num,:,:].unsqueeze(1)
    ddataset = ddataset[:,num,:,:].unsqueeze(1)
    eval = dataset.reshape(-1,1,dim)
    H = H[:,num]
else:
    num = random.randint(0,dataset.shape[1]-1)
    eval = dataset[:,num,:,:]
    H = H[:,num]

data = torch.cat((dataset,ddataset,H.reshape(dataset.shape[0],-1,1,1)),dim=-1)
print(data.shape)
snapshots = make_snapshots(data,TIME_SIZE)
ts = t[0:TIME_SIZE]
model = RNN_ODEINT(dim,dim,MODEL_SIZE)

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
        name_wandb = DATASET + "_RNN"
        wandb.init(project=name_wandb,config={
                    "epochs": EPOCHS,
                    "model": "RNN",
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "time_size":TIME_SIZE,
                    "optimizer": OPTI,
                    "loss": LOSS})
        metrics={"train_loss":0, "train_grad":0, "test_loss" :0, "test_grad":0}
        

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
        y = model(y0.reshape(1,batch.shape[1],-1),ts)
        #y = odeint(model,y0,ts,method="rk4")
        #print(y.shape)
        #print(batch[0:dim].shape)
        loss = lossfn(y,batch[:,:,:,0:dim].squeeze())
        if REG == "ridge":
            loss+= 0.01 * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= 0.01 * sum(p.abs().sum() for p in model.parameters())
        
        container[0,epoch] += loss.item()
        loss.backward()
        opti.step()
    
    container[0,epoch] /= N_train
    
    model.eval()
    
    test_loader = DataLoader(test,1,shuffle=False)
    N_test = len(test_loader)
    
    for batch in test_loader:
        batch=batch.transpose(0,1)
        y0t = batch[0,:,:,0:dim]
        #yt = odeint(model,y0t,ts,method="rk4")
        yt = model(y0t.reshape(1,batch.shape[1],-1),ts)
        loss = lossfn(yt.squeeze(),batch[:,:,:,0:dim].squeeze())
        if REG == "ridge":
            loss+= alpha * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= alpha * sum(p.abs().sum() for p in model.parameters())
        
        container[1,epoch]+=loss.item()
    
    
    container[1,epoch] /= N_test
    if WANDB:
        
        metrics["train_loss"] = container[0,epoch]
        metrics["test_loss"] = container[1,epoch]
        
        wandb.log(metrics)  
 
    print("Epoch: {}\nLOSS: train: {:.6f}    |   test: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch]))
    
if WANDB:
    wandb.finish()
    
if not SINGLE:
    visualize_loss("Loss at "+DATASET,container)
else: 
    visualize_loss("Singular loss at "+DATASET,container)
y0 = eval[0,:,0:dim]
y_eval = model(y0.reshape(1,batch.shape[1],-1),t)
print(y_eval.shape)
print(eval.shape)

viz_traj(eval.squeeze(),y_eval.squeeze(),DATASET)

H_eval = func_ham(y_eval.unsqueeze(1))

print("H {}".format(H.shape))
print("eval H {}".format(H_eval.shape))

#print("H {}".format(H))
#print("eval H {}".format(H_eval))

visualize_hamiltonian(H.squeeze(),H_eval.squeeze(),t)