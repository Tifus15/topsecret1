import torch 
from tqdm import tqdm
from GNNODE_model import *
from utils import open_yamls, make_dataset, make_snapshots, visualize_loss, viz_traj,visualize_hamiltonian, make_graph_dataset,make_graph_snapshots,dof2_hamiltonian,dof3_hamiltonian
from device_util import *
import random
from torch.utils.data import DataLoader
import wandb
from torchdiffeq import odeint_adjoint as odeint
import time
import dgl

MODEL = "GCN" #["GCN","GAT","REC"]
SOB = True # sobolev - gradients training
s_alpha = 0.01
alpha = 0.01
OPTI = "adamW" # ["adamW","RMS","SGD"]
LOSS = "Huber" # ["MSE","MAE","Huber"]
REG = "none" #["lasso","ridge","none"]
ACT_FUNC = ["tanh","relu",""] # activations - don't touch this

WANDB = False

MODEL_SIZE = 128
#DATASETSIZE = 512
SINGLE = True

EPOCHS = 100
BATCH_SIZE = 32
TIME_SIZE = 64
LR = 1e-3
SPLIT = 0.9
DATASET = "dof3"#["dof2","dof3"]


if DATASET=="dof2":
    DICT = open_yamls("dof2pendelum.yaml")
    t = torch.linspace(0,2.5,251)
if DATASET=="dof3":
    DICT = open_yamls("dof3pendelum.yaml")
    t = torch.linspace(0,2.5,251)
graph = dgl.graph((DICT["src"],DICT["dst"]))
dim = DICT["dim"]

data=make_graph_dataset(DICT)
if SINGLE:
    num = random.randint(0,data.shape[1]-1)
    temp = data[:,num,:,:].unsqueeze(1)
    eval = data[:,num,:,:]
    H = eval[:,:,-1]
    data = temp
else:
    num = random.randint(0,data.shape[1]-1)
    eval = data[:,num,:,:]
    H = data[:,num,:,-1]
print(data.shape)
snapshots = make_snapshots(data,TIME_SIZE)
random.shuffle(snapshots)
graph_snapshots = make_graph_snapshots(snapshots,DICT["nodes"],DICT["dim"]) # needs to be automatic
ts = t[0:TIME_SIZE]
half = int(DICT["dim"]/DICT["nodes"]) 
model = GNN_maker(graph,half,MODEL_SIZE,half, ACT_FUNC,bias=True,type=MODEL)

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


train = graph_snapshots[0:int(SPLIT*len(graph_snapshots))]
test = graph_snapshots[int(SPLIT*len(graph_snapshots)):]

if WANDB:
        #wandb.login()
        name_wandb = DATASET + "_GNNODE"
        wandb.init(project=name_wandb,config={
                    "epochs": EPOCHS,
                    "model": "GNN",
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "time_size":TIME_SIZE,
                    "optimizer": OPTI,
                    "loss": LOSS})
        metrics={"train_loss":0, "train_grad":0, "test_loss" :0, "test_grad":0}
        

container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]
if WANDB:
    wandb.watch(model,log='all')
for epoch in tqdm(range(EPOCHS)):
    model.train()
    
    train_loader = DataLoader(train,BATCH_SIZE,shuffle=True)
    N_train = len(train_loader)
    for batch in train_loader:
        opti.zero_grad()
        loss=0
        loss2=0
        batch = batch.transpose(0,1)
        #print(batch.shape)
        x = batch[:,:,:,0:half]
        #print(x.shape)
        dx = batch[:,:,:,half:2*half]
        #print(dx.shape)
        x0 = x[:,:,0,:]
        #print(x0.shape)
        y = odeint(model,x0,ts,method="rk4").transpose(0,1).transpose(1,2)
        #print(y.shape)
        loss = lossfn(y,x)
        if REG == "ridge":
            loss+= alpha * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= alpha * sum(p.abs().sum() for p in model.parameters())
        dy = model(0,y)
        loss2 = lossfn(dy,dx)
        container[1,epoch] += loss2.item()
        if SOB:
            loss+= s_alpha * loss2
            
        container[0,epoch] += loss.item()
    loss.backward()
    opti.step()
    
    container[0:2,epoch] /= N_train
    model.eval()
    test_loader = DataLoader(test,1,shuffle=False)
    N_test = len(test_loader)
    for batch in test_loader:
        loss=0
        loss2=0
        batch = batch.transpose(0,1)
        xt = batch[:,:,:,0:half]
        dxt = batch[:,:,:,half:2*half]
        x0t = xt[:,:,0,:]
        yt = odeint(model,x0t,ts,method="rk4").transpose(0,1).transpose(1,2)
        loss= lossfn(yt,xt)
        if REG == "ridge":
            loss+= 0.01 * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= 0.01 * sum(p.abs().sum() for p in model.parameters())
        
        dyt = model(0,yt)
        loss2 = lossfn(dyt,dxt)
        container[3,epoch] += loss2.item()
        if SOB:
            loss+= s_alpha * loss2
        container[2,epoch] += loss.item()
    
    container[2:4,epoch] /= N_test
    if WANDB:
        metrics["train_loss"] = container[0,epoch]
        metrics["train_grad"] = container[1,epoch]
        metrics["test_loss"] = container[2,epoch]
        metrics["test_grad"] = container[3,epoch]
        wandb.log(metrics)  
    print("Epoch: {}\nLOSS: train: {:.6f} grad: {:.6f}   |   test: {:.6f} grad: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
if WANDB:
    wandb.finish()

visualize_loss("loss of GNNODE",container)
eval_graph = make_graph_snapshots([eval],DICT["nodes"],DICT["dim"])
print(len(eval_graph))
print(eval_graph[0].shape)
ev = odeint(model,eval_graph[0][:,0,0:half],t,method="rk4").transpose(0,1)
print(ev.shape)
viz_traj(eval_graph[0],ev,DATASET)
if DATASET == "dof2":
    h = dof2_hamiltonian(ev)
elif DATASET == "dof3":
    h = dof3_hamiltonian(ev)
H = eval_graph[0][0,:,-1]
print(h.shape)
print(H.shape)
visualize_hamiltonian(H,h,t)
    
