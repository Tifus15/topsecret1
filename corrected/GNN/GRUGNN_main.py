import torch 
from tqdm import tqdm
from GRUGNN_model import *
from utils import open_yamls, make_dataset, make_snapshots, visualize_loss, viz_traj,visualize_hamiltonian,convert2dgl_snapshots, make_graph_dataset,make_graph_snapshots,dof2_hamiltonian,dof3_hamiltonian
from device_util import *
import random
from torch.utils.data import DataLoader
import wandb
from torchdiffeq import odeint_adjoint as odeint
import time
import dgl
from dgl.dataloading import GraphDataLoader

MODEL = "GCN" #["GCN","GAT"]
SOB = True  # sobolev - gradients training
s_alpha = 1.0
alpha = 0.01
OPTI = "adamW" # ["adamW","RMS","SGD"]
LOSS = "Huber" # ["MSE","MAE","Huber"]
REG = "none" #["lasso","ridge","none"]
ACT_FUNC = ["tanh",""] # activations - don't touch this

WANDB = False

MODEL_SIZE = 256
#DATASETSIZE = 512
SINGLE = True

EPOCHS = 500
BATCH_SIZE = 8
TIME_SIZE = 64
LR = 5e-4
SPLIT = 0.9
DATASET = "dof3"#["dof2","dof3"]


if DATASET=="dof2":
    DICT = open_yamls("dof2pendelum.yaml")
    t = torch.linspace(0,2.55,256)[0:128]
if DATASET=="dof3":
    DICT = open_yamls("dof3pendelum.yaml")
    t = torch.linspace(0,2.55,256)[0:128]
graph = dgl.graph((DICT["src"],DICT["dst"]))
dim = DICT["dim"]

data=make_graph_dataset(DICT)
if SINGLE:
    num = random.randint(0,data.shape[1]-1)
    temp = data[0:128,num,:,:].unsqueeze(1)
    eval = data[0:128,num,:,:]
    H = eval[0:128,:,-1]
    data = temp
else:
    #num = random.randint(0,dataset.shape[1]-1)
    eval = data[0:128,-1,:,:].unsqueeze(1)
    H = data[0:128,-1,:,-1]
    data = data[0:128,:-1,:,:]
print(data.shape)

print(data.shape)
snapshots = make_snapshots(data,TIME_SIZE) # just 128 to keep everything in 2^i
random.shuffle(snapshots)

graph_snapshots = make_graph_snapshots(snapshots,DICT["nodes"],DICT["dim"])

dgl_snapshots = convert2dgl_snapshots(graph_snapshots,DICT["src"],DICT["dst"])
print(len(dgl_snapshots))
ts = t[0:TIME_SIZE]

half = int(DICT["dim"]/DICT["nodes"]) 
model = rollout_GNN_GRU(graph,half,MODEL_SIZE,MODEL_SIZE,ACT_FUNC,type=MODEL)

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


train = dgl_snapshots[0:int(SPLIT*len(dgl_snapshots))]
test = dgl_snapshots[int(SPLIT*len(dgl_snapshots)):]

if WANDB:
        #wandb.login()
        name_wandb = DATASET + "_GRUGNN"
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
        train_loader=GraphDataLoader(train, batch_size=BATCH_SIZE,shuffle=True)
        N_train = len(train_loader)
        print("TRAIN")
        for train_sample in tqdm(train_loader):
            loss=0
            opti.zero_grad()
            model.change_graph(train_sample)
            x = train_sample.ndata["x"].transpose(0,1)
            dx = train_sample.ndata["dx"].transpose(0,1)
            x0 = x[0,:,:]
            #print(x.shape)
            #print(dx.shape)
            #print(x0.shape)
            predy = model(ts,x0)
            y = predy[:,:,0:2]
            dy = predy[:,:,2:4]
            #print(y.shape)
            loss = lossfn(y,x)
            if REG == "ridge":
                loss += alpha * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            loss2 = lossfn(dy,dx)
            if SOB:
                loss += s_alpha * loss2
            container[1,epoch]+=loss2.item()
            container[0,epoch] += loss.item()
            loss.backward()
            opti.step()
        container[0:2,epoch]/=N_train
        print("TEST")
        model.eval()
        test_loader=GraphDataLoader(test,batch_size=1,shuffle=True)
        N_test = len(test_loader)
        for test_sample in tqdm(test_loader):
            testloss = 0
            model.change_graph(test_sample)
            xt = test_sample.ndata["x"].transpose(0,1)
            dxt = test_sample.ndata["dx"].transpose(0,1)
            x0t = xt[0,:,:]
            #print(xt.shape)
            #print(dxt.shape)
            #print(x0t.shape)
            predyt = model(ts,x0t)
            #print(predyt.shape)
            yt = predyt[:,:,0:2]
            dyt = predyt[:,:,2:4]
            #print(yt.shape)
            #print(x0t.shape)
            testloss = lossfn(yt,xt)
            if REG == "ridge":
                testloss += alpha * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                testloss += alpha * sum(p.abs().sum() for p in model.parameters())
            testloss2 = lossfn(dyt,dxt)
            if SOB:
                testloss += s_alpha * testloss2
            container[3,epoch]+=testloss2.item()
            container[2,epoch] += testloss.item()
        container[2:4,epoch]/=N_test
        if WANDB:
            metrics["train_loss"] = container[0,epoch]
            metrics["test_loss"] = container[2,epoch]
            metrics["train_grad_loss"] = container[1,epoch]
            metrics["test_grad_loss"] = container[3,epoch]
            wandb.log(metrics)
            #wandb.log_artifact(model)
            
        print("Epoch: {}\nLOSS: train: {:.6f} grad: {:.6f}   |   test: {:.6f} grad: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
if WANDB:
    wandb.finish()
    
visualize_loss("loss of GNN",container)
eval_graph = make_graph_snapshots([eval[0:128,:,:]],DICT["nodes"],DICT["dim"])
print(len(eval_graph))
eval_dgl = convert2dgl_snapshots(eval_graph,DICT["src"],DICT["dst"])[0]
x = eval_dgl.ndata["x"]
print("X {}".format(x.shape))
model.reset_graph()
print(t)
t=t[0:128]
ev = model(t,x[:,0,:]).transpose(0,1)
print("ev {}".format(ev.shape))
viz_traj(x,ev,DATASET,t)
if DATASET == "dof2":
    h = dof2_hamiltonian(ev)
elif DATASET == "dof3":
    h = dof3_hamiltonian(ev)
H = eval_dgl.ndata["H"][0,:]
print(h.shape)
print(H.shape)
visualize_hamiltonian(H,h,t[0:128])
    
    