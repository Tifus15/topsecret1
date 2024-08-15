import torch 
from tqdm import tqdm
from GNN_model import *
from utils import open_yamls, make_dataset, make_snapshots, visualize_loss, viz_traj,visualize_hamiltonian, make_graph_dataset,make_graph_snapshots,dof2_hamiltonian,dof3_hamiltonian
from device_util import *
import random
from torch.utils.data import DataLoader
import wandb
import dgl

MODEL = "REC" #["GCN","GAT","REC"]
OPTI = "adamW" # ["adamW","RMS","SGD"]
LOSS = "Huber" # ["MSE","MAE","Huber"]
REG = "none" #["lasso","ridge","none"]
ACT_FUNC = ["tanh","relu",""] # activations - don't touch this
alpha =0.01
WANDB = False

MODEL_SIZE = 128
#DATASETSIZE = 512
SINGLE = True

EPOCHS = 3000
BATCH_SIZE = 8
TIME_SIZE = 2
LR = 1e-4
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
#print(eval.shape)
#print(H.shape)
#print(H[0:10])

#data = torch.cat((dataset,ddataset,H.reshape(dataset.shape[0],-1,1,1)),dim=-1)
#print(data.shape)

snapshots = make_snapshots(data,TIME_SIZE)
random.shuffle(snapshots)
graph_snapshots = make_graph_snapshots(snapshots,DICT["nodes"],DICT["dim"]) # needs to be automatic
ts = t[0:TIME_SIZE]
print(len(graph_snapshots))
half = int(DICT["dim"]/DICT["nodes"])  

model = GNN_rollout(graph,half,MODEL_SIZE,half, ACT_FUNC,type=MODEL)
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


train = graph_snapshots[0:int(SPLIT*len(graph_snapshots))]
test = graph_snapshots[int(SPLIT*len(graph_snapshots)):]

if WANDB:
        #wandb.login()
        name_wandb = DATASET + "_GNN_traj"
        wandb.init(project=name_wandb,config={
                    "epochs": EPOCHS,
                    "model": MODEL,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "time_size":TIME_SIZE,
                    "optimizer": OPTI,
                    "loss": LOSS})
        metrics={"train_loss":0, "test_loss" :0}
        

container = torch.zeros(2,EPOCHS)
if WANDB:
    wandb.watch(model,log='all')
   
  
for epoch in tqdm(range(EPOCHS)):
    ## TRAINING
    model.train()
    
    train_loader = DataLoader(train,BATCH_SIZE,shuffle=True)
    N_train = len(train_loader)
    for batch in train_loader:
        #print(batch.shape)
        opti.zero_grad()
        batch = batch.transpose(0,1)
        x = batch[:,:,:,0:half]
        x0 = x[:,:,0,:]
        #print(x.shape)
        #print(x0.shape)
        y = model(ts,x0)
        #print(x.shape)
        #print(y.shape)
        
        loss = lossfn(y,x)
        if REG == "ridge":
            loss+= alpha* sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= alpha * sum(p.abs().sum() for p in model.parameters())
        
        container[0,epoch] += loss.item()
        loss.backward()
        opti.step()
        
    container[0,epoch] /= N_train
    model.eval()
    test_loader = DataLoader(test,1,shuffle=False)
    N_test = len(test_loader)
    for batch in test_loader:
        #print(batch.shape)
        batch=batch.transpose(0,1)
        
        xt = batch[:,:,:,0:half]
        #print(xt.shape)
        xt0 = xt[:,:,0,:]
        #yt = rollout_mlp(model,y0t,ts)
        yt = model(ts,xt0)
        loss = lossfn(yt,xt)
        if REG == "ridge":
            loss+= alpha * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= alpha * sum(p.abs().sum() for p in model.parameters())
        
        container[1,epoch] += loss.item()
        
    container[1,epoch] /= N_test
    if WANDB:
        metrics["train_loss"] = container[0,epoch]
        metrics["test_loss"] = container[1,epoch]
        wandb.log(metrics)  
    print("Epoch: {}\nLOSS: train: {:.6f}    |   test: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch]))


visualize_loss("loss of GNN",container)
eval_graph = make_graph_snapshots([eval],DICT["nodes"],DICT["dim"])
print(len(eval_graph))
print(eval_graph[0].shape)
ev = model(t,eval_graph[0][:,0,0:half])
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
