import torch 
from tqdm import tqdm
from mlp import *
from utils import open_yamls, make_dataset, make_snapshots, visualize_loss, viz_traj,visualize_hamiltonian
from device_util import *
import random
from torch.utils.data import DataLoader
import wandb


OPTI = "adamW" # ["adamW","RMS","SGD"]
LOSS = "Huber" # ["MSE","MAE","Huber"]
REG = "none" #["lasso","ridge","none"]
ACT_FUNC = ["tanh","relu",""]#["tanh",""] # activations - don't touch this

WANDB = True

MODEL_SIZE = 256
#DATASETSIZE = 512
SINGLE = False

EPOCHS = 1000
BATCH_SIZE = 32
TIME_SIZE = 2
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

#model = mlp(dim,MODEL_SIZE,dim, ACT_FUNC)
model = mlp_rollout(dim,MODEL_SIZE,dim, ACT_FUNC)

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
        name_wandb = DATASET + "_mlp_traj"
        wandb.init(project=name_wandb,config={
                    "epochs": EPOCHS,
                    "model": "MLP",
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
        opti.zero_grad()
        batch = batch.transpose(0,1)
        y0 = batch[0,:,:,0:dim]
        #y = rollout_mlp(model,y0,ts)
        y = model(ts,y0)
        #print(y.shape)
        loss = lossfn(y,batch[:,:,:,0:dim])
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
        #yt = rollout_mlp(model,y0t,ts)
        yt = model(ts,y0t)
        loss = lossfn(yt,batch[:,:,:,0:dim])
        if REG == "ridge":
            loss+= 0.01 * sum(p.square().sum() for p in model.parameters())
        if REG == "lasso":
            loss+= 0.01 * sum(p.abs().sum() for p in model.parameters())
        
        container[1,epoch] += loss.item()
    
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
y_eval = model(t,eval[0,:,:])
print(y_eval.shape)
print(eval.shape)
viz_traj(eval,y_eval,DATASET)

H_eval = func_ham(y_eval.unsqueeze(1))

print(H.shape)
print(H_eval.shape)
print(H)
print(H_eval)

visualize_hamiltonian(H.squeeze(),H_eval.squeeze(),t)

    


