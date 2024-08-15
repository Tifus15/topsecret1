import torch 
from utils import *
from tqdm import tqdm
from GRUGNN_model import *
#from utils import open_yamls, make_dataset, make_snapshots, visualize_loss, viz_traj,visualize_hamiltonian,convert2dgl_snapshots, make_graph_dataset,make_graph_snapshots,dof2_hamiltonian,dof3_hamiltonian
#from device_util import *
import random
#from torch.utils.data import DataLoader
#import wandb
#from torchdiffeq import odeint_adjoint as odeint
#import time
#import dgl
from dgl.dataloading import GraphDataLoader
#from new_body import src_list, dst_list



def train6body(configs):

    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this

    WANDB = False

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    SINGLE = configs["single"]

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "Nbody"
    print(EPOCHS)
    
    t = torch.linspace(0,5.11,512)[0:128]
    H = torch.load("H6b1000.pt").transpose(0,1).unsqueeze(-1).unsqueeze(-1)
    x = torch.load("x6b1000.pt").transpose(0,1)
    dx = torch.load("dx6b1000.pt").transpose(0,1)
    temp = H
    for i in range(5):
        
        H = torch.cat((H,temp),dim=2)


    data = torch.cat((x,dx,H),dim=-1)

    print(H.shape)
    print(x.shape)
    print(dx.shape)

    print(src_list(6))
    print(dst_list(6))
    src = [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5] #without self loops
    dst = [ 1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 5, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4] # without self loops
    graph = dgl.graph((src,dst))
    dim = 4



    if SINGLE:
        num = random.randint(0,data.shape[1]-1)
        temp = data[:,num,:,:].unsqueeze(1)
        eval = data[:,num,:,:]
        H = eval[:,:,-1]
        data = temp
    else:
        #num = random.randint(0,dataset.shape[1]-1)
        eval = data[:,-1,:,:].unsqueeze(1)
        H = data[:,-1,:,-1]
        data = data[:,:-1,:,:]
    print(data.shape)

    print(data.shape)

    snapshots = make_snapshots(data,TIME_SIZE) # just 128 to keep everything in 2^i
    random.shuffle(snapshots)
    print(snapshots[0].shape)

    #graph_snapshots = make_graph_snapshots(snapshots,nodes=6,feats=4)

    dgl_snapshots = convert2dgl_snapshots(snapshots,src,dst)
    print(len(dgl_snapshots))
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = rollout_GNN_GRU(graph,4,MODEL_SIZE,MODEL_SIZE,ACT_FUNC,type=MODEL)

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
           # print(x.shape)
           # print(dx.shape)
           # print(x0.shape)
            predy = model(ts,x0)
            y = predy[:,:,0:4]
            dy = predy[:,:,4:8]
           # print(y.shape)
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
        test_loader=GraphDataLoader(test,batch_size=BATCH_SIZE,shuffle=True)
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
            yt = predyt[:,:,0:4]
            dyt = predyt[:,:,4:8]
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
    torch.save(model.state_dict(), "server_6body.pth")

def train7body(configs):
    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this

    WANDB = False

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    SINGLE = configs["single"]

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "Nbody"
    print(EPOCHS)
    
    t = torch.linspace(0,5.11,512)[0:128]
    H = torch.load("H7b1000.pt").transpose(0,1).unsqueeze(-1).unsqueeze(-1)
    x = torch.load("x7b1000.pt").transpose(0,1)
    dx = torch.load("dx7b1000.pt").transpose(0,1)
    temp = H
    for i in range(6):
        
        H = torch.cat((H,temp),dim=2)


    data = torch.cat((x,dx,H),dim=-1)

    print(H.shape)
    print(x.shape)
    print(dx.shape)

    print(src_list(7))
    print(dst_list(7))
    src = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6] #without self loops
    dst = [1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6, 0, 1, 3, 4, 5, 6, 0, 1, 2, 4, 5, 6, 0, 1, 2, 3, 5, 6, 0, 1, 2, 3, 4, 6, 0, 1, 2, 3, 4, 5] # without self loops
    graph = dgl.graph((src,dst))
    dim = 4


    if SINGLE:
        num = random.randint(0,data.shape[1]-1)
        temp = data[:,num,:,:].unsqueeze(1)
        eval = data[:,num,:,:]
        H = eval[:,:,-1]
        data = temp
    else:
        #num = random.randint(0,dataset.shape[1]-1)
        eval = data[:,-1,:,:].unsqueeze(1)
        H = data[:,-1,:,-1]
        data = data[:,:-1,:,:]
    print(data.shape)

    print(data.shape)

    snapshots = make_snapshots(data,TIME_SIZE) # just 128 to keep everything in 2^i
    random.shuffle(snapshots)
    print(snapshots[0].shape)

    #graph_snapshots = make_graph_snapshots(snapshots,nodes=6,feats=4)

    dgl_snapshots = convert2dgl_snapshots(snapshots,src,dst)
    print(len(dgl_snapshots))
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = rollout_GNN_GRU(graph,4,MODEL_SIZE,MODEL_SIZE,ACT_FUNC,type=MODEL)
    model = load_model(model,"server_6body.pth")

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
           # print(x.shape)
           # print(dx.shape)
           # print(x0.shape)
            predy = model(ts,x0)
            y = predy[:,:,0:4]
            dy = predy[:,:,4:8]
           # print(y.shape)
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
        test_loader=GraphDataLoader(test,batch_size=BATCH_SIZE,shuffle=True)
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
            yt = predyt[:,:,0:4]
            dyt = predyt[:,:,4:8]
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
    torch.save(model.state_dict(), "server_7body.pth")
    
train6body(configs)
train7body(configs)