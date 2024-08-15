import torch
import dgl
from utils import *
import random
import yaml
from HGNN import *
import wandb
from tqdm import tqdm

from dgl.dataloading import GraphDataLoader
#from torch_symplectic_adjoint import odeint
from torchdiffeq import odeint
import os
def full(configs):
    print("begin 1dof")
    train1dof(configs)
    print("end 1dof")
    print("begin 2dof")
    train2dof(configs)
    print("end 2dof")
    print("begin 3dof")
    train3dof(configs)
    print("end 3dof")
    print("begin 4dof")
    train4dof(configs)
    print("end 4dof")
    
def train1dof(configs):
    

    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    NOLOOPS = configs["noloops"]
    WANDB = True
    BIAS = configs["bias"]
    S = configs["samples"]
    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "1dof pendelum"
    print(EPOCHS)
    DOF = 1
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
    data = torch.load("traj_1dof.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(1)
    dst = dst_list(1)
    if NOLOOPS and DOF != 1 :
        src,dst = del_loops(src,dst)
        
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    
    
    model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias=BIAS)
    print(model)
    
    
    data[:,:,:,0] = angle_transformer(data[:,:,:,0])
    
    
    #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = eval[:,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)

    x_temp = data[:,:,:,0:4]
    H_temp = data[:,:,:,-1]
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    

    train_snapshots = create_pend1dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend1dof_graph_snapshots(test,h_test,src,dst)
    #graph_snapshots = make_graph_snapshots(snapshots,nodes=6,feats=4)

    #dgl_snapshots = convert2dgl_snapshots(snapshots,src,dst)
    

 
    ts = t[0:TIME_SIZE]

   
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

    
    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if DOF != 1:
            src, dst = make_graph_no_loops(1,0)
        else:
            src = src_list(1)
            dst = dst_list(1)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_loss_d1":0,  "train_H_d1":0, "test_loss_d1" :0, "test_H_d1" :0}
        

    container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            h_tr = train_sample.ndata["h"].transpose(0,1)
            x0 = x_tr[0,:,:]
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            model.change_graph(roll_g)
            #print(roll_g)
            x_tr = x_tr.requires_grad_()
            x_tr_flat = x_tr.reshape(-1,2)
            #print(x_tr_flat.shape)
            h_pred = model(x_tr_flat)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            """
            if REG == "ridge":
                loss += alpha[0] * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            """
            x0 = x_tr[0,:,:].requires_grad_()
            model.change_graph(train_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            container[1,epoch]+=lossH.item()
            container[0,epoch] += loss.item()
            
            loss.backward()
            opti.step()
        container[0:2,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = test_sample.ndata["h"].transpose(0,1)
            model.change_graph(roll_g)
            x_ts = x_ts.requires_grad_()
            x_ts_flat = x_ts.reshape(-1,2)
            h_pred = model(x_ts_flat)
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())
            x0 = x_ts[0,:,:].requires_grad_()
            model.change_graph(test_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
        
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[0] * lossROLLt
                
           
            container[2,epoch]+=losst.item()
            container[3,epoch] += lossHt.item()
        container[2:4,epoch]/=N_test
    
        metrics["train_loss_d1"] = container[0,epoch]
        metrics["test_loss_d1"] = container[2,epoch]
        metrics["train_H_d1"] = container[1,epoch]
        metrics["test_H_d1"] = container[3,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
       
        print("Epoch: {}\nLOSS: train: {:.6f}   ham: {:.6f} |   test: {:.6f}  ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
        dict={}
        for namew , param in model.named_parameters():
            dict[namew+"_grad"] = torch.mean(param.grad).item()
        print(dict)
   
    
    visualize_loss("loss of 1dof pendelum",container)
    torch.save(model.state_dict(),"server_1dof.pth")
    #torch.save(model,"whole_model_dof1.pt")
    
def train2dof(configs):
    S = configs["samples"]
    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    NOLOOPS = configs["noloops"]
    WANDB = True
    BIAS = configs["bias"]

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
 

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "2dof pendelum"
    print(EPOCHS)
    
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
    data = torch.load("traj_2dof.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(2)
    dst = dst_list(2)
    if NOLOOPS:
        src,dst = del_loops(src,dst)
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:2] = angle_transformer(data[:,:,:,0:2])

    
        #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = eval[:,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)
    
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]

    
    xs, hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
  
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    
    train_snapshots = create_pend2dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend2dof_graph_snapshots(test,h_test,src,dst)
    
    
    
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias =BIAS)
    if os.path.isfile("server_1dof.pth"):
        print("loading prevoius model")
        model = load_model(model,"server_1dof.pth")
    

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

   
    
 

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if NOLOOPS:
            src, dst = make_graph_no_loops(2,0)
        else:
            src = src_list(2)
            dst = dst_list(2)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_loss_d2":0,  "train_H_d2":0, "test_loss_d2" :0, "test_H_d2" :0}
        

    container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            #h_tr = train_sample.ndata["h"].transpose(0,1)
            h_tr = correct_ham_data(train_sample)
    
            x0 = x_tr[0,:,:]
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            model.change_graph(roll_g)
            #print(roll_g)
            x_tr = x_tr.requires_grad_()
            x_tr_flat = x_tr.reshape(-1,2)
            
            #print(x_tr_flat.shape)
            h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            """
            if REG == "ridge":
                loss += alpha[0] * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            """
            x0 = x_tr[0,:,:].requires_grad_()
            model.change_graph(train_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            container[1,epoch]+=lossH.item()
            container[0,epoch] += loss.item()
            
            loss.backward()
            opti.step()
        container[0:2,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = h_tr = correct_ham_data(test_sample)
            model.change_graph(roll_g)
            x_ts = x_ts.requires_grad_()
            x_ts_flat = x_ts.reshape(-1,2)
            h_pred = model(x_ts_flat)
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())
            x0 = x_ts[0,:,:].requires_grad_()
            model.change_graph(test_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
        
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[0] * lossROLLt
                
           
            container[2,epoch]+=losst.item()
            container[3,epoch] += lossHt.item()
        container[2:4,epoch]/=N_test
    
        metrics["train_loss_d2"] = container[0,epoch]
        metrics["test_loss_d2"] = container[2,epoch]
        metrics["train_H_d2"] = container[1,epoch]
        metrics["test_H_d2"] = container[3,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
        dict={}
        print("Epoch: {}\nLOSS: train: {:.6f}   ham: {:.6f} |   test: {:.6f}  ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
        for namew , param in model.named_parameters():
            dict[namew+"_grad"] = torch.mean(param.grad).item()
        print(dict)
   
   
    
    visualize_loss("loss of 2dof pendelum",container)
    torch.save(model.state_dict(),"server_2dof.pth")
    #torch.save(model,"whole_model_dof2.pt")
    
def train3dof(configs):

    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    BIAS = configs["bias"]
    WANDB = True

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]
    S= configs["samples"]
    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "3dof pendelum"
    print(EPOCHS)
    NOLOOPS = configs["noloops"]
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
    data = torch.load("traj_3dof.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(3)
    dst = dst_list(3)
    if NOLOOPS:
        src,dst = del_loops(src,dst)
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:3] = angle_transformer(data[:,:,:,0:3])

    
        #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = data[:,-1,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)

    
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
    
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    train_snapshots = create_pend3dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend3dof_graph_snapshots(test,h_test,src,dst)
    
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias=BIAS)
    if os.path.isfile("server_2dof.pth"):
        print("loading prevoius model")
        model = load_model(model,"server_2dof.pth")
    

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

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if NOLOOPS:
            src, dst = make_graph_no_loops(3,0)
        else:
            src = src_list(3)
            dst = dst_list(3)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_loss_d3":0,  "train_H_d3":0, "test_loss_d3" :0, "test_H_d3" :0}
        

    container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            #h_tr = train_sample.ndata["h"].transpose(0,1)
            h_tr = correct_ham_data(train_sample)
    
            x0 = x_tr[0,:,:]
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            model.change_graph(roll_g)
            #print(roll_g)
            x_tr = x_tr.requires_grad_()
            x_tr_flat = x_tr.reshape(-1,2)
            
            #print(x_tr_flat.shape)
            h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            """
            if REG == "ridge":
                loss += alpha[0] * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            """
            x0 = x_tr[0,:,:].requires_grad_()
            model.change_graph(train_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            container[1,epoch]+=lossH.item()
            container[0,epoch] += loss.item()
            
            loss.backward()
            opti.step()
        container[0:2,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = h_tr = correct_ham_data(test_sample)
            model.change_graph(roll_g)
            x_ts = x_ts.requires_grad_()
            x_ts_flat = x_ts.reshape(-1,2)
            h_pred = model(x_ts_flat)
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())
            x0 = x_ts[0,:,:].requires_grad_()
            model.change_graph(test_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
        
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[0] * lossROLLt
                
           
            container[2,epoch]+=losst.item()
            container[3,epoch] += lossHt.item()
        container[2:4,epoch]/=N_test
    
        metrics["train_loss_d3"] = container[0,epoch]
        metrics["test_loss_d3"] = container[2,epoch]
        metrics["train_H_d3"] = container[1,epoch]
        metrics["test_H_d3"] = container[3,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
        dict={}
        print("Epoch: {}\nLOSS: train: {:.6f}   ham: {:.6f} |   test: {:.6f}  ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
        for namew , param in model.named_parameters():
            dict[namew+"_grad"] = torch.mean(param.grad).item()
        print(dict)
   
   
    
    visualize_loss("loss of 3dof pendelum",container)
    torch.save(model.state_dict(),"server_3dof.pth")
    #torch.save(model,"whole_model_dof3.pt")
    
def train4dof(configs):

    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this

    WANDB = True

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]
    S = configs["samples"]
    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "4dof pendelum"
    BIAS = configs["bias"]
    print(EPOCHS)
    
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
    NOLOOPS = configs["noloops"]
    data = torch.load("traj_4dof.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(4)
    dst = dst_list(4)
    if NOLOOPS:
        src,dst = del_loops(src,dst)
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:4] = angle_transformer(data[:,:,:,0:4])

    
        #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = data[:,-1,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]
    
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE)
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    train_snapshots = create_pend4dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend4dof_graph_snapshots(test,h_test,src,dst)
    
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias=BIAS)
    if os.path.isfile("server_3dof.pth"):
        print("loading prevoius model")
        model = load_model(model,"server_3dof.pth")
    

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

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if NOLOOPS:
            src, dst = make_graph_no_loops(4,0)
        else:
            src = src_list(4)
            dst = dst_list(4)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_loss_d4":0,  "train_H_d4":0, "test_loss_d4" :0, "test_H_d4" :0}
        

    container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            #h_tr = train_sample.ndata["h"].transpose(0,1)
            h_tr = correct_ham_data(train_sample)
    
            x0 = x_tr[0,:,:]
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            model.change_graph(roll_g)
            #print(roll_g)
            x_tr = x_tr.requires_grad_()
            x_tr_flat = x_tr.reshape(-1,2)
            
            #print(x_tr_flat.shape)
            h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            """
            if REG == "ridge":
                loss += alpha[0] * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            """
            x0 = x_tr[0,:,:].requires_grad_()
            model.change_graph(train_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            container[1,epoch]+=lossH.item()
            container[0,epoch] += loss.item()
            
            loss.backward()
            opti.step()
        container[0:2,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = h_tr = correct_ham_data(test_sample)
            model.change_graph(roll_g)
            x_ts = x_ts.requires_grad_()
            x_ts_flat = x_ts.reshape(-1,2)
            h_pred = model(x_ts_flat)
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())
            x0 = x_ts[0,:,:].requires_grad_()
            model.change_graph(test_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
        
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[0] * lossROLLt
                
           
            container[2,epoch]+=losst.item()
            container[3,epoch] += lossHt.item()
        container[2:4,epoch]/=N_test
    
        metrics["train_loss_d4"] = container[0,epoch]
        metrics["test_loss_d4"] = container[2,epoch]
        metrics["train_H_d4"] = container[1,epoch]
        metrics["test_H_d4"] = container[3,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
        dict={}
        print("Epoch: {}\nLOSS: train: {:.6f}   ham: {:.6f} |   test: {:.6f}  ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
        for namew , param in model.named_parameters():
            dict[namew+"_grad"] = torch.mean(param.grad).item()
        print(dict)
   
   
    
    visualize_loss("loss of 4dof pendelum",container)
    torch.save(model.state_dict(),"server_4dof.pth")


if __name__ == "__main__":
    with open("pend.yaml", 'r') as f:
        configs = yaml.load(f, yaml.Loader)
    wandb.init()
    print('Config file content:')
    print(configs)
    train1dof(configs)
    
    train2dof(configs)
    train3dof(configs)
    train4dof(configs)
    wandb.finish()