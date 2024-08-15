import torch
import dgl
from utils import *
import random
import yaml
from HGN import *
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

    WANDB = True

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    SINGLE = configs["single"]

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "1dof pendelum"
    print(EPOCHS)
    
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
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0] = angle_transformer(data[:,:,:,0])

    if SINGLE:
        temp = data[:,-1,:,:].unsqueeze(1)
        eval = data[:,-1,:,:]
        H = eval[:,:,-1]
        data = temp
        data[:,:,:,0] = angle_transformer(data[:,:,:,0])

    else:
        #num = random.randint(0,dataset.shape[1]-1)
        eval = data[:,-1,:,:].unsqueeze(1)
        H = eval[:,:,-1]
        data = data[:,:-1,:,:]
    print(data.shape)

    
    snapshots = make_snapshots(data[:,0:50,:,:],TIME_SIZE) # just 128 to keep everything in 2^i
    del data
    
    random.shuffle(snapshots)
    print(snapshots[0].shape)
    dgl_snapshots = create_pend1dof_graph_snapshots(snapshots,src,dst)
    
    #graph_snapshots = make_graph_snapshots(snapshots,nodes=6,feats=4)

    #dgl_snapshots = convert2dgl_snapshots(snapshots,src,dst)
    
    del snapshots
    print(len(dgl_snapshots))
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = HNN_maker(graph,2,MODEL_SIZE,32,type=MODEL)
    

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
    del dgl_snapshots
    
    
    
    metrics={"train_loss_d1":0, "train_grad_d1":0, "train_H_d1":0, "test_loss_d1" :0, "test_grad_d1":0,"test_H_d1" :0}
        

    container = torch.zeros(6,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loader=GraphDataLoader(train, batch_size=BATCH_SIZE,shuffle=True)
        N_train = len(train_loader)
        print("TRAIN")
        
        for train_sample in tqdm(train_loader):
            #print(train_sample.num_nodes())
        
            loss=0
            opti.zero_grad()
            model.set_graph(train_sample)
            x = train_sample.ndata["xfeat"].transpose(0,1)
            dx = train_sample.ndata["dxfeat"].transpose(0,1)
            h = train_sample.ndata["hfeat"].transpose(0,1)
            x0 = x[0,:,:]
            #print(x.shape)
           # print(dx.shape)
           # print(x0.shape)
            #print("odeint")
            predy = odeint(model,x0,ts,method="rk4")
            #print("dy")
            #print(x.shape)
            dy = model.rolldx(x)
            #print("dy done")
            #print(dy.shape)
           # print(y.shape)
            loss = lossfn(predy,x)
            if REG == "ridge":
                loss += alpha[0] * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            loss2 = lossfn(dy,dx)
            if SOB[0]:
                loss += s_alpha[1] * loss2
            #print("before H roll")
            hy = model.rollH(x)
            loss3 = lossfn(hy,h)
            #print("after h roll")
            if SOB[1]:
                loss += s_alpha[2] * loss3
            container[1,epoch]+=loss2.item()
            container[0,epoch] += loss.item()
            container[2,epoch]+=loss3.item()
            loss.backward()
            opti.step()
        container[0:3,epoch]/=N_train
        print("TEST")
        model.eval()
        test_loader=GraphDataLoader(test,batch_size=BATCH_SIZE,shuffle=True)
        N_test = len(test_loader)
        
        for test_sample in tqdm(test_loader):
            testloss = 0
            model.set_graph(test_sample)
            xt = test_sample.ndata["xfeat"].transpose(0,1)
            dxt = test_sample.ndata["dxfeat"].transpose(0,1)
            ht = test_sample.ndata["hfeat"].transpose(0,1)
            x0t = xt[0,:,:]
            #print(xt.shape)
            #print(dxt.shape)
            #print(x0t.shape)
            yt = odeint(model,x0t,ts,method="rk4")
            dyt = model.rolldx(xt)
            hyt = model.rollH(xt)
            #print(predyt.shape)
    
            #print(yt.shape)
            #print(x0t.shape)
            testloss = s_alpha[0]*lossfn(yt,xt)
            if REG == "ridge":
                testloss += alpha * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                testloss += alpha * sum(p.abs().sum() for p in model.parameters())
            testloss2 = lossfn(dyt,dxt)
            testloss3 = lossfn(hyt,ht)
            if SOB[0]:
                testloss += s_alpha[1] * testloss2
            if SOB[1]:
                testloss += s_alpha[2] * testloss3
                
            container[5,epoch]+=testloss3.item()
            container[4,epoch]+=testloss2.item()
            container[3,epoch] += testloss.item()
        container[3:6,epoch]/=N_test
    
        metrics["train_loss_d1"] = container[0,epoch]
        metrics["test_loss_d1"] = container[3,epoch]
        metrics["train_grad_d1"] = container[1,epoch]
        metrics["test_grad_d1"] = container[4,epoch]
        metrics["train_H_d1"] = container[2,epoch]
        metrics["test_H_d1"] = container[5,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
       
        print("Epoch: {}\nLOSS: train: {:.6f} grad: {:.6f}  ham: {:.6f} |   test: {:.6f} grad: {:.6f} ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch],container[4,epoch],container[5,epoch]))
    del train
    del train_loader
    del test
    del test_loader
   
    
    visualize_loss("loss of 1dof pendelum",container)
    torch.save(model.state_dict(),"server_1dof.pth")
    
def train2dof(configs):

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
    SINGLE = configs["single"]

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "1dof pendelum"
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
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:2] = angle_transformer(data[:,:,:,0:2])

    if SINGLE:
        #num = random.randint(0,data.shape[1]-1)
        temp = data[:,-1,:,:].unsqueeze(1)
        eval = data[:,-1,:,:]
        H = eval[:,:,-1]
        data = temp
        data[:,:,:,0:2] = angle_transformer(data[:,:,:,0:2])
    else:
        #num = random.randint(0,dataset.shape[1]-1)
        eval = data[:,-1,:,:].unsqueeze(1)
        H = eval[:,:,-1]
        data = data[:,:-1,:,:]
    print(data.shape)

    
    snapshots = make_snapshots(data[:,0:50,:,:],TIME_SIZE) # just 128 to keep everything in 2^i
    del data
    
    random.shuffle(snapshots)
    print(snapshots[0].shape)
    dgl_snapshots = create_pend2dof_graph_snapshots(snapshots,src,dst)
    
    #graph_snapshots = make_graph_snapshots(snapshots,nodes=6,feats=4)

    #dgl_snapshots = convert2dgl_snapshots(snapshots,src,dst)
    
    del snapshots
    print(len(dgl_snapshots))
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = HNN_maker(graph,2,MODEL_SIZE,32,type=MODEL)
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

    train = dgl_snapshots[0:int(SPLIT*len(dgl_snapshots))]
    test = dgl_snapshots[int(SPLIT*len(dgl_snapshots)):]
    del dgl_snapshots
    
 

    metrics={"train_loss_d2":0, "train_grad_d2":0, "train_H_d2":0, "test_loss_d2" :0, "test_grad_d2":0, "test_H_d2":0}
           

    container = torch.zeros(6,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loader=GraphDataLoader(train, batch_size=BATCH_SIZE,shuffle=True)
        N_train = len(train_loader)
        print("TRAIN")
        
        for train_sample in tqdm(train_loader):
            #print(train_sample.num_nodes())
        
            loss=0
            opti.zero_grad()
            model.set_graph(train_sample)
            x = train_sample.ndata["xfeat"].transpose(0,1)
            dx = train_sample.ndata["dxfeat"].transpose(0,1)
            h = correct_ham_data(train_sample)
            x0 = x[0,:,:]
            #print(x.shape)
            
            #print(dx.shape)
            #print(x0.shape)
            #print("odeint")
            predy = odeint(model,x0,ts,method="rk4")
            #print(predy.shape)
            #print("dy")
            #print(x.shape)
            dy = model.rolldx(x)
            #print("dy done")
            #print(dy.shape)
           # print(y.shape)
            loss = s_alpha[0]*lossfn(predy,x)
            if REG == "ridge":
                loss += alpha * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            loss2 = lossfn(dy,dx)
            if SOB[0]:
                loss += s_alpha[1]  * loss2
            #print("before H roll")
            hy = model.rollH(x)
            #print("h: {}".format(h.shape))
            #print("h: {}".format(correct_ham_data(train_sample,2).shape))
            #print("hy:{}".format(hy.shape))
            loss3 = lossfn(hy,h)
            #print("after h roll")
            if SOB[1]:
                loss += s_alpha[2]  * loss3
        
            container[1,epoch]+=loss2.item()
            container[0,epoch] += loss.item()
            container[2,epoch]+=loss3.item()
            loss.backward()
            opti.step()
        container[0:3,epoch]/=N_train
        print("TEST")
        model.eval()
        test_loader=GraphDataLoader(test,batch_size=BATCH_SIZE,shuffle=True)
        N_test = len(test_loader)
        
        for test_sample in tqdm(test_loader):
            testloss = 0
            model.set_graph(test_sample)
            xt = test_sample.ndata["xfeat"].transpose(0,1)
            dxt = test_sample.ndata["dxfeat"].transpose(0,1)
            ht = correct_ham_data(test_sample)
            x0t = xt[0,:,:]
            #print(xt.shape)
            #print(dxt.shape)
            #print(x0t.shape)
            yt = odeint(model,x0t,ts,method="rk4")
            dyt = model.rolldx(xt)
            hyt = model.rollH(xt)
            #print(predyt.shape)
    
            #print(yt.shape)
            #print(x0t.shape)
            testloss = s_alpha[0]*lossfn(yt,xt)
            if REG == "ridge":
                testloss += alpha * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                testloss += alpha * sum(p.abs().sum() for p in model.parameters())
            testloss2 = lossfn(dyt,dxt)
            testloss3 = lossfn(hyt,ht)
            if SOB[0]:
                testloss += s_alpha[1] * testloss2
            if SOB[1]:
                testloss += s_alpha[2] * testloss3
                
            container[5,epoch]+=testloss3.item()
            container[4,epoch]+=testloss2.item()
            container[3,epoch] += testloss.item()
        container[3:6,epoch]/=N_test
        
        metrics["train_loss_d2"] = container[0,epoch]
        metrics["test_loss_d2"] = container[3,epoch]
        metrics["train_grad_d2"] = container[1,epoch]
        metrics["test_grad_d2"] = container[4,epoch]
        metrics["train_H_d2"] = container[2,epoch]
        metrics["test_H_d2"] = container[5,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
            
        print("Epoch: {}\nLOSS: train: {:.6f} grad: {:.6f}  ham: {:.6f} |   test: {:.6f} grad: {:.6f} ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch],container[4,epoch],container[5,epoch]))
    del train
    del train_loader
    del test
    del test_loader
   
    
    visualize_loss("loss of 2dof pendelum",container)
    torch.save(model.state_dict(),"server_2dof.pth")
    
def train3dof(configs):

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
    SINGLE = configs["single"]

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "1dof pendelum"
    print(EPOCHS)
    
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
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:3] = angle_transformer(data[:,:,:,0:3])

    if SINGLE:
        num = random.randint(0,data.shape[1]-1)
        temp = data[:,-1,:,:].unsqueeze(1)
        eval = data[:,-1,:,:]
        H = eval[:,:,-1]
        data = temp
    else:
        #num = random.randint(0,dataset.shape[1]-1)
        eval = data[:,-1,:,:].unsqueeze(1)
        H = data[:,-1,:,-1]
        data = data[:,:-1,:,:]
    print(data.shape)

    
    snapshots = make_snapshots(data[:,0:50,:,:],TIME_SIZE) # just 128 to keep everything in 2^i
    del data
    
    random.shuffle(snapshots)
    print(snapshots[0].shape)
    dgl_snapshots = create_pend3dof_graph_snapshots(snapshots,src,dst)
    
    #graph_snapshots = make_graph_snapshots(snapshots,nodes=6,feats=4)

    #dgl_snapshots = convert2dgl_snapshots(snapshots,src,dst)
    
    del snapshots
    print(len(dgl_snapshots))
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = HNN_maker(graph,2,MODEL_SIZE,32,type=MODEL)
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

    train = dgl_snapshots[0:int(SPLIT*len(dgl_snapshots))]
    test = dgl_snapshots[int(SPLIT*len(dgl_snapshots)):]
    del dgl_snapshots
 
    
    
    metrics={"train_loss_d3":0, "train_grad_d3":0,"train_H_d3":0, "test_loss_d3" :0, "test_grad_d3":0,"test_H_d3":0}
          

    container = torch.zeros(6,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

   
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loader=GraphDataLoader(train, batch_size=BATCH_SIZE,shuffle=True)
        N_train = len(train_loader)
        print("TRAIN")
        
        for train_sample in tqdm(train_loader):
            #print(train_sample.num_nodes())
        
            loss=0
            opti.zero_grad()
            model.set_graph(train_sample)
            x = train_sample.ndata["xfeat"].transpose(0,1)
            dx = train_sample.ndata["dxfeat"].transpose(0,1)
            h = correct_ham_data(train_sample)
            x0 = x[0,:,:]
            #print(x.shape)
           # print(dx.shape)
           # print(x0.shape)
            #print("odeint")
            predy = odeint(model,x0,ts,method="rk4")
            #print("dy")
            #print(x.shape)
            dy = model.rolldx(x)
            #print("dy done")
            #print(dy.shape)
           # print(y.shape)
            loss = s_alpha[0]*lossfn(predy,x)
            if REG == "ridge":
                loss += alpha * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            loss2 = lossfn(dy,dx)
            if SOB[0]:
                loss += s_alpha[1] * loss2
            #print("before H roll")
            hy = model.rollH(x)
            loss3 = lossfn(hy,h)
            #print("after h roll")
            if SOB[1]:
                loss += s_alpha[2] * loss3
        
            container[1,epoch]+=loss2.item()
            container[0,epoch] += loss.item()
            container[2,epoch]+=loss3.item()
            loss.backward()
            opti.step()
        container[0:3,epoch]/=N_train
        print("TEST")
        model.eval()
        test_loader=GraphDataLoader(test,batch_size=BATCH_SIZE,shuffle=True)
        N_test = len(test_loader)
        
        for test_sample in tqdm(test_loader):
            testloss = 0
            model.set_graph(test_sample)
            xt = test_sample.ndata["xfeat"].transpose(0,1)
            dxt = test_sample.ndata["dxfeat"].transpose(0,1)
            ht = correct_ham_data(test_sample)
            x0t = xt[0,:,:]
            #print(xt.shape)
            #print(dxt.shape)
            #print(x0t.shape)
            yt = odeint(model,x0t,ts,method="rk4")
            dyt = model.rolldx(xt)
            hyt = model.rollH(xt)
            #print(predyt.shape)
    
            #print(yt.shape)
            #print(x0t.shape)
            testloss = s_alpha[0]*lossfn(yt,xt)
            if REG == "ridge":
                testloss += alpha * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                testloss += alpha * sum(p.abs().sum() for p in model.parameters())
            testloss2 = lossfn(dyt,dxt)
            testloss3 = lossfn(hyt,ht)
            if SOB[0]:
                testloss += s_alpha[1] * testloss2
            if SOB[1]:
                testloss += s_alpha[2] * testloss3
                
            container[5,epoch]+=testloss3.item()
            container[4,epoch]+=testloss2.item()
            container[3,epoch] += testloss.item()
        container[3:6,epoch]/=N_test
        
        metrics["train_loss_d3"] = container[0,epoch]
        metrics["test_loss_d3"] = container[3,epoch]
        metrics["train_grad_d3"] = container[1,epoch]
        metrics["test_grad_d3"] = container[4,epoch]
        metrics["train_H_d3"] = container[2,epoch]
        metrics["test_H_d3"] = container[5,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
            
        print("Epoch: {}\nLOSS: train: {:.6f} grad: {:.6f}  ham: {:.6f} |   test: {:.6f} grad: {:.6f} ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch],container[4,epoch],container[5,epoch]))
    del train              
   
    
    visualize_loss("loss of 3dof pendelum",container)
    torch.save(model.state_dict(),"server_3dof.pth")
    
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
    SINGLE = configs["single"]

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "1dof pendelum"
    print(EPOCHS)
    
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
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
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:4] = angle_transformer(data[:,:,:,0:4])

    if SINGLE:
        num = random.randint(0,data.shape[1]-1)
        temp = data[:,-1,:,:].unsqueeze(1)
        eval = data[:,-1,:,:]
        H = eval[:,:,-1]
        data = temp
    else:
        #num = random.randint(0,dataset.shape[1]-1)
        eval = data[:,-1,:,:].unsqueeze(1)
        H = data[:,-1,:,-1]
        data = data[:,:-1,:,:]
    print(data.shape)

    
    snapshots = make_snapshots(data[:,0:50,:,:],TIME_SIZE) # just 128 to keep everything in 2^i
    del data
    
    random.shuffle(snapshots)
    print(snapshots[0].shape)
    dgl_snapshots = create_pend4dof_graph_snapshots(snapshots,src,dst)
    
    #graph_snapshots = make_graph_snapshots(snapshots,nodes=6,feats=4)

    #dgl_snapshots = convert2dgl_snapshots(snapshots,src,dst)
    
    del snapshots
    print(len(dgl_snapshots))
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = HNN_maker(graph,2,MODEL_SIZE,32,type=MODEL)
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

    train = dgl_snapshots[0:int(SPLIT*len(dgl_snapshots))]
    test = dgl_snapshots[int(SPLIT*len(dgl_snapshots)):]
    del dgl_snapshots
   
    
    
    metrics={"train_loss_d4":0, "train_grad_d4":0, "train_H_d4":0,"test_loss_d4" :0, "test_grad_d4":0,"test_H_d4":0}
          

    container = torch.zeros(6,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loader=GraphDataLoader(train, batch_size=BATCH_SIZE,shuffle=True)
        N_train = len(train_loader)
        print("TRAIN")
        
        for train_sample in tqdm(train_loader):
            #print(train_sample.num_nodes())
        
            loss=0
            opti.zero_grad()
            model.set_graph(train_sample)
            x = train_sample.ndata["xfeat"].transpose(0,1)
            dx = train_sample.ndata["dxfeat"].transpose(0,1)
            h = correct_ham_data(train_sample)
            x0 = x[0,:,:]
            #print(x.shape)
           # print(dx.shape)
           # print(x0.shape)
            #print("odeint")
            predy = odeint(model,x0,ts,method="rk4")
            #print("dy")
            #print(x.shape)
            dy = model.rolldx(x)
            #print("dy done")
            #print(dy.shape)
           # print(y.shape)
            loss = s_alpha[0]*lossfn(predy,x)
            if REG == "ridge":
                loss += alpha * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            loss2 = lossfn(dy,dx)
            if SOB[0]:
                loss += s_alpha[1] * loss2
            #print("before H roll")
            hy = model.rollH(x)
            loss3 = lossfn(hy,h)
            #print("after h roll")
            if SOB[1]:
                loss += s_alpha[2] * loss3
        
            container[1,epoch]+=loss2.item()
            container[0,epoch] += loss.item()
            container[2,epoch]+=loss3.item()
            loss.backward()
            opti.step()
        container[0:3,epoch]/=N_train
        print("TEST")
        model.eval()
        test_loader=GraphDataLoader(test,batch_size=BATCH_SIZE,shuffle=True)
        N_test = len(test_loader)
        
        for test_sample in tqdm(test_loader):
            testloss = 0
            model.set_graph(test_sample)
            xt = test_sample.ndata["xfeat"].transpose(0,1)
            dxt = test_sample.ndata["dxfeat"].transpose(0,1)
            ht = correct_ham_data(test_sample)
            x0t = xt[0,:,:]
            #print(xt.shape)
            #print(dxt.shape)
            #print(x0t.shape)
            yt = odeint(model,x0t,ts,method="rk4")
            dyt = model.rolldx(xt)
            hyt = model.rollH(xt)
            #print(predyt.shape)
    
            #print(yt.shape)
            #print(x0t.shape)
            testloss = s_alpha[0]*lossfn(yt,xt)
            if REG == "ridge":
                testloss += alpha * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                testloss += alpha * sum(p.abs().sum() for p in model.parameters())
            testloss2 = lossfn(dyt,dxt)
            testloss3 = lossfn(hyt,ht)
            if SOB[0]:
                testloss += s_alpha[1] * testloss2
            if SOB[1]:
                testloss += s_alpha[2] * testloss3
                
            container[5,epoch]+=testloss3.item()
            container[4,epoch]+=testloss2.item()
            container[3,epoch] += testloss.item()
        container[3:6,epoch]/=N_test
        
        metrics["train_loss_d4"] = container[0,epoch]
        metrics["test_loss_d4"] = container[3,epoch]
        metrics["train_grad_d4"] = container[1,epoch]
        metrics["test_grad_d4"] = container[4,epoch]
        metrics["train_H_d4"] = container[2,epoch]
        metrics["test_H_d4"] = container[5,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
            
        print("Epoch: {}\nLOSS: train: {:.6f} grad: {:.6f}  ham: {:.6f} |   test: {:.6f} grad: {:.6f} ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch],container[4,epoch],container[5,epoch]))
    del train              
   
    
    visualize_loss("loss of 4dof pendelum",container)
    torch.save(model.state_dict(),"server_4dof.pth")


if __name__ == "__main__":
    with open("dofpend.yaml", 'r') as f:
        configs = yaml.load(f, yaml.Loader)

    print('Config file content:')
    print(configs)
    #train1dof(configs)
    
    #train2dof(configs)
    train3dof(configs)
