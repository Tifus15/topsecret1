import torch
import torch.optim as opti
from utils import *
from model_gnn import *
from dgl.dataloading import GraphDataLoader
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
TIMESIZE = 2
BATCHSIZE = 1
EPOCHS = 100
SELF_LOOPS = False
LR = 1e-3
S=100#100
DOFS = [2,3,4]
trial = -1
src = [0]
dst = [0]
g = dgl.graph((src,dst))
container = torch.Tensor(len(DOFS)*EPOCHS,2)


model =  GNN(g,["tanh"],2,2,[256],bias = True,type="gcn")
print(model)
for name,param in model.named_parameters():
    print("{} : {}".format(name, param.data.shape))
optimizer = opti.AdamW(model.parameters(),lr=LR)
loss_fn = nn.MSELoss()
loss_r = nn.L1Loss()
t = torch.linspace(0,1.27,128)

print("############")
print("DOF1")
i=1
print("############")
if i in DOFS:
    trial +=1
    dataset = torch.load("traj_1dof.pt")
    dataset.requires_grad=False
    print(dataset.shape)

    snaps = snap_maker(dataset[:,0:S,:,:],TIMESIZE)
    gdataset = create_pend1dof_graph_snapshots(snaps,src,dst)
    border = int(0.9*len(gdataset))

    train = gdataset[0:border]
    test = gdataset[border:]
    print(train[0])

    tb = t[0:TIMESIZE]
    trainset=GraphDataLoader(train, batch_size=BATCHSIZE,shuffle=True,drop_last=True)
    testset =GraphDataLoader(test, batch_size=BATCHSIZE,shuffle=True,drop_last=True)

    it = iter(trainset)
    g_batch = next(it)

    model.change_graph(g_batch)
    a=1
    for epoch in tqdm(range(int(trial*EPOCHS),int((trial+1)*EPOCHS))):
        losstr_acc=0
        lossts_acc=0
        model.train()
        print("TRAIN")
        for sample in tqdm(trainset):
            optimizer.zero_grad()
            data = sample.ndata["x"].transpose(0,1)
            x0 = data[0,:,:]
            pred = odeint(model,x0,tb,method="rk4")
            #print("pred: {}".format(pred))
            #print("data: {}".format(data.shape))
            loss = a*(loss_fn(pred[:,:,0],data[:,:,0]) + loss_fn(pred[:,:,1],data[:,:,1]))
            #print(loss)
            loss.backward()
            optimizer.step()
            losstr_acc+=loss.item()
        
        #model.change_graph(g_test)
        model.eval()
        print("TEST")
        for tsample in tqdm(testset):
            tdata = tsample.ndata["x"].transpose(0,1)
            x0t = tdata[0,:,:]
            #print(x0t.shape)
            predt = odeint(model,x0t,tb,method="rk4")
            loss = a*(loss_fn(predt[:,:,0],tdata[:,:,0]) + loss_fn(predt[:,:,1],tdata[:,:,1]))
            lossts_acc+=loss.item()
        dict={}
        for name,param in model.named_parameters():
            #dict[name]=torch.mean(param.data.flatten())
            dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()
        print("epoch: {}, train loss: {}, test_loss: {}".format(epoch+1,losstr_acc/len(trainset),lossts_acc/len(testset)))
        print(dict)
        container[epoch,0]=losstr_acc/len(trainset)
        container[epoch,1]=lossts_acc/len(testset)

print("############")
print("DOF2")
i = 2
print("############")
if i in DOFS:
    trial += 1
    if SELF_LOOPS:
        src = [0,0,1,1]
        dst = [0,1,0,1]
    else:
        src = [0,1]
        dst = [1,0]
    dataset = torch.load("traj_2dof.pt")
    dataset.requires_grad=False
    print(dataset.shape)

    snaps = snap_maker(dataset[:,0:S,:,:],TIMESIZE)
    gdataset = create_pend2dof_graph_snapshots(snaps,src,dst)
    border = int(0.9*len(gdataset))

    train = gdataset[0:border]
    test = gdataset[border:]
    print(train[0])

    tb = t[0:TIMESIZE]
    trainset=GraphDataLoader(train, batch_size=BATCHSIZE,shuffle=True,drop_last=True)
    testset =GraphDataLoader(test, batch_size=BATCHSIZE,shuffle=True,drop_last=True)

    it = iter(trainset)
    g_batch = next(it)

    model.change_graph(g_batch)
    a=1
    for epoch in tqdm(range(int(trial*EPOCHS),int((trial+1)*EPOCHS))):
        losstr_acc=0
        lossts_acc=0
        model.train()
        print("TRAIN")
        for sample in tqdm(trainset):
            optimizer.zero_grad()
            data = sample.ndata["x"].transpose(0,1)
            x0 = data[0,:,:]
            pred = odeint(model,x0,tb,method="rk4")
            #print("pred: {}".format(pred))
            #print("data: {}".format(data.shape))
            loss = a*(loss_fn(pred[:,:,0],data[:,:,0]) + loss_fn(pred[:,:,1],data[:,:,1]))
            #print(loss)
            loss.backward()
            optimizer.step()
            losstr_acc+=loss.item()
        
        #model.change_graph(g_test)
        model.eval()
        print("TEST")
        for tsample in tqdm(testset):
            tdata = tsample.ndata["x"].transpose(0,1)
            x0t = tdata[0,:,:]
            #print(x0t.shape)
            predt = odeint(model,x0t,tb,method="rk4")
            loss = a*(loss_fn(predt[:,:,0],tdata[:,:,0]) + loss_fn(predt[:,:,1],tdata[:,:,1]))
            lossts_acc+=loss.item()
        dict={}
        for name,param in model.named_parameters():
            #dict[name]=torch.mean(param.data.flatten())
            dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()
        print("epoch: {}, train loss: {}, test_loss: {}".format(epoch+1,losstr_acc/len(trainset),lossts_acc/len(testset)))
        print(dict)
        container[epoch,0]=losstr_acc/len(trainset)
        container[epoch,1]=lossts_acc/len(testset)


print("############")
print("DOF3")
i = 3
print("############")
if i in DOFS:
    trial += 1
    if SELF_LOOPS:
        src = [0,0,0,1,1,1,2,2,2]
        dst = [0,1,2,0,1,2,0,1,2]
    else:
        src = [0,0,1,1,2,2]
        dst = [1,2,0,2,0,1]

    dataset = torch.load("traj_3dof.pt")
    dataset.requires_grad=False
    print(dataset.shape)

    snaps = snap_maker(dataset[:,0:S,:,:],TIMESIZE)
    gdataset = create_pend3dof_graph_snapshots(snaps,src,dst)
    border = int(0.9*len(gdataset))

    train = gdataset[0:border]
    test = gdataset[border:]
    print(train[0])

    tb = t[0:TIMESIZE]
    trainset=GraphDataLoader(train, batch_size=BATCHSIZE,shuffle=True,drop_last=True)
    testset =GraphDataLoader(test, batch_size=BATCHSIZE,shuffle=True,drop_last=True)

    it = iter(trainset)
    g_batch = next(it)

    model.change_graph(g_batch)
    a=1
    for epoch in tqdm(range(int(trial*EPOCHS),int((trial+1)*EPOCHS))):
        losstr_acc=0
        lossts_acc=0
        model.train()
        print("TRAIN")
        for sample in tqdm(trainset):
            optimizer.zero_grad()
            data = sample.ndata["x"].transpose(0,1)
            x0 = data[0,:,:]
            pred = odeint(model,x0,tb,method="rk4")
            #print("pred: {}".format(pred))
            #print("data: {}".format(data.shape))
            loss = a*(loss_fn(pred[:,:,0],data[:,:,0]) + loss_fn(pred[:,:,1],data[:,:,1]))
            #print(loss)
            loss.backward()
            optimizer.step()
            losstr_acc+=loss.item()
        
        #model.change_graph(g_test)
        model.eval()
        print("TEST")
        for tsample in tqdm(testset):
            tdata = tsample.ndata["x"].transpose(0,1)
            x0t = tdata[0,:,:]
            #print(x0t.shape)
            predt = odeint(model,x0t,tb,method="rk4")
            loss = a*(loss_fn(predt[:,:,0],tdata[:,:,0]) + loss_fn(predt[:,:,1],tdata[:,:,1]))
            lossts_acc+=loss.item()
        dict={}
        for name,param in model.named_parameters():
            #dict[name]=torch.mean(param.data.flatten())
            dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()
        print("epoch: {}, train loss: {}, test_loss: {}".format(epoch+1,losstr_acc/len(trainset),lossts_acc/len(testset)))
        print(dict)
        container[epoch,0]=losstr_acc/len(trainset)
        container[epoch,1]=lossts_acc/len(testset)
    
print("############")
print("DOF4")
i = 4
print("############")
if i in DOFS:
    trial += 1
    if SELF_LOOPS:
        src = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
        dst = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    else:
        src = [0,0,0,1,1,1,2,2,2,3,3,3]
        dst = [1,2,3,0,2,3,0,1,3,0,1,2]

    dataset = torch.load("traj_4dof.pt")
    dataset.requires_grad=False
    print(dataset.shape)

    snaps = snap_maker(dataset[:,0:S,:,:],TIMESIZE)
    gdataset = create_pend4dof_graph_snapshots(snaps,src,dst)
    border = int(0.9*len(gdataset))

    train = gdataset[0:border]
    test = gdataset[border:]
    print(train[0])

    tb = t[0:TIMESIZE]
    trainset=GraphDataLoader(train, batch_size=BATCHSIZE,shuffle=True,drop_last=True)
    testset =GraphDataLoader(test, batch_size=BATCHSIZE,shuffle=True,drop_last=True)

    it = iter(trainset)
    g_batch = next(it)

    model.change_graph(g_batch)
    a=1
    for epoch in tqdm(range(int(trial*EPOCHS),int((trial+1)*EPOCHS))):
        losstr_acc=0
        lossts_acc=0
        model.train()
        print("TRAIN")
        for sample in tqdm(trainset):
            optimizer.zero_grad()
            data = sample.ndata["x"].transpose(0,1)
            x0 = data[0,:,:]
            pred = odeint(model,x0,tb,method="rk4")
            #print("pred: {}".format(pred))
            #print("data: {}".format(data.shape))
            loss = a*(loss_fn(pred[:,:,0],data[:,:,0]) + loss_fn(pred[:,:,1],data[:,:,1]))
            #print(loss)
            loss.backward()
            optimizer.step()
            losstr_acc+=loss.item()
        
        #model.change_graph(g_test)
        model.eval()
        print("TEST")
        for tsample in tqdm(testset):
            tdata = tsample.ndata["x"].transpose(0,1)
            x0t = tdata[0,:,:]
            #print(x0t.shape)
            predt = odeint(model,x0t,tb,method="rk4")
            loss = a*(loss_fn(predt[:,:,0],tdata[:,:,0]) + loss_fn(predt[:,:,1],tdata[:,:,1]))
            lossts_acc+=loss.item()
        dict={}
        for name,param in model.named_parameters():
            #dict[name]=torch.mean(param.data.flatten())
            dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()
        print("epoch: {}, train loss: {}, test_loss: {}".format(epoch+1,losstr_acc/len(trainset),lossts_acc/len(testset)))
        print(dict)
        container[epoch,0]=losstr_acc/len(trainset)
        container[epoch,1]=lossts_acc/len(testset)

visualize_loss("losses",container.transpose(0,1))







