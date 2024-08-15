import torch
import torch.nn as nn
import torch.optim as optim
import random
#from model_bench import ode_model_eval
#from models import roll
from dglmodel import roll
#from torch_geometric.loader import DataLoader
from device_util import DEVICE
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import transform_Data_3dof
import math
from dgl.dataloading import GraphDataLoader

def training_dgl(settings, train_set, test_set,model,t,eval_pack=None):
    EPOCHS = settings["epochs"]
    loss_container = torch.zeros(EPOCHS,2)
    dgl_model = model
    

    if settings["opti"]== "AdamW":
        print("AdamW as optimizer")
        optimizer_dgl = optim.AdamW(dgl_model.parameters(),settings["lr"])
       
    elif settings["opti"]=="RMS":
        print("RMSprop as optimizer")
        optimizer_dgl = optim.RMSprop(dgl_model.parameters(),settings["lr"])
      
    else:
        print("SGD as optimizer")
        optimizer_dgl = optim.SGD(dgl_model.parameters(),settings["lr"])
     
    
    train_dgl = train_set

    test_dgl = test_set

    s_train = len(train_dgl)
    s_test = len(test_dgl)

    print("train samples: {}, test samples: {}".format(s_train,s_test))
    if settings["loss"]=="MSE":
        print("MSELoss")
        loss_fn = nn.MSELoss()
    elif settings["loss"] =="Huber":
        print("HuberLoss")
        loss_fn = nn.HuberLoss()
    else:
        print("MAELoss")
        loss_fn = nn.L1Loss()
    
    for epoch in tqdm(range(EPOCHS)):
        load_dgl = GraphDataLoader(train_dgl,batch_size=settings["batch"])
        n_train = len(load_dgl)
        print("{} : TRAIN batches".format(n_train))

        dgl_model.train()
        for dgl_sample in tqdm(load_dgl):
            optimizer_dgl.zero_grad()
           
            
            dglx, dgly, edges = dgl_sample.ndata["x"].to(DEVICE) , dgl_sample.ndata["y"].to(DEVICE), dgl_sample.to(DEVICE)

            
            dgltrain_y = roll(dgl_model,dglx,t.to(DEVICE),edges,settings["type"])




            #print("TRAIN: shapes: {}, {}, {}, {}".format(pygy.shape,pygtrain_y.shape,dgly.shape,dgltrain_y.shape))
            loss_dgl = loss_fn(dgltrain_y[:,0,:],dgly[:,0,:]) + loss_fn(dgltrain_y[:,1,:],dgly[:,1,:])

            loss_container[epoch,0] += loss_dgl.item()

            
            loss_dgl.backward()

           
            optimizer_dgl.step()
        
        loss_container[epoch,0]/=(n_train*(len(t)+1))

        testload_dgl = GraphDataLoader(test_dgl,batch_size=int(math.sqrt(len(test_dgl))))
    
        n_test = len(testload_dgl)



        dgl_model.eval()
 
        print("{} : TEST batches".format(n_test))
        for dgl_sample in tqdm(testload_dgl):
            
            dgltx, dglty, edges = dgl_sample.ndata["x"].to(DEVICE) , dgl_sample.ndata["y"].to(DEVICE), dgl_sample.to(DEVICE)
            


            dgltest_y = roll(dgl_model,dgltx,t.to(DEVICE),edges,settings["type"])
            #print("TEST: shapes: {}, {}, {}, {}".format())

            testloss_dgl = loss_fn(dgltest_y[:,0,:],dglty[:,0,:]) + loss_fn(dgltest_y[:,1,:],dglty[:,1,:])

            loss_container[epoch,1] += testloss_dgl.item()
       
        loss_container[epoch,1]/=(n_test*len(t))
        print("E: {} GGNN_HNN:: TRAIN: {} TEST: {}\n".format(epoch, loss_container[epoch,0], loss_container[epoch,1]))
        if eval_pack:
            visualize_eval("GGCNN",model,eval_pack[1],eval_pack[0],eval_pack[2])
    return dgl_model, loss_container



def training_pyg(settings, train_set, test_set,model,t,eval_pack=None):
    EPOCHS = settings["epochs"]
    loss_container = torch.zeros(EPOCHS,2)
    pyg_model = model
    

    if settings["opti"]== "AdamW":
        print("AdamW as optimizer")
        optimizer_pyg = optim.AdamW(pyg_model.parameters(),settings["lr"])
       
    elif settings["opti"]=="RMS":
        print("RMSprop as optimizer")
        optimizer_pyg = optim.RMSprop(pyg_model.parameters(),settings["lr"])
      
    else:
        print("SGD as optimizer")
        optimizer_pyg = optim.SGD(pyg_model.parameters(),settings["lr"])
     
    
    train_pyg = train_set

    test_pyg = test_set


    s_train = len(train_pyg)
    s_test = len(test_pyg)

    print("train samples: {}, test samples: {}".format(s_train,s_test))
    if settings["loss"]=="MSE":
        print("MSELoss")
        loss_fn = nn.MSELoss()
    elif settings["loss"] =="Huber":
        print("HuberLoss")
        loss_fn = nn.HuberLoss()
    else:
        print("MAELoss")
        loss_fn = nn.L1Loss()


    for epoch in tqdm(range(EPOCHS)):
        load_pyg = DataLoader(train_pyg,settings["batch"],shuffle=True)
        n_train = len(load_pyg)
        print("{} : TRAIN batches".format(n_train))

        pyg_model.train()
        for pyg_sample in tqdm(load_pyg):
            optimizer_pyg.zero_grad()
           
            
            pygx, pygy, edges = pyg_sample.x.to(DEVICE) , pyg_sample.y.to(DEVICE), pyg_sample.edge_index.to(DEVICE)

            
            pygtrain_y = roll(pyg_model,pygx,t.to(DEVICE),edges,settings["type"])




            #print("TRAIN: shapes: {}, {}, {}, {}".format(pygy.shape,pygtrain_y.shape,dgly.shape,dgltrain_y.shape))
            loss_pyg = loss_fn(pygtrain_y[:,0,:],pygy[:,0,:]) + loss_fn(pygtrain_y[:,1,:],pygy[:,1,:])

            loss_container[epoch,0] += loss_pyg.item()

            
            loss_pyg.backward()

           
            optimizer_pyg.step()

        loss_container[epoch,0]/=(n_train*(len(t)+1))

        testload_pyg = DataLoader(test_pyg,int(math.sqrt(len(test_pyg))))
    
        n_test = len(testload_pyg)



        pyg_model.eval()
 
        print("{} : TEST batches".format(n_test))
        for pyg_sample in tqdm(testload_pyg):
            
            pygtx, pygty, edges = pyg_sample.x.to(DEVICE) , pyg_sample.y.to(DEVICE), pyg_sample.edge_index.to(DEVICE)
            


            pygtest_y = roll(pyg_model,pygtx,t.to(DEVICE),edges,settings["type"])
            #print("TEST: shapes: {}, {}, {}, {}".format())

            testloss_pyg = loss_fn(pygtest_y[:,0,:],pygty[:,0,:]) + loss_fn(pygtest_y[:,1,:],pygty[:,1,:])

            loss_container[epoch,1] += testloss_pyg.item()

        loss_container[epoch,1]/=(n_test*len(t))

        print("E: {} DGL_HNN:: TRAIN: {} TEST: {}\n".format(epoch, loss_container[epoch,0], loss_container[epoch,1]))
        if eval_pack:
            visualize_eval("GGCNN",model,eval_pack[1],eval_pack[0],eval_pack[2])
    return pyg_model, loss_container
            
def visualize_loss(cont):
    fig = plt.figure()
    plt.semilogy(list(range(1,cont.shape[0]+1)),cont[:,0].cpu())
    plt.semilogy(list(range(1,cont.shape[0]+1)),cont[:,1].cpu())
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["train","test"])
    plt.savefig("loss.plt")

def visualize_eval(title,model,t,eval,edges):
    ground_truth = transform_Data_3dof(eval[:,0,:])
    y0 = ground_truth[:,:,0]
    ground_truth=ground_truth.transpose(0,2).transpose(1,2)
    print(y0.shape)
    y =roll(model.to(DEVICE),y0.to(DEVICE),t.to(DEVICE),edges.to(DEVICE),"rk4").transpose(0,2).transpose(1,2)
    #print(y0.shape)
    #y = odeint(ode_model,y0.to(DEVICE),t.to(DEVICE),method="rk4")
    fig, axes = plt.subplots(2,3)
    fig.suptitle(title, fontsize=20)
    axes[0,0].cla()
    axes[0,1].cla()
    axes[0,2].cla()
    axes[1,0].cla()
    axes[1,1].cla()
    axes[1,2].cla()
    axes[0,0].set_title("theta1")
    axes[0,1].set_title("theta2") 
    axes[0,2].set_title("theta3" )
    axes[1,0].set_title("gt theta1") 
    axes[1,1].set_title("gt theta2" )
    axes[1,2].set_title("gt theta3" )
    axes[0,0].plot(y[:,0,0].cpu().detach().numpy(),y[:,0,1].cpu().detach().numpy(),c="r")
    axes[0,1].plot(y[:,1,0].cpu().detach().numpy(),y[:,1,1].cpu().detach().numpy(),c="r")
    axes[0,2].plot(y[:,2,0].cpu().detach().numpy(),y[:,2,1].cpu().detach().numpy(),c="r")
    axes[0,0].plot(ground_truth[:,0,0].cpu().detach().numpy(),ground_truth[:,0,1].cpu().detach().numpy(),c="y",alpha=0.2)
    axes[0,1].plot(ground_truth[:,1,0].cpu().detach().numpy(),ground_truth[:,1,1].cpu().detach().numpy(),c="y",alpha=0.2)
    axes[0,2].plot(ground_truth[:,2,0].cpu().detach().numpy(),ground_truth[:,2,1].cpu().detach().numpy(),c="y",alpha=0.2)
    axes[1,0].plot(ground_truth[:,0,0].cpu().detach().numpy(),ground_truth[:,0,1].cpu().detach().numpy(),c="g")
    axes[1,1].plot(ground_truth[:,1,0].cpu().detach().numpy(),ground_truth[:,1,1].cpu().detach().numpy(),c="g")
    axes[1,2].plot(ground_truth[:,2,0].cpu().detach().numpy(),ground_truth[:,2,1].cpu().detach().numpy(),c="g")
    axes[0,0].legend(["prediction","ground truth"])
    axes[0,1].legend(["prediction","ground truth"])
    axes[0,2].legend(["prediction","ground truth"])
    axes[0,0].set_xlabel("q")
    axes[0,0].set_ylabel("p") 
    axes[0,1].set_xlabel("q")
    axes[0,1].set_ylabel("p")
    axes[0,2].set_xlabel("q")
    axes[0,2].set_ylabel("p")
    axes[1,0].set_xlabel("q")
    axes[1,0].set_ylabel("p") 
    axes[1,1].set_xlabel("q")
    axes[1,1].set_ylabel("p")
    axes[1,2].set_xlabel("q")
    axes[1,2].set_ylabel("p")
    plt.savefig("traj.plt")




    



    