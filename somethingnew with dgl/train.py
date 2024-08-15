import torch
import torch.nn as nn
import torch.optim as optim
import random
#from model_bench import ode_model_eval
from models import roll, H_roll

#from torch_geometric.loader import DataLoader
from device_util import DEVICE

from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import transform_Data_3dof
import math
from dgl.dataloading import GraphDataLoader

def training(settings, train_set, test_set,model,t,H=None,eval_pack=None):
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
            dgl_model.g = dgl_sample
           
            
            dglx, dgly, edges = dgl_sample.ndata["x"].to(DEVICE) , dgl_sample.ndata["y"].to(DEVICE), dgl_sample.to(DEVICE)
            dgl_model.g.ndata["x"] , dgl_model.g.ndata["y"] = dgl_sample.ndata["x"].to(DEVICE),dgl_sample.ndata["y"].to(DEVICE)
            #print("TRAIN x:{}".format(dglx.shape))
            dgltrain_y = roll(dgl_model,dglx.requires_grad_(),t.to(DEVICE),settings["type"])
            if H is None:
                H_train = None
            else:
                H_train = H_roll(dgl_model,dgltrain_y)
                #print("H_train {}".format(len(H_train)))
                #print("H {}".format(len(H)))
                



            #print("TRAIN: shapes: {}, {}, {}, {}".format(pygy.shape,pygtrain_y.shape,dgly.shape,dgltrain_y.shape))
            loss_dgl = dgl_model.calculate_loss(dgltrain_y,H_train,H*torch.ones(H_train.shape),func=loss_fn)

            loss_container[epoch,0] += loss_dgl.item()

            
            loss_dgl.backward()

           
            optimizer_dgl.step()
        
        loss_container[epoch,0]/=(n_train*(len(t)+1))

        testload_dgl = GraphDataLoader(test_dgl,batch_size=settings["batch"])
    
        n_test = len(testload_dgl)



        dgl_model.eval()
 
        print("{} : TEST batches".format(n_test))
        for dgl_sample in tqdm(testload_dgl):
            dgltx, dglty, edges = dgl_sample.ndata["x"].to(DEVICE) , dgl_sample.ndata["y"].to(DEVICE), dgl_sample.to(DEVICE)
            dgl_model.g.ndata["x"] , dgl_model.g.ndata["y"] = dgl_sample.ndata["x"].to(DEVICE),dgl_sample.ndata["y"].to(DEVICE)


            dgltest_y = roll(dgl_model,dgltx.requires_grad_(),t.to(DEVICE),settings["type"])
            
            if H is None:
                H_test = None
            else:
                H_test = H_roll(dgl_model,dgltest_y)
                #print("H_train {}".format(len(H_test)))
                #print("H {}".format(len(H)))

            testloss_dgl = dgl_model.calculate_loss(dgltest_y,H_test,H*torch.ones(H_train.shape),func=loss_fn)

            loss_container[epoch,1] += testloss_dgl.item()
       
        loss_container[epoch,1]/=(n_test*len(t))
        print("E: {} PORT_HNN:: TRAIN: {} TEST: {}\n".format(epoch, loss_container[epoch,0], loss_container[epoch,1]))
        if eval_pack:
            visualize_eval("PORTHNN",model,eval_pack[1],eval_pack[0],eval_pack[2])
    return dgl_model, loss_container

def visualize_loss(cont):
    fig = plt.figure()
    plt.semilogy(list(range(1,cont.shape[0]+1)),cont[:,0].cpu())
    plt.semilogy(list(range(1,cont.shape[0]+1)),cont[:,1].cpu())
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["train","test"])
    plt.show()

def visualize_eval(title,model,t,eval):
    ground_truth = transform_Data_3dof(eval[:,0,:])
    y0 = ground_truth[:,:,0].requires_grad_()
    ground_truth=ground_truth.transpose(0,2).transpose(1,2)
    print(y0.shape)
    y =roll(model.to(DEVICE),y0.to(DEVICE),t.to(DEVICE),"rk4").transpose(0,2).transpose(1,2)
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
    plt.show()