import yaml
import torch
from device_util import ROOT_PATH
#from oscilator import *
#from twobody import *
#from threebody import *
#from samples import SAMPLES
import matplotlib.pyplot as plt

def visualize_loss(title, loss_container):
    if loss_container.shape[0]==2:
        t = torch.linspace(0,loss_container.shape[1],loss_container.shape[1])
        fig = plt.figure()
        plt.title("{} logy".format(title))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.semilogy(t,loss_container[0,:],c="r")
        plt.semilogy(t,loss_container[1,:],c="b")
        plt.legend(["train","test"])
        plt.show()
    else:
        t = torch.linspace(0,loss_container.shape[1],loss_container.shape[1])
        fig,ax = plt.subplots(1,2)
        ax[0].set_title("{} loss".format(title))
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("loss")
        ax[0].semilogy(t,loss_container[0,:],c="r")
        ax[0].semilogy(t,loss_container[2,:],c="b")
        ax[0].legend(["train","test"])
        ax[1].set_title("{} gradient loss".format(title))
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("loss")
        ax[1].semilogy(t,loss_container[1,:],c="r")
        ax[1].semilogy(t,loss_container[3,:],c="b")
        ax[1].legend(["train grad","test grad"])
        

def visualize_hamiltonian(H_ground, H,t):
    fig = plt.figure()
    plt.title("{} plot".format("hamiltonian"))
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.scatter(t,H.detach().numpy(),c="r")
    plt.scatter(t,H_ground.detach().numpy(),c="b")
    plt.legend(["prediction","ground truth"])
    plt.show()  

def viz_traj(ground,model_eval,type):
    ground = ground.squeeze()
    model_eval = model_eval.squeeze()
    if type == "dof2":
        fig, ax = plt.subplots(2)
        ax[0].scatter(model_eval[0,:,0].detach().numpy(),model_eval[0,:,1].detach().numpy())
        ax[0].scatter(ground[0,:,0].detach().numpy(),ground[0,:,1].detach().numpy())
        ax[0].legend(["prediction","ground_truth"])
        ax[1].scatter(model_eval[1,:,0].detach().numpy(),model_eval[1,:,1].detach().numpy())
        ax[1].scatter(ground[1,:,0].detach().numpy(),ground[1,:,1].detach().numpy())
        ax[1].legend(["prediction","ground_truth"])
        
    if type == "dof3":
        fig, ax = plt.subplots(3)
        ax[0].scatter(model_eval[0,:,0].detach().numpy(),model_eval[0,:,1].detach().numpy())
        ax[0].scatter(ground[0,:,0].detach().numpy(),ground[0,:,1].detach().numpy())
        ax[0].legend(["prediction","ground_truth"])
        ax[1].scatter(model_eval[1,:,0].detach().numpy(),model_eval[1,:,1].detach().numpy())
        ax[1].scatter(ground[1,:,0].detach().numpy(),ground[1,:,1].detach().numpy())
        ax[1].legend(["prediction","ground_truth"])
        ax[2].scatter(model_eval[2,:,0].detach().numpy(),model_eval[2,:,1].detach().numpy())
        ax[2].scatter(ground[2,:,0].detach().numpy(),ground[2,:,1].detach().numpy())
        ax[2].legend(["prediction","ground_truth"])
    
    plt.show()

def make_snapshots(data, TIME_SIZE):
    time_size =TIME_SIZE
    points = data.shape[0]
    print("{} snapshots".format((points-time_size)*data.shape[1]))
    list = []
    
    
    for i in range(data.shape[1]):
        for j in range(points-time_size):
            list.append(data[j:j+time_size,i,:,:])
                
    return list

def make_graph_snapshots(datalist,nodes,feats):
    graph_data = []
    for sample in datalist:
        #print(sample.shape)   
        qp = torch.split(sample,feats,dim=-1)
        q_p = qp[0]
        dqp = qp[1]
        H = qp[2].squeeze(0)
        H = H.unsqueeze(0).reshape(1,-1,1)
        
        #print(H.shape)
        qp_l = torch.split(q_p,1,dim=-1)
        dqp_l = torch.split(dqp,1,dim=-1)
        #print(len(dqp_l))
        #print(qp_l[0].shape)
        H_list = []
        
        nodes_list= []
        for i in range(nodes):
            q = qp_l[i].squeeze().unsqueeze(0).unsqueeze(-1)
            p = qp_l[nodes + i].squeeze().unsqueeze(0).unsqueeze(-1)
            dq = qp_l[i].squeeze().unsqueeze(0).unsqueeze(-1)
            dp = qp_l[nodes + 1].squeeze().unsqueeze(0).unsqueeze(-1) 
            temp = torch.cat((q,p,dq,dp,H),dim=-1)
            #print(temp.shape)
            nodes_list.append(temp)
        graph_data.append(torch.cat((nodes_list),dim=0))
      
    return graph_data
            



def open_yamls(name):
    with open(ROOT_PATH+"/yamls/"+name) as stream:
        try:
            data_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data_dict


def make_graph_dataset(dict):
    type = dict["type"]
    if type == "dof2pend":
        dataset = torch.load("traj_2dof.pt")
        return dataset
    if type == "dof3pend":
        dataset = torch.load("traj_3dof.pt")
        return dataset
    


def dof2_hamiltonian(traj):
    x1 = torch.sin(traj[0,:,0])
    x2 = x1 + torch.sin(traj[1,:,0])
    y1 = -torch.cos(traj[0,:,0])
    y2 = y1-torch.cos(traj[1,:,0])
    dx1 = traj[0,:,1] * torch.cos(traj[0,:,0])
    dy1 = traj[0,:,1] * torch.sin(traj[0,:,0])
    dx2 = traj[1,:,1] * torch.cos(traj[1,:,0])
    dy2 = traj[1,:,1] * torch.sin(traj[1,:,0])
    vs1 = dx1.pow(2) + dy2.pow(2)
    vs2 = dx1.pow(2) + dy2.pow(2) 
    kin = 0.5 * (vs1+vs2)
    #print(kin.shape)
    pot = 9.81*(y1 + torch.Tensor([1.0])) + 9.81*(y2 + torch.Tensor([2.0]))
    #print(pot.shape)
    return kin + pot

def dof3_hamiltonian(traj):
    x1 = torch.sin(traj[0,:,0])
    x2 = x1 + torch.sin(traj[1,:,0])
    x3 = x2 + torch.sin(traj[2,:,0])
    y1 = -torch.cos(traj[0,:,0])
    y2 = y1-torch.cos(traj[1,:,0])
    y3 = y1-torch.cos(traj[2,:,0])
    dx1 = traj[0,:,1] * torch.cos(traj[0,:,0])
    dy1 = traj[0,:,1] * torch.sin(traj[0,:,0])
    dx2 = traj[1,:,1] * torch.cos(traj[1,:,0])
    dy2 = traj[1,:,1] * torch.sin(traj[1,:,0])
    dx3 = traj[2,:,1] * torch.cos(traj[2,:,0])
    dy3 = traj[2,:,1] * torch.sin(traj[2,:,0])
    vs1 = dx1.pow(2) + dy1.pow(2)
    vs2 = dx2.pow(2) + dy2.pow(2)
    vs3 = dx3.pow(2) + dy3.pow(2)
    kin = 0.5 * (vs1+vs2+vs3)
    #print(kin.shape)
    pot = 9.81*(y1 + torch.Tensor([1.0])) + 9.81*(y2 + torch.Tensor([2.0]))+ 9.81*(y3 + torch.Tensor([3.0]))
    #print(pot.shape)
    return kin + pot
    







def make_dataset(dict):
    type = dict["type"]
    if type == "oscilator":
        print("OSCILATOR DATASET")
        points = dict["points"]
        samples = dict["samples"]
        data_maker = oscilator(dict["m"],dict["k"])
        H_span = dict["H_span"]
        #print(H_span)
        data ,ddata, t, H = data_maker.make_dataset(points,samples,(H_span[0],H_span[1]))
        func = data_maker.hamiltonian
        return data,ddata, t, H, func
    elif type == "twobody":
        print("TWOBODY DATASET")
        points = dict["points"]
        samples = dict["samples"]
        data_maker = twobody(dict["m1"],dict["m2"],dict["G"])
        data,ddata, t, H = data_maker.make_dataset(points,samples,T=dict["T"],r1_range=(dict["r1_range"][0],dict["r1_range"][1]))
        func =data_maker.hamiltonian
        return data,ddata, t, H, func
    elif type == "threebody":
        print("THREEBODY DATASET")
        points = dict["points"]
        samples = dict["samples"]
        sample = dict["sample"]
        data_maker = threebody(SAMPLES)
        data,ddata, t, H = data_maker.dataset_onekind(sample,samples,points,phi_span=(dict["phi_span"][0],dict["phi_span"][1]))
        func = data_maker.hamiltonian
        return data ,ddata, t, H, func
