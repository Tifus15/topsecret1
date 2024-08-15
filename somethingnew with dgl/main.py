import torch
#from models import GGCNN_HNN, PortHamModel
from models import *
from datasets import make3dofBaseDataset,makeDGL3dofDataset,transformFromGraph
from device_util import ROOT_PATH, DEVICE
from dof3_pendelum_torch import *
import os
import random
from train import visualize_loss,visualize_eval,training
import dgl



print("The benchamark is using: {}".format(DEVICE))

dataset_settings={"dt" : 0.01,
                  "T" : 5,
                  "samples" : 50,
                  "pi_range" : [-torch.pi/2.5,torch.pi/2.5]}
filename = "trajdof3.pt"



#50 samples got me 4h of waiting time- ALWAYS SAVE THE SAMPLES!!!!!
######################################################## 
if not os.path.isfile(ROOT_PATH + "/"+ filename):
    dataset_base = make3dofBaseDataset(dataset_settings)
    torch.save(dataset_base,filename)
else:
    dataset_base = torch.load(ROOT_PATH + "/"+ filename)
print(dataset_base.shape)
########################################################

edge_list = [[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]]
g = dgl.graph(([0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2])) # fully connected graph

sample_id = random.randint(0,50)

t_size = 150
t_batchsize= 50
data = dataset_base[0:t_size,sample_id,:,:] 
H = eval_ham(data)
print(H)
sc_H = H[0]
# cut single sample

#cutted_data = makePyG3dofDataset(edges,data.unsqueeze(1),t_batchsize)
cutted_data = makeDGL3dofDataset(edge_list,data.unsqueeze(1),t_batchsize)

print(len(cutted_data))

# shuffle and make train and test samples
random.shuffle(cutted_data)
cut = int(len(cutted_data)*0.9)
train = cutted_data[:cut]
test= cutted_data[cut:]

print(len(train))
print(len(test))

t = torch.linspace(0,dataset_settings["T"],501)[0:t_size]
t_b = t[0:t_batchsize]

#model = GGCNN_HNN(2,grad=False).to(DEVICE)
#model = dgl_HNN(edges,2,300).to(DEVICE)
#model = PortHamModel(model)
model = PortHNN(g,2,1000,1).to(DEVICE)

sim_settings = {"epochs" : 100,
                "opti" : "RMS",
                "loss" : "Huber",
                "lr" : 0.5,
                "batch" : 1,
                "type" : "rk4"}#look at console and calculate optimal batch size

model_dgl_trained,  loss_container = training(sim_settings,train,test,model,t_b,sc_H)
torch.save(model_dgl_trained.state_dict(),"model_single_pc.pt")
visualize_loss(loss_container)
model_dgl_trained.g=g 
visualize_eval("GATPortHNN_single", model_dgl_trained, t, data)

"""
model_pyg_trained,  loss_container = training_pyg(sim_settings,train,test,model,t_b)
torch.save(model_pyg_trained.state_dict(),"model_single.pt")
visualize_loss(loss_container)
visualize_eval("GGCNN_single", model, t, data,torch.tensor(edges,dtype=torch.int64))
"""