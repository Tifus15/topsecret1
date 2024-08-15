import torch
#from models import GGCNN_HNN, PortHamModel
from dglmodel import *
from datasets import make3dofBaseDataset, makePyG3dofDataset,makeDGL3dofDataset
from device_util import ROOT_PATH, DEVICE
import os
import random
from train import training_pyg,visualize_loss,visualize_eval,training_dgl



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


edges = [[0,0,1,1,2],[0,1,1,2,2]]

sample_id = random.randint(0,50)

t_size = 150
t_batchsize= 50
data = dataset_base[0:t_size,sample_id,:,:] 
# cut single sample

#cutted_data = makePyG3dofDataset(edges,data.unsqueeze(1),t_batchsize)
cutted_data = makeDGL3dofDataset(edges,data.unsqueeze(1),t_batchsize)

print(len(cutted_data))

# shuffle and make train and test samples
random.shuffle(cutted_data)
cut = int(len(cutted_data)*0.8)
train = cutted_data[:cut]
test= cutted_data[cut:]

print(len(train))
print(len(test))

t = torch.linspace(0,dataset_settings["T"],501)[0:t_size]
t_b = t[0:t_batchsize]

#model = GGCNN_HNN(2,grad=False).to(DEVICE)
#model = dgl_HNN(edges,2,300).to(DEVICE)
#model = PortHamModel(model)
model=portHNN_split_dgl(edges,2,100,25).to(DEVICE)

sim_settings = {"epochs" : 500,
                "opti" : "RMS",
                "loss" : "MSE",
                "lr" : 0.001,
                "batch" : 4,
                "type" : "rk4"}#look at console and calculate optimal batch size

model_dgl_trained,  loss_container = training_dgl(sim_settings,train,test,model,t_b)
torch.save(model_dgl_trained.state_dict(),"model_single.pt")
visualize_loss(loss_container)
visualize_eval("DGLHNN_single", model, t, data,dgl.graph((edges[0],edges[1])))

"""
model_pyg_trained,  loss_container = training_pyg(sim_settings,train,test,model,t_b)
torch.save(model_pyg_trained.state_dict(),"model_single.pt")
visualize_loss(loss_container)
visualize_eval("GGCNN_single", model, t, data,torch.tensor(edges,dtype=torch.int64))
"""

