import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import sys

import yaml

from device_util import ROOT_PATH, DEVICE
import torch
#from models import GGCNN_HNN, PortHamModel
from dglmodel import *
from datasets import make3dofBaseDataset,makeDGL3dofDataset
from device_util import ROOT_PATH, DEVICE
import os
import random
from train import visualize_loss,visualize_eval,training_dgl


from experiment_launcher import run_experiment, single_experiment_yaml

@single_experiment_yaml
def experiment(
    #######################################
    config_file_path: str = './config/GGCNN.yaml',

    some_default_param: str = 'b',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = 41,
    results_dir: str = 'logs_osci',

    #######################################
    # OPTIONAL
    # accept unknown arguments
    **kwargs
):
    # EXPERIMENT
    print(f'DEBUG MODE: {debug}')

    with open(config_file_path, 'r') as f:
        configs = yaml.load(f, yaml.Loader)

    print('Config file content:')
    print(configs)

    ## MY EXPERIMENT
    

    filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    out_str = f'Running experiment with seed {seed} and with device {device_util.DEVICE}'
    with open(filename, 'w') as file:
        file.write('Some logs in a log file.\n')
        file.write(out_str)

    dataset_base = torch.load(ROOT_PATH + "/"+ filename)
    print(dataset_base.shape)

    edges = [[0,0,1,1,2],[0,1,1,2,2]]

    sample_id = random.randint(0,50)

    t_size = configs["t_size"]
    t_batchsize= configs["t_batch"]
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

    t = torch.linspace(0,configs["T"],501)[0:t_size]
    t_b = t[0:t_batchsize]
    
    model=portHNN_split_dgl(edges,2,100,25).to(DEVICE)

    sim_settings = {"epochs" : configs["epochs"],
                    "opti" : configs["opti"],
                    "loss" : configs["loss"],
                    "lr" : configs["lr"],
                    "batch" : configs["batch"],
                    "type" : "rk4"}
    
    model_dgl_trained,  loss_container = training_dgl(sim_settings,train,test,model,t_b)
    torch.save(model_dgl_trained.state_dict(),"model_single.pt")



    

    