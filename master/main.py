import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import *
from main_body import *
import yaml


import wandb

from experiment_launcher import run_experiment, single_experiment_yaml


# This decorator creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment_yaml
def experiment(
    #######################################
    config_file_path: str = './configs/nbody.yaml',

    some_default_param: str = 'b',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = 41,
    results_dir: str = 'logs_nbody',

    #######################################
    # OPTIONAL
    # accept unknown arguments
    **kwargs
):
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
    
    full(configs)
    wandb.finish()
    
if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)