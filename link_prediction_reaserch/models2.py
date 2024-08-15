import torch.nn as nn
import dgl.function as fn
import torch
import dgl
import dgl.nn as dglnn
import torch.nn.functional as F

class HNN(nn.Module):
    def __init__(self,GraphNN_model):
        super().__init__()
        self.base = GraphNN_model

    def forward(g, x):
        H_layer = self.base(x)
        H = H_layer.sum()

        dH = torch.autograd.grad(H,x,ret
