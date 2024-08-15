import torch.nn as nn
import torch 

def function_act(name):
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sin":
        return Sin()
    if name == "softplus":
        return nn.Softplus()
    else:
        return nn.Identity()

"""sin activation function as torch module"""
    
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(1.0 * x)

"""MLP module"""
class mlp(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,acts,bias=True): # input layer,hidden_layersout,put_layer
        super(mlp,self).__init__()
        modules = []
        modules.append(nn.Linear(input_dim,hidden_dim))
        modules.append(function_act(acts[0]))
        for i in range(1,len(acts)-1):
            modules.append(nn.Linear(hidden_dim,hidden_dim,bias=bias))
            if acts[i] != "":
                modules.append(function_act(acts[i]))
        modules.append(nn.Linear(hidden_dim,output_dim))
        self.net = nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0.,std=0.1)
                nn.init.constant_(m.bias,val=0)
        
    def forward(self,t,y):
        return self.net(y.float())
    
def rollout_mlp_vec(model,x):
    out_list = []
    for i in range(x.shape[0]):
        temp = model(0,x[i,:,:,:]).unsqueeze(0)
        #print(temp.shape)
        out_list.append(temp)
    return torch.cat((out_list),dim=0)