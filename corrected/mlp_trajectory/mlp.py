import torch.nn as nn
import torch 
import device_util

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
        
    def forward(self,y):
        return self.net(y.float())
    
def rollout_mlp(model,x0,t):
    out_list = []
    #print(x0.shape)
    out_list.append(x0.unsqueeze(0))
    for i in range(1,len(t)):
        temp = model(out_list[-1])
        #print(temp.shape)
        out_list.append(temp)
    return torch.cat((out_list),dim=0)

class mlp_rollout(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,acts,bias=True):
        super(mlp_rollout,self).__init__()
        torch.autograd.set_detect_anomaly(True)
        MLP = mlp(input_dim,hidden_dim,output_dim,acts,bias).to(device_util.DEVICE)
        self.net = MLP.net

    def forward(self,t,y):
        if y.dim() == 3:#batch, more dim dataset
           
            self.trajectory = torch.Tensor(len(t),y.shape[0],1,y.shape[-1],).to(device_util.DEVICE)
            
            self.trajectory[0,:,:,:] = y.clone()
        else: # singledataset
           
            self.trajectory = device_util.DEVICETensor(len(t),1,y.shape[-1]).to(device_util.DEVICE)
    
            self.trajectory[0,:,:] = y.clone()
        for i in range(len(t)-1):
            
            if y.dim() == 3:
                ynow= self.trajectory[i,:,:,:].clone()
                self.trajectory[i+1,:,:,:] = self.net(ynow.float())
            else:
                ynow = self.trajectory[i,:,:].clone()
                self.trajectory[i+1,:,:] = self.net(ynow.float())
        return self.trajectory
    
        