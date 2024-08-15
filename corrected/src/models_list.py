import torch.nn as nn
import torch
from src.device_util import DEVICETensor
import src.device_util as device_util

"""activation functions for the mlp"""

def function_act(name):
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sin":
        return Sin()
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
        for i in range(1,len(acts)):
            modules.append(nn.Linear(hidden_dim,hidden_dim,bias=bias))
            if acts[i] != "":
                modules.append(function_act(acts[i]))
        modules.append(nn.Linear(hidden_dim,output_dim))
        if acts[i] != "":
            modules.append(function_act(acts[-1]))
        self.net = nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0.,std=0.1)
                nn.init.constant_(m.bias,val=0)
        
    def forward(self,y):
        return self.net(y.float())
    
class ode_mlp(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,hid_layers,acts,bias=True): # input layer,hidden_layersout,put_layer
        super(ode_mlp,self).__init__()
        MLP = mlp(input_dim,hidden_dim,output_dim,hid_layers,acts,bias)
        self.net = MLP.net
        self.trajectory = []
    
    def forward(self,t,y):
        return self.net(y.float())
    
class mlp_rollout(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,hid_layers,acts,bias=True):
        super(mlp_rollout,self).__init__()
        torch.autograd.set_detect_anomaly(True)
        MLP = mlp(input_dim,hidden_dim,output_dim,hid_layers,acts,bias).to(device_util.DEVICE)
        self.net = MLP.net

    def forward(self,t,y):
        if y.dim() == 3:#batch, more dim dataset
           
            self.trajectory = torch.Tensor(len(t),y.shape[0],1,y.shape[-1],).to(device_util.DEVICE)
            
            self.trajectory[0,:,:,:] = y.clone()
        else: # singledataset
           
            self.trajectory = DEVICETensor(len(t),1,y.shape[-1]).to(device_util.DEVICE)
    
            self.trajectory[0,:,:] = y.clone()
        for i in range(len(t)-1):
            
            if y.dim() == 3:
                ynow= self.trajectory[i,:,:,:].clone()
                self.trajectory[i+1,:,:,:] = self.net(ynow.float())
            else:
                ynow = self.trajectory[i,:,:].clone()
                self.trajectory[i+1,:,:] = self.net(ynow.float())
        return self.trajectory
        
        
        
        
class GRU_ODEINT(nn.Module):
    def __init__(self,in_feats,out_feats,hidden):
        super(GRU_ODEINT,self).__init__()

        self.gru_block = nn.GRU(in_feats,hidden,1).to(device_util.DEVICE)
        self.linear = nn.Linear(hidden,out_feats)
        self.act = nn.Tanh()
        self.in_feats = in_feats
        self.hidden_size = hidden

        nn.init.normal_(self.linear.weight,mean=0.,std=0.1)
        nn.init.constant_(self.linear.bias,val=0)
        nn.init.normal_(self.gru_block.weight_ih_l0,mean=0.,std=0.1)
        nn.init.normal_(self.gru_block.weight_hh_l0,mean=0.,std=0.1)
        nn.init.constant_(self.gru_block.bias_ih_l0,val=0)
        nn.init.constant_(self.gru_block.bias_hh_l0,val=0)

    def init_h0(self,batch_size=0):
        
        if batch_size == 0:
            return torch.zeros(1,self.hidden_size).to(device_util.DEVICE)
        
        else:
            return torch.zeros(1,batch_size,self.hidden_size).to(device_util.DEVICE)
        
    def forward(self,x_in,t):

        if x_in.dim()>2: 
            
            h_0 = self.init_h0(x_in.shape[1])
            self.trajectory = torch.Tensor(len(t),x_in.shape[1],x_in.shape[2]).to(device_util.DEVICE)
            self.trajectory[0,:,:] = x_in
        
        else:
            
            h_0 = self.init_h0()
            self.trajectory = torch.Tensor(len(t),x_in.shape[1]).to(device_util.DEVICE)
            self.trajectory[0,:] = x_in
            
            
            
            
        x_h, h = self.gru_block(x_in,h_0)

        x_out = self.linear(x_h)
        

        for i in range(len(t)-1):
            
            if x_in.dim()>2: 
                
                x_new= x_out
                self.trajectory[i+1,:,:] = x_new
            
            else:
                
                x_new = x_out
                self.trajectory[i+1,:] = x_new
            
            x_h, h = self.gru_block(x_new,h)
            x_out = self.linear(x_h)
        
        return self.trajectory
    

class GRU_ODEINT_EULER(nn.Module):
    def __init__(self,in_feats,out_feats,hidden):
        super(GRU_ODEINT_EULER,self).__init__()

        self.gru_block = nn.GRU(in_feats,hidden,1).to(device_util.DEVICE)
        self.linear = nn.Linear(hidden,out_feats)
        self.act = nn.Tanh()
        self.in_feats = in_feats
        self.hidden_size = hidden

        nn.init.normal_(self.linear.weight,mean=0.,std=0.1)
        nn.init.constant_(self.linear.bias,val=0)
        nn.init.normal_(self.gru_block.weight_ih_l0,mean=0.,std=0.1)
        nn.init.normal_(self.gru_block.weight_hh_l0,mean=0.,std=0.1)
        nn.init.constant_(self.gru_block.bias_ih_l0,val=0)
        nn.init.constant_(self.gru_block.bias_hh_l0,val=0)

    def init_h0(self,batch_size=0):
        
        if batch_size == 0:
            return torch.zeros(1,self.hidden_size).to(device_util.DEVICE)
        
        else:
            return torch.zeros(1,batch_size,self.hidden_size).to(device_util.DEVICE)
    
    def euler(self,x,dx,dt):
        return x + dx*dt
        
    def forward(self,x_in,t):

        if x_in.dim()>2: 
            
            h_0 = self.init_h0(x_in.shape[1])
            self.trajectory = torch.Tensor(len(t),x_in.shape[1],x_in.shape[2]).to(device_util.DEVICE)
            self.trajectory[0,:,:] = x_in
        
        else:
            
            h_0 = self.init_h0()
            self.trajectory = torch.Tensor(len(t),x_in.shape[1]).to(device_util.DEVICE)
            self.trajectory[0,:] = x_in
            
            
            
            
        x_h, h = self.gru_block(x_in,h_0)


        dx= self.linear(x_h)

        x_out = self.euler(x_in,dx,t[1]-t[0])
        

        for i in range(len(t)-1):
            
            if x_in.dim()>2: 
                
                x_new= x_out
                self.trajectory[i+1,:,:] = x_new
            
            else:
                
                x_new = x_out
                self.trajectory[i+1,:] = x_new
            
            x_h, h = self.gru_block(x_new,h)
            dx = self.linear(x_h)
            x_out = self.euler(x_out,dx,t[i+1]-t[i])
        
        return self.trajectory
    

class RNN_ODEINT(nn.Module):
    def __init__(self,in_feats,out_feats,hidden):
        super(RNN_ODEINT,self).__init__()

        self.rnn_block = nn.RNN(in_feats,hidden,1).to(device_util.DEVICE)
        self.linear = nn.Linear(hidden,out_feats)
        self.act = nn.Tanh()
        self.in_feats = in_feats
        self.hidden_size = hidden

        nn.init.normal_(self.linear.weight,mean=0.,std=0.1)
        nn.init.constant_(self.linear.bias,val=0)
        nn.init.normal_(self.rnn_block.weight_ih_l0,mean=0.,std=0.1)
        nn.init.normal_(self.rnn_block.weight_hh_l0,mean=0.,std=0.1)
        nn.init.constant_(self.rnn_block.bias_ih_l0,val=0)
        nn.init.constant_(self.rnn_block.bias_hh_l0,val=0)

    def init_h0(self,batch_size=0):
        
        if batch_size == 0:
            return torch.zeros(1,self.hidden_size).to(device_util.DEVICE)
        
        else:
            return torch.zeros(1,batch_size,self.hidden_size).to(device_util.DEVICE)
        
    def forward(self,x_in,t):

        if x_in.dim()>2: 
            
            h_0 = self.init_h0(x_in.shape[1])
            self.trajectory = torch.Tensor(len(t),x_in.shape[1],x_in.shape[2]).to(device_util.DEVICE)
            self.trajectory[0,:,:] = x_in
        
        else:
            
            h_0 = self.init_h0()
            self.trajectory = torch.Tensor(len(t),x_in.shape[1]).to(device_util.DEVICE)
            self.trajectory[0,:] = x_in
            
            
            
            
        x_h, h = self.rnn_block(x_in,h_0)

        x_out = self.linear(x_h)
        

        for i in range(len(t)-1):
            
            if x_in.dim()>2: 
                
                x_new= x_out
                self.trajectory[i+1,:,:] = x_new
            
            else:
                
                x_new = x_out
                self.trajectory[i+1,:] = x_new
            
            x_h, h = self.rnn_block(x_new,h)
            x_out = self.linear(x_h)
        
        return self.trajectory
    

