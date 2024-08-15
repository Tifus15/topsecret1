import torch
import device_util
import torch.nn as nn
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
            return torch.zeros(1,self.hidden_size)#.to(device_util.DEVICE)
        
        else:
            return torch.zeros(1,batch_size,self.hidden_size)#.to(device_util.DEVICE)
    
    def euler(self,x,dx,dt):
        return x + dx*dt
        
    def forward(self,x_in,t):

        if x_in.dim()>2: 
            
            h_0 = self.init_h0(x_in.shape[1])
            trajectory = torch.Tensor(len(t),x_in.shape[1],x_in.shape[2])#.to(device_util.DEVICE)
            dtrajectory = torch.Tensor(len(t),x_in.shape[1],x_in.shape[2])#.to(device_util.DEVICE)
            trajectory[0,:,:] = x_in
        
        else:
            
            h_0 = self.init_h0()
            trajectory = torch.Tensor(len(t),x_in.shape[1])#.to(device_util.DEVICE)
            dtrajectory = torch.Tensor(len(t),x_in.shape[1])#.to(device_util.DEVICE)
            trajectory[0,:] = x_in
            
            
            
            
        x_h, h = self.gru_block(x_in,h_0)


        dx= self.linear(x_h)
        if dx.dim()>2:
            dtrajectory[0,:,:] = dx
        else: 
            dtrajectory[0,:] = dx
            
        x_out = self.euler(x_in,dx,t[1]-t[0])
        

        for i in range(len(t)-1):
            
            if x_in.dim()>2: 
                
                x_new= x_out
                trajectory[i+1,:,:] = x_new
            
            else:
                
                x_new = x_out
                trajectory[i+1,:] = x_new
            
            x_h, h = self.gru_block(x_new,h)
            dx = self.linear(x_h)
            
            if dx.dim()>2:
                dtrajectory[i,:,:] = dx
            else: 
                dtrajectory[i,:] = dx
                
            x_out = self.euler(x_out,dx,t[i+1]-t[i])
        x_h, h = self.gru_block(x_new,h)
        dx = self.linear(x_h)
        if dx.dim()>2:
            dtrajectory[-1,:,:] = dx
        else: 
            dtrajectory[-1,:] = dx
        
        return torch.cat((trajectory,dtrajectory),dim=-1)#self.trajectory