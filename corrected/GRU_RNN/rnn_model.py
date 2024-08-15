import torch.nn as nn
import device_util
import torch
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