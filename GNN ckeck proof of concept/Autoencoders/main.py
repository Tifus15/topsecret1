import torch
import torch.nn as nn
import torch.optim as opti
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *


class Autoencoder(nn.Module):
    def __init__(self,in_dim,hidden,out_dim):
        super(Autoencoder,self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim,hidden),
            nn.Tanh(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.Tanh(),
            nn.Linear(hidden,out_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(out_dim,hidden),
            nn.Tanh(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.Tanh(),
            nn.Linear(hidden,in_dim),
        )
        
    def encoder(self,x):
        return self.enc(x)
    
    def decoder(self,x):
        return self.dec(x)
    
    def forward(self,x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z
    
def train(dataset,EPOCHS):
    x = dataset
    dataset_list=[]
    for i in range(x.shape[0]):
        dataset_list.append(x[i,:,:])
    random.shuffle(dataset_list)
    SPLIT = 0.9
    N = len(dataset_list)
    breakingpoint = int(N*SPLIT)
    
    train = dataset_list[0:breakingpoint]
    test = dataset_list[breakingpoint:]
    
    model= Autoencoder(4,128,2)
    
    optimizer = opti.Adam(model.parameters(),lr=1e-4)
    lossfn = nn.MSELoss()
    
    
    
   
    for epoch in tqdm(range(EPOCHS)):
        train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test, batch_size=64, shuffle=True)
        loss_acc = 0
        loss_acct = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            x = model(batch)
            loss = lossfn(x,batch)
            loss_acc += loss.item()
            loss.backward()
            optimizer.step()
            
            
        for t in tqdm(test_dataloader):
            
            x = model(t)
            loss = lossfn(x,t)
            loss_acct += loss.item()
        print("EPOCH {} | train {} | test {}".format(epoch+1,loss_acc/len(train_dataloader),loss_acct/len(test_dataloader)))
    
    return model  
    
    
    
        
x= torch.load("x6b1000.pt")
x,max,min=minmax(x)   
y = x.reshape(-1,6,4) 
print(torch.max(y.flatten()))  
print(torch.min(y.flatten()))   
    
model = train(y,10)

random_point = y[random.randint(0,y.shape[0]-1),:,:]

print(random_point)
print(model(random_point))
print(model.encoder(random_point))
print(torch.mean(torch.abs(random_point-model(random_point))))


