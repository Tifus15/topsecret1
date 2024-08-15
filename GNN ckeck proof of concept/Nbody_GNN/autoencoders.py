import torch
import torch.nn as nn
from mlp import *
from data_func import *
from torch.utils.data import DataLoader

class autoencoder_Nbody(nn.Module):
    def __init__(self,i, hidd,out,acts):
        super(autoencoder_Nbody,self).__init__()
        self.enc_q = mlp(i,hidd,out,acts)
        self.dec_q = mlp(out,hidd,i,acts)
        self.enc_p = mlp(i,hidd,out,acts)
        self.dec_p = mlp(out,hidd,i,acts)
    
    def auto_q(self,q):
        enc = self.enc_q(q)
        out = self.dec_q(enc)
        return out
    
    def auto_p(self,p):
        enc = self.enc_p(p)
        out = self.dec_p(enc)
        return out
    
    def forward(self,q,p):
        newq = self.auto_q(q)
        newp = self.auto_p(p)

        return newq, newp
    



fileH = "data/nbody_4_H.pt"
filet = "data/nbody_4_traj.pt"

x , H =load_dataset(filet,fileH)


x_orig = x
print(x.shape)
print(H.shape)
feat_dim = x.shape[-1]

q_reg, qmax, qmin = minmax(x[:,:,:,0:int(feat_dim/2)]) 
p_reg, pmax, pmin = minmax(x[:,:,:,int(feat_dim/2):]) 

x_reg = torch.cat((q_reg,p_reg),dim=-1)
x = x_reg
SPLIT = 0.9
EPOCHS = 250

inits_q = []
inits_p = []
for i in range(x.shape[1]):
    for j in range(x.shape[0]):
        inits_q.append(x[j,i,:,0:3].float())
        inits_p.append(x[j,i,:,3:6].float())

borderqp = int(SPLIT*len(inits_q))

trainq = inits_q[0:borderqp]
trainp = inits_p[0:borderqp]
testq = inits_q[borderqp:]
testp = inits_p[borderqp:]

#enc_q = mlp(3,128,1,["relu","tanh",""])
#dec_q = mlp(1,128,3,["relu","tanh",""])
#enc_p = mlp(3,128,1,["relu","tanh",""])
#dec_p = mlp(1,128,3,["relu","tanh",""])
model = autoencoder_Nbody(3,128,1,["relu"])

opti = torch.optim.Adam(model.parameters(),lr=1e-5)
#opti_enc_q = torch.optim.AdamW(enc_q.parameters(),lr=1e-3)
#opti_enc_p = torch.optim.AdamW(enc_p.parameters(),lr=1e-3)
#opti_dec_q = torch.optim.AdamW(dec_q.parameters(),lr=1e-3)
#opti_dec_p = torch.optim.AdamW(dec_p.parameters(),lr=1e-3)

BATCHSIZE = 128

trainloader_q = DataLoader(trainq,BATCHSIZE,shuffle=True)
trainloader_p = DataLoader(trainp,BATCHSIZE,shuffle=True)
testloader_q = DataLoader(testq,BATCHSIZE)
testloader_p = DataLoader(testp,BATCHSIZE)
lossfn= nn.MSELoss()
Ntrain = len(trainloader_p)
Ntest = len(testloader_p)

for epoch in range(EPOCHS):
    logp = 0
    logq = 0
    print(len(trainloader_q))
    for batchq, batchp in zip(trainloader_q,trainloader_p):
        #opti_enc_p.zero_grad()
        #opti_enc_q.zero_grad()
        #opti_dec_q.zero_grad()
        #opti_dec_p.zero_grad()

        opti.zero_grad()

        q_hat, p_hat = model(batchq.squeeze(),batchp.squeeze())

        #general_q = enc_q(batchq)
        #general_p = enc_p(batchp)

        #q_hat = dec_q(general_q)
        #p_hat = dec_p(general_p)

        lossq = lossfn(q_hat,batchq.squeeze())
        lossp = lossfn(p_hat,batchp.squeeze())

        logq += lossq.item()
        logp += lossp.item()
        loss = lossq + lossp
        loss.backward()
        #lossp.backward()

        #opti_enc_p.step()
        #opti_dec_p.step()
        #opti_enc_q.step()
        #opti_dec_q.step()
        opti.step()
    logq/=Ntrain
    logp/=Ntrain

    logtq =0
    logtp =0
    print(len(testloader_q))
    for batchqt, batchpt in zip(testloader_q,testloader_p):
        

        #general_q = enc_q(batchq)
        #general_p = enc_p(batchp)

        #q_hat = dec_q(general_q)
        #p_hat = dec_p(general_p)
        qt_hat, pt_hat = model(batchq.squeeze(),batchp.squeeze())
        lossqt = lossfn(qt_hat,batchqt.squeeze())
        losspt = lossfn(pt_hat,batchpt.squeeze())

        logtq += lossqt.item()
        logtp += losspt.item()
    
    logtq = logtq/Ntest
    logtp/=Ntest
    dict={}
    for name, parm in model.named_parameters():
        dict[name+"_grad"]=torch.mean(parm.grad)
    print("EPOCH {} \nautoencq train {} test {}\nautoencp train {} test {} ".format(epoch,logq,logtq,logp,logtp))
    print(dict)
    
datasetq = trainq + testq
datasetp = trainp + testp

N = len(datasetq)

print(N)
maxq = 0
maxp = 0
for i in range(N):
    batchq = datasetq[i]
    batchp = datasetp[i]
    predq,predp = model(batchq,batchp)
    a = torch.linalg.vec_norm(predq-datasetq[i])
    b = torch.linalg.vec_norm(predp-datasetp[i])
    if a > maxq:
        maxq = a
    if b > maxp:
        maxp = b

print("max error radius for q: {} and for p {}".format(maxq,maxp))
    
    




















