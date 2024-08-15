import torch
from tqdm import tqdm
from model_gnn import *
from data_func import *
import dgl
from dgl.dataloading import GraphDataLoader
from torchdiffeq import odeint_adjoint as odeint
import random
import matplotlib.pyplot as plt
NO_LOOPS = True
SPLIT = 0.9
TIMESIZE = 127
BATCHSIZE = 8
EPOCHS = 100
REG=True
a=0.1
S = 100

if NO_LOOPS:
    src5 = [1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3]
    dst5 = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
    graph5 = dgl.graph((src5,dst5))
    
else:
    src5 = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
    dst5 = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
    graph5 = dgl.graph((src5,dst5))

if NO_LOOPS:
    src4 = [1,2,3,0,2,3,0,1,3,0,1,2]
    dst4 = [0,0,0,1,1,1,2,2,2,3,3,3]
    graph4 = dgl.graph((src4,dst4))
    
else:
    src4 = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    dst4 = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    graph4 = dgl.graph((src4,dst4))



data1 , H1, data2 , H2  =dataset_loader()
if REG:
    data = torch.cat((data1,data2),dim=2)

    q_reg, qmaxim, qminim = minmax(data[:,:,:,0:3])
    p_reg, pmaxim, pminim = minmax(data[:,:,:,3:6])
    dq_reg, dqmaxim, dqminim = minmax(data[:,:,:,6:9])
    dp_reg, dpmaxim, dpminim = minmax(data[:,:,:,9:12])
    print(q_reg.shape)
    print(p_reg.shape)
    print(dq_reg.shape)
    print(dp_reg.shape)


    reg4 = torch.cat((q_reg[:,:,0:4,:],p_reg[:,:,0:4,:],dq_reg[:,:,0:4,:],dp_reg[:,:,0:4,:]),dim=-1)
    print(torch.max(reg4.flatten()))
    print(torch.min(reg4.flatten()))

    reg5 = torch.cat((q_reg[:,:,4:9,:],p_reg[:,:,4:9,:],dq_reg[:,:,4:9,:],dp_reg[:,:,4:9,:]),dim=-1)
    print(torch.max(reg5.flatten()))
    print(torch.min(reg5.flatten()))

    x1_eval = reg4[:,-1,:,0:6]
    x2_eval = reg5[:,-1,:,0:6]
    print(reg4.shape)
    x1 = reg4[:,0:S,:,:]
    x2 = reg5[:,0:S,:,:]
else:
    x1_eval = data1[:,-1,:,0:6]
    x2_eval = data2[:,-1,:,0:6]
    #print(reg4.shape)
    x1 = data1[:,0:S,:,:]
    x2 = data2[:,0:S,:,:]

H1 = H1[:,0:S]
H2 = H2[:,0:S]
# the first dataset will be used fully as training and test set
t = torch.linspace(0,1.27,128) # dt = 0.01
print(t[1]-t[0])

model = GATGNN(graph4,["sin"],6,6,[128],True)
print(model)

opti = torch.optim.Adam(model.parameters(),lr=1e-5)
lossfn = torch.nn.MSELoss()

print(H1.shape)

xs, hs = make_snapshots(x1.float(),H1.reshape(H1.shape[0],H1.shape[1],1).float(),TIMESIZE)
#xs, hs = make_snapshots(reg.float(),H1.reshape(H1.shape[0],H1.shape[1],1).float(),TIMESIZE)
print(xs[0].shape)
border = int(SPLIT*len(xs))

train = xs[0:border]
test = xs[border:]
random.shuffle(train)
random.shuffle(test)
gtrain = transform_dgl(src4,dst4,train)
gtest = transform_dgl(src4,dst4,test)
#trainH = hs[0:border]
#testH = hs[border:]
ts = t[0:TIMESIZE]
#a=[1.0,0.01,0.5]
trainloss1 = []
testloss1 = []
trainloss2 = []
testloss2 = []

trainset = GraphDataLoader(gtrain,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
it = iter(trainset)
g = next(it)
#model.change_graph(g)
N_train=len(trainset)

testset = GraphDataLoader(gtest,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
it = iter(testset)
gt = next(it)

#model.change_graph(g)
N_test=len(testset) 

for epoch in tqdm(range(EPOCHS)):
    #c = list(zip(train,trainH))
    model.train()
    #trainsetH = DataLoader(trainH,batch_size=BATCHSIZE)
    print("Training")
    ploss=0
    model.change_graph(g)
    print("BATCHES: {}".format(len(trainset)))
    for sample in tqdm(trainset):
        opti.zero_grad()
        x = sample.ndata["x"].transpose(0,1)
        dx = sample.ndata["dx"].transpose(0,1)
        #print(sample.shape)
        #print("x: {}".format(x.shape))
        #print("dx: {}".format(dx.shape))
        #print("H: {}".format(H.shape))
        x0 = x[0,:,:]
        #h_hat = model.H(x0)
        x_hat = odeint(model,x0,ts,method="rk4")
        dx_l=[]
        for i in range(x.shape[0]):
            dx_l.append(model(0,x[i,:,:]).unsqueeze(0))
        dx_hat = torch.cat((dx_l),dim=0)
       # h_hat = model.H_rollout(x)
       # dx_hat = model.dx_rollout(x.requires_grad_())
        #print("x_hat {}".format(x_hat.shape))
        #print("h_hat {}".format(h_hat.shape))
        #print("dx_hat {}".format(dx_hat.shape))
        #print("Hs {}".format(Hs.shape))
        #loss = lossfn(x_hat[:,:,0:3],x[:,:,0:3])+ lossfn(x_hat[:,:,3:6],x[:,:,3:6])
        loss = lossfn(dx_hat[:,:,0:3],dx[:,:,0:3])+ lossfn(dx_hat[:,:,3:6],dx[:,:,3:6]) + a*(lossfn(x_hat[:,:,0:3],x[:,:,0:3])+ lossfn(x_hat[:,:,3:6],x[:,:,3:6]))
        #lossh = lossfn(h_hat,H)
        #lossv = lossfn(dx_hat[:,:,0:3],dx[:,:,0:3])+ lossfn(dx_hat[:,:,3:6],dx[:,:,3:6])
        #loss = a[0]*losst+a[1]*lossv+a[2]*lossh
        #p = model.parameters()
        ploss+=loss.item()
        
        loss.backward()
        opti.step()
    ploss/=N_train
    #print(ploss)
    print("TEST")
    
    tloss = 0
    model.eval()
    model.change_graph(gt)
    print("BATCHES: {}".format(len(testset)))
    for sample in tqdm(testset):
        xt = sample.ndata["x"].transpose(0,1)
        dxt = sample.ndata["dx"].transpose(0,1)
        #print(sample.shape)
        #dxt = sample[:,:,:,6:12]
        #Ht = H.transpose(0,1)
        #print("x: {}".format(x.shape))
        #print("dx: {}".format(dx.shape))
        #print("H: {}".format(H.shape))
        x0t = xt[0,:,:]
        #h_hat = model.H(x0)
        xt_hat = odeint(model,x0t,ts,method="rk4")
        dxt_l=[]
        for i in range(x.shape[0]):
            dxt_l.append(model(0,x[i,:,:]).unsqueeze(0))
        dxt_hat = torch.cat((dxt_l),dim=0)
        #ht_hat = model.H_rollout(xt)
        #dxt_hat = model.dx_rollout(xt.requires_grad_())
        #print("x_hat {}".format(x_hat.shape))
        #print("h_hat {}".format(h_hat.shape))
        #print("dx_hat {}".format(dx_hat.shape))
        #print("Hs {}".format(Hs.shape))
        loss = lossfn(dxt_hat[:,:,0:3],dxt[:,:,0:3])+ lossfn(dxt_hat[:,:,3:6],dxt[:,:,3:6])+a*(lossfn(xt_hat[:,:,0:3],xt[:,:,0:3])+ lossfn(xt_hat[:,:,3:6],xt[:,:,3:6]))
        #lossh = lossfn(h_hat,H)
        #lossh = lossfn(ht_hat,Ht)
        #lossv = lossfn(dxt_hat[:,:,0:3],dxt[:,:,0:3])+ lossfn(dxt_hat[:,:,3:6],dxt[:,:,3:6])
        #loss = a[0]*losst+a[1]*lossv+a[2]*lossh
        #p = model.parameters()
        tloss+=loss.item()

    tloss/=N_test
    #print(tloss)
    trainloss1.append(ploss)
    testloss1.append(tloss)
    dict={}
    for name,param in model.named_parameters():
            #dict[name]=torch.mean(param.data.flatten())
            dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()
    print("EPOCH: {} trainloss: {} testloss: {}".format(epoch+1,ploss,tloss))
    print(dict)
    
fig = plt.figure()
plt.title("4body")
plt.semilogy(np.linspace(0,EPOCHS,EPOCHS),trainloss1)
plt.semilogy(np.linspace(0,EPOCHS,EPOCHS),testloss1)
plt.legend(["train loss", "test_loss"])
plt.pause(0.01)
print("first phase ended")
model.change_graph(graph4)
pred4_1 = odeint(model,x1_eval[0,:,:],t,method="rk4")
model.change_graph(graph5)
pred5_1 = odeint(model,x2_eval[0,:,:],t,method="rk4")

fig,ax = plt.subplots(3,4)

ax[0,0].plot(x1_eval[:,0,0].detach().numpy(),x1_eval[:,0,3].detach().numpy())
ax[1,0].plot(x1_eval[:,0,1].detach().numpy(),x1_eval[:,0,4].detach().numpy())
ax[2,0].plot(x1_eval[:,0,2].detach().numpy(),x1_eval[:,0,5].detach().numpy())
ax[0,0].scatter(pred4_1[:,0,0].detach().numpy(),pred4_1[:,0,3].detach().numpy(),c="r",s=10)
ax[1,0].scatter(pred4_1[:,0,1].detach().numpy(),pred4_1[:,0,4].detach().numpy(),c="r",s=10)
ax[2,0].scatter(pred4_1[:,0,2].detach().numpy(),pred4_1[:,0,5].detach().numpy(),c="r",s=10)
ax[0,0].set_title("T4_body1_x")
ax[1,0].set_title("T4_body1_y")
ax[2,0].set_title("T4_body1_z")

ax[0,1].plot(x1_eval[:,1,0].detach().numpy(),x1_eval[:,1,3].detach().numpy())
ax[1,1].plot(x1_eval[:,1,1].detach().numpy(),x1_eval[:,1,4].detach().numpy())
ax[2,1].plot(x1_eval[:,1,2].detach().numpy(),x1_eval[:,1,5].detach().numpy())
ax[0,1].scatter(pred4_1[:,1,0].detach().numpy(),pred4_1[:,1,3].detach().numpy(),c="r",s=10)
ax[1,1].scatter(pred4_1[:,1,1].detach().numpy(),pred4_1[:,1,4].detach().numpy(),c="r",s=10)
ax[2,1].scatter(pred4_1[:,1,2].detach().numpy(),pred4_1[:,1,5].detach().numpy(),c="r",s=10)
ax[0,1].set_title("T4_body2_x")
ax[1,1].set_title("T4_body2_y")
ax[2,1].set_title("T4_body2_z")
ax[0,2].plot(x1_eval[:,2,0].detach().numpy(),x1_eval[:,2,3].detach().numpy())
ax[1,2].plot(x1_eval[:,2,1].detach().numpy(),x1_eval[:,2,4].detach().numpy())
ax[2,2].plot(x1_eval[:,2,2].detach().numpy(),x1_eval[:,2,5].detach().numpy())
ax[0,2].scatter(pred4_1[:,2,0].detach().numpy(),pred4_1[:,2,3].detach().numpy(),c="r",s=10)
ax[1,2].scatter(pred4_1[:,2,1].detach().numpy(),pred4_1[:,2,4].detach().numpy(),c="r",s=10)
ax[2,2].scatter(pred4_1[:,2,2].detach().numpy(),pred4_1[:,2,5].detach().numpy(),c="r",s=10)
ax[0,2].set_title("T4_body3_x")
ax[1,2].set_title("T4_body3_y")
ax[2,2].set_title("T4_body3_z")
ax[0,3].plot(x1_eval[:,3,0].detach().numpy(),x1_eval[:,3,3].detach().numpy())
ax[1,3].plot(x1_eval[:,3,1].detach().numpy(),x1_eval[:,3,4].detach().numpy())
ax[2,3].plot(x1_eval[:,3,2].detach().numpy(),x1_eval[:,3,5].detach().numpy())
ax[0,3].scatter(pred4_1[:,3,0].detach().numpy(),pred4_1[:,3,3].detach().numpy(),c="r",s=10)
ax[1,3].scatter(pred4_1[:,3,1].detach().numpy(),pred4_1[:,3,4].detach().numpy(),c="r",s=10)
ax[2,3].scatter(pred4_1[:,3,2].detach().numpy(),pred4_1[:,3,5].detach().numpy(),c="r",s=10)
ax[0,3].set_title("T4_body4_x")
ax[1,3].set_title("T4_body4_y")
ax[2,3].set_title("T4_body4_z")
plt.pause(0.001)

fig,ax = plt.subplots(3,5)

ax[0,0].plot(x2_eval[:,0,0].detach().numpy(),x2_eval[:,0,3].detach().numpy())
ax[1,0].plot(x2_eval[:,0,1].detach().numpy(),x2_eval[:,0,4].detach().numpy())
ax[2,0].plot(x2_eval[:,0,2].detach().numpy(),x2_eval[:,0,5].detach().numpy())
ax[0,0].scatter(pred5_1[:,0,0].detach().numpy(),pred5_1[:,0,3].detach().numpy(),c="r",s=10)
ax[1,0].scatter(pred5_1[:,0,1].detach().numpy(),pred5_1[:,0,4].detach().numpy(),c="r",s=10)
ax[2,0].scatter(pred5_1[:,0,2].detach().numpy(),pred5_1[:,0,5].detach().numpy(),c="r",s=10)
ax[0,0].set_title("T4_body1_x")
ax[1,0].set_title("T4_body1_y")
ax[2,0].set_title("T4_body1_z")

ax[0,1].plot(x2_eval[:,1,0].detach().numpy(),x2_eval[:,1,3].detach().numpy())
ax[1,1].plot(x2_eval[:,1,1].detach().numpy(),x2_eval[:,1,4].detach().numpy())
ax[2,1].plot(x2_eval[:,1,2].detach().numpy(),x2_eval[:,1,5].detach().numpy())
ax[0,1].scatter(pred5_1[:,1,0].detach().numpy(),pred5_1[:,1,3].detach().numpy(),c="r",s=10)
ax[1,1].scatter(pred5_1[:,1,1].detach().numpy(),pred5_1[:,1,4].detach().numpy(),c="r",s=10)
ax[2,1].scatter(pred5_1[:,1,2].detach().numpy(),pred5_1[:,1,5].detach().numpy(),c="r",s=10)
ax[0,1].set_title("T4_body2_x")
ax[1,1].set_title("T4_body2_y")
ax[2,1].set_title("T4_body2_z")
ax[0,2].plot(x2_eval[:,2,0].detach().numpy(),x2_eval[:,2,3].detach().numpy())
ax[1,2].plot(x2_eval[:,2,1].detach().numpy(),x2_eval[:,2,4].detach().numpy())
ax[2,2].plot(x2_eval[:,2,2].detach().numpy(),x2_eval[:,2,5].detach().numpy())
ax[0,2].scatter(pred5_1[:,2,0].detach().numpy(),pred5_1[:,2,3].detach().numpy(),c="r",s=10)
ax[1,2].scatter(pred5_1[:,2,1].detach().numpy(),pred5_1[:,2,4].detach().numpy(),c="r",s=10)
ax[2,2].scatter(pred5_1[:,2,2].detach().numpy(),pred5_1[:,2,5].detach().numpy(),c="r",s=10)
ax[0,2].set_title("T4_body3_x")
ax[1,2].set_title("T4_body3_y")
ax[2,2].set_title("T4_body3_z")
ax[0,3].plot(x2_eval[:,3,0].detach().numpy(),x2_eval[:,3,3].detach().numpy())
ax[1,3].plot(x2_eval[:,3,1].detach().numpy(),x2_eval[:,3,4].detach().numpy())
ax[2,3].plot(x2_eval[:,3,2].detach().numpy(),x2_eval[:,3,5].detach().numpy())
ax[0,3].scatter(pred5_1[:,3,0].detach().numpy(),pred5_1[:,3,3].detach().numpy(),c="r",s=10)
ax[1,3].scatter(pred5_1[:,3,1].detach().numpy(),pred5_1[:,3,4].detach().numpy(),c="r",s=10)
ax[2,3].scatter(pred5_1[:,3,2].detach().numpy(),pred5_1[:,3,5].detach().numpy(),c="r",s=10)
ax[0,3].set_title("T4_body4_x")
ax[1,3].set_title("T4_body4_y")
ax[2,3].set_title("T4_body4_z")

ax[0,4].plot(x2_eval[:,4,0].detach().numpy(),x2_eval[:,4,3].detach().numpy())
ax[1,4].plot(x2_eval[:,4,1].detach().numpy(),x2_eval[:,4,4].detach().numpy())
ax[2,4].plot(x2_eval[:,4,2].detach().numpy(),x2_eval[:,4,5].detach().numpy())
ax[0,4].scatter(pred5_1[:,4,0].detach().numpy(),pred5_1[:,4,3].detach().numpy(),c="r",s=10)
ax[1,4].scatter(pred5_1[:,4,1].detach().numpy(),pred5_1[:,4,4].detach().numpy(),c="r",s=10)
ax[2,4].scatter(pred5_1[:,4,2].detach().numpy(),pred5_1[:,4,5].detach().numpy(),c="r",s=10)
ax[0,4].set_title("T4_body5_x")
ax[1,4].set_title("T4_body5_y")
ax[2,4].set_title("T4_body5_z")
plt.pause(0.001)




print("SECOND PHASE")


    
    
xs, hs = make_snapshots(x2.float(),H2.reshape(H2.shape[0],H2.shape[1],1).float(),TIMESIZE)
print(xs[0].shape)
border = int(SPLIT*len(xs))

train = xs[0:border]
test = xs[border:]
random.shuffle(train)
random.shuffle(test)
gtrain = transform_dgl(src5,dst5,train)
gtest = transform_dgl(src5,dst5,test)
#trainH = hs[0:border]
#testH = hs[border:]
ts = t[0:TIMESIZE]
#a=[1.0,0.01,0.5]


trainset = GraphDataLoader(gtrain,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
it = iter(trainset)
g=next(it)
#model.change_graph(g)
N_train=len(trainset)

testset = GraphDataLoader(gtest,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
it = iter(testset)
gt = next(it)

N_test=len(testset)


for epoch in tqdm(range(EPOCHS)):
    #c = list(zip(train,trainH))
    model.train()
    #trainsetH = DataLoader(trainH,batch_size=BATCHSIZE)
    print("Training")
    ploss=0
    model.change_graph(g)
    print("BATCHES: {}".format(len(trainset)))
    for sample in tqdm(trainset):
        opti.zero_grad()
        x = sample.ndata["x"].transpose(0,1)
        dx = sample.ndata["dx"].transpose(0,1)
        x0 = x[0,:,:]
        #h_hat = model.H(x0)
        x_hat = odeint(model,x0,ts,method="rk4")
        #print("x: {}".format(x.shape))    x1_eval = data1[:,-1,:,0:6]
        dx_l=[]
        for i in range(x.shape[0]):
            dx_l.append(model(0,x[i,:,:]).unsqueeze(0))
        dx_hat = torch.cat((dx_l),dim=0)
       # h_hat = model.H_rollout(x)
       # dx_hat = model.dx_rollout(x.requires_grad_())
        #print("x_hat {}".format(x_hat.shape))
        #print("h_hat {}".format(h_hat.shape))
        #print("dx_hat {}".format(dx_hat.shape))
        #print("Hs {}".format(Hs.shape))
        #loss = lossfn(x_hat[:,:,0:3],x[:,:,0:3])+ lossfn(x_hat[:,:,3:6],x[:,:,3:6])
        loss = lossfn(dx_hat[:,:,0:3],dx[:,:,0:3])+ lossfn(dx_hat[:,:,3:6],dx[:,:,3:6])+a*(lossfn(x_hat[:,:,0:3],x[:,:,0:3])+ lossfn(x_hat[:,:,3:6],x[:,:,3:6]))
        #lossh = lossfn(h_hat,H)
        #lossv = lossfn(dx_hat[:,:,0:3],dx[:,:,0:3])+ lossfn(dx_hat[:,:,3:6],dx[:,:,3:6])
        #loss = a[0]*losst+a[1]*lossv+a[2]*lossh
        #p = model.parameters()
        ploss+=loss.item()
        
        loss.backward()
        opti.step()
    ploss/=N_train
    #print(ploss)
    print("TEST")
    
    tloss = 0
    model.change_graph(g)
    model.eval()
    print("BATCHES: {}".format(len(testset)))
    for sample in tqdm(testset):
        xt = sample.ndata["x"].transpose(0,1)
        dxt = sample.ndata["dx"].transpose(0,1)
        #print(sample.shape)
        #dxt = sample[:,:,:,6:12]
        #Ht = H.transpose(0,1)
        #print("x: {}".format(x.shape))
        #print("dx: {}".format(dx.shape))
        #print("H: {}".format(H.shape))
        x0t = xt[0,:,:]
        #h_hat = model.H(x0)
        xt_hat = odeint(model,x0t,ts,method="rk4")
        dxt_l=[]
        for i in range(x.shape[0]):
            dxt_l.append(model(0,x[i,:,:]).unsqueeze(0))
        dxt_hat = torch.cat((dxt_l),dim=0)
        #ht_hat = model.H_rollout(xt)
        #dxt_hat = model.dx_rollout(xt.requires_grad_())
        #print("x_hat {}".format(x_hat.shape))
        #print("h_hat {}".format(h_hat.shape))
        #print("dx_hat {}".format(dx_hat.shape))
        #print("Hs {}".format(Hs.shape))
        loss = lossfn(dxt_hat[:,:,0:3],dxt[:,:,0:3])+ lossfn(dxt_hat[:,:,3:6],dxt[:,:,3:6])+a*(lossfn(xt_hat[:,:,0:3],xt[:,:,0:3])+ lossfn(xt_hat[:,:,3:6],xt[:,:,3:6]))
        #lossh = lossfn(ht_hat,Ht)
        #lossv = lossfn(dxt_hat[:,:,0:3],dxt[:,:,0:3])+ lossfn(dxt_hat[:,:,3:6],dxt[:,:,3:6])
        #loss = a[0]*losst+a[1]*lossv+a[2]*lossh
        #p = model.parameters()
        tloss+=loss.item()

    tloss/=N_test
    #print(tloss)
    trainloss2.append(ploss)
    testloss2.append(tloss)
    

    print("EPOCH: {} trainloss: {} testloss: {}".format(epoch+1,ploss,tloss))
    dict={}
    for name,param in model.named_parameters():
            #dict[name]=torch.mean(param.data.flatten())
            dict[name+"_grad"]=torch.mean(param.grad.flatten()).item()
    
    print(dict)

fig = plt.figure()
plt.title("5body")
plt.semilogy(np.linspace(0,EPOCHS,EPOCHS),trainloss2)
plt.semilogy(np.linspace(0,EPOCHS,EPOCHS),testloss2)
plt.legend(["train loss", "test_loss"])
plt.pause(0.01)


model.change_graph(graph4)
pred4_1 = odeint(model,x1_eval[0,:,:],t,method="rk4")
model.change_graph(graph5)
pred5_1 = odeint(model,x2_eval[0,:,:],t,method="rk4")

fig,ax = plt.subplots(3,4)

ax[0,0].plot(x1_eval[:,0,0].detach().numpy(),x1_eval[:,0,3].detach().numpy())
ax[1,0].plot(x1_eval[:,0,1].detach().numpy(),x1_eval[:,0,4].detach().numpy())
ax[2,0].plot(x1_eval[:,0,2].detach().numpy(),x1_eval[:,0,5].detach().numpy())
ax[0,0].scatter(pred4_1[:,0,0].detach().numpy(),pred4_1[:,0,3].detach().numpy(),c="r",s=10)
ax[1,0].scatter(pred4_1[:,0,1].detach().numpy(),pred4_1[:,0,4].detach().numpy(),c="r",s=10)
ax[2,0].scatter(pred4_1[:,0,2].detach().numpy(),pred4_1[:,0,5].detach().numpy(),c="r",s=10)
ax[0,0].set_title("T5 body1_x")
ax[1,0].set_title("T5 body1_y")
ax[2,0].set_title("T5 body1_z")

ax[0,1].plot(x1_eval[:,1,0].detach().numpy(),x1_eval[:,1,3].detach().numpy())
ax[1,1].plot(x1_eval[:,1,1].detach().numpy(),x1_eval[:,1,4].detach().numpy())
ax[2,1].plot(x1_eval[:,1,2].detach().numpy(),x1_eval[:,1,5].detach().numpy())
ax[0,1].scatter(pred4_1[:,1,0].detach().numpy(),pred4_1[:,1,3].detach().numpy(),c="r",s=10)
ax[1,1].scatter(pred4_1[:,1,1].detach().numpy(),pred4_1[:,1,4].detach().numpy(),c="r",s=10)
ax[2,1].scatter(pred4_1[:,1,2].detach().numpy(),pred4_1[:,1,5].detach().numpy(),c="r",s=10)
ax[0,1].set_title("T5 body2_x")
ax[1,1].set_title("T5 body2_y")
ax[2,1].set_title("T5 body2_z")
ax[0,2].plot(x1_eval[:,2,0].detach().numpy(),x1_eval[:,2,3].detach().numpy())
ax[1,2].plot(x1_eval[:,2,1].detach().numpy(),x1_eval[:,2,4].detach().numpy())
ax[2,2].plot(x1_eval[:,2,2].detach().numpy(),x1_eval[:,2,5].detach().numpy())
ax[0,2].scatter(pred4_1[:,2,0].detach().numpy(),pred4_1[:,2,3].detach().numpy(),c="r",s=10)
ax[1,2].scatter(pred4_1[:,2,1].detach().numpy(),pred4_1[:,2,4].detach().numpy(),c="r",s=10)
ax[2,2].scatter(pred4_1[:,2,2].detach().numpy(),pred4_1[:,2,5].detach().numpy(),c="r",s=10)
ax[0,2].set_title("T5 body3_x")
ax[1,2].set_title("T5 body3_y")
ax[2,2].set_title("T5 body3_z")
ax[0,3].plot(x1_eval[:,3,0].detach().numpy(),x1_eval[:,3,3].detach().numpy())
ax[1,3].plot(x1_eval[:,3,1].detach().numpy(),x1_eval[:,3,4].detach().numpy())
ax[2,3].plot(x1_eval[:,3,2].detach().numpy(),x1_eval[:,3,5].detach().numpy())
ax[0,3].scatter(pred4_1[:,3,0].detach().numpy(),pred4_1[:,3,3].detach().numpy(),c="r",s=10)
ax[1,3].scatter(pred4_1[:,3,1].detach().numpy(),pred4_1[:,3,4].detach().numpy(),c="r",s=10)
ax[2,3].scatter(pred4_1[:,3,2].detach().numpy(),pred4_1[:,3,5].detach().numpy(),c="r",s=10)
ax[0,3].set_title("T5 body4_x")
ax[1,3].set_title("T5 body4_y")
ax[2,3].set_title("T5 body4_z")
plt.pause(0.001)

fig,ax = plt.subplots(3,5)

ax[0,0].plot(x2_eval[:,0,0].detach().numpy(),x2_eval[:,0,3].detach().numpy())
ax[1,0].plot(x2_eval[:,0,1].detach().numpy(),x2_eval[:,0,4].detach().numpy())
ax[2,0].plot(x2_eval[:,0,2].detach().numpy(),x2_eval[:,0,5].detach().numpy())
ax[0,0].scatter(pred5_1[:,0,0].detach().numpy(),pred5_1[:,0,3].detach().numpy(),c="r",s=10)
ax[1,0].scatter(pred5_1[:,0,1].detach().numpy(),pred5_1[:,0,4].detach().numpy(),c="r",s=10)
ax[2,0].scatter(pred5_1[:,0,2].detach().numpy(),pred5_1[:,0,5].detach().numpy(),c="r",s=10)
ax[0,0].set_title("T5 body1_x")
ax[1,0].set_title("T5 body1_y")
ax[2,0].set_title("T5 body1_z")

ax[0,1].plot(x2_eval[:,1,0].detach().numpy(),x2_eval[:,1,3].detach().numpy())
ax[1,1].plot(x2_eval[:,1,1].detach().numpy(),x2_eval[:,1,4].detach().numpy())
ax[2,1].plot(x2_eval[:,1,2].detach().numpy(),x2_eval[:,1,5].detach().numpy())
ax[0,1].scatter(pred5_1[:,1,0].detach().numpy(),pred5_1[:,1,3].detach().numpy(),c="r",s=10)
ax[1,1].scatter(pred5_1[:,1,1].detach().numpy(),pred5_1[:,1,4].detach().numpy(),c="r",s=10)
ax[2,1].scatter(pred5_1[:,1,2].detach().numpy(),pred5_1[:,1,5].detach().numpy(),c="r",s=10)
ax[0,1].set_title("T5 body2_x")
ax[1,1].set_title("T5 body2_y")
ax[2,1].set_title("T5 body2_z")
ax[0,2].plot(x2_eval[:,2,0].detach().numpy(),x2_eval[:,2,3].detach().numpy())
ax[1,2].plot(x2_eval[:,2,1].detach().numpy(),x2_eval[:,2,4].detach().numpy())
ax[2,2].plot(x2_eval[:,2,2].detach().numpy(),x2_eval[:,2,5].detach().numpy())
ax[0,2].scatter(pred5_1[:,2,0].detach().numpy(),pred5_1[:,2,3].detach().numpy(),c="r",s=10)
ax[1,2].scatter(pred5_1[:,2,1].detach().numpy(),pred5_1[:,2,4].detach().numpy(),c="r",s=10)
ax[2,2].scatter(pred5_1[:,2,2].detach().numpy(),pred5_1[:,2,5].detach().numpy(),c="r",s=10)
ax[0,2].set_title("T5 body3_x")
ax[1,2].set_title("T5 body3_y")
ax[2,2].set_title("T5 body3_z")
ax[0,3].plot(x2_eval[:,3,0].detach().numpy(),x2_eval[:,3,3].detach().numpy())
ax[1,3].plot(x2_eval[:,3,1].detach().numpy(),x2_eval[:,3,4].detach().numpy())
ax[2,3].plot(x2_eval[:,3,2].detach().numpy(),x2_eval[:,3,5].detach().numpy())
ax[0,3].scatter(pred5_1[:,3,0].detach().numpy(),pred5_1[:,3,3].detach().numpy(),c="r",s=10)
ax[1,3].scatter(pred5_1[:,3,1].detach().numpy(),pred5_1[:,3,4].detach().numpy(),c="r",s=10)
ax[2,3].scatter(pred5_1[:,3,2].detach().numpy(),pred5_1[:,3,5].detach().numpy(),c="r",s=10)
ax[0,3].set_title("body4_x")
ax[1,3].set_title("body4_y")
ax[2,3].set_title("body4_z")

ax[0,4].plot(x2_eval[:,4,0].detach().numpy(),x2_eval[:,4,3].detach().numpy())
ax[1,4].plot(x2_eval[:,4,1].detach().numpy(),x2_eval[:,4,4].detach().numpy())
ax[2,4].plot(x2_eval[:,4,2].detach().numpy(),x2_eval[:,4,5].detach().numpy())
ax[0,4].scatter(pred5_1[:,4,0].detach().numpy(),pred5_1[:,4,3].detach().numpy(),c="r",s=10)
ax[1,4].scatter(pred5_1[:,4,1].detach().numpy(),pred5_1[:,4,4].detach().numpy(),c="r",s=10)
ax[2,4].scatter(pred5_1[:,4,2].detach().numpy(),pred5_1[:,4,5].detach().numpy(),c="r",s=10)
ax[0,4].set_title("T5 body5_x")
ax[1,4].set_title("T5 body5_y")
ax[2,4].set_title("T5 body5_z")
plt.pause(0.001)

plt.show()