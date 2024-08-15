import torch
import dgl
import matplotlib.pyplot as plt

def transform2dgl(src,dst,gsnaps,dgsnaps=None):
    l = []
    if dgsnaps is None:
        for snap in gsnaps:
            g = dgl.graph((src,dst))
            g.ndata["x"] = snap.transpose(0,1)
        #print(g.ndata["x"].shape)
            l.append(g)
    else:
        for snap, dsnap in zip(gsnaps,dgsnaps):
            #print("tr x{}".format(snap.shape))
            #print("tr dx{}".format(dsnap.shape))
            g = dgl.graph((src,dst))
            g.ndata["x"] = snap.transpose(0,1)
            g.ndata["dx"] = dsnap.transpose(0,1)
        #print(g.ndata["x"].shape)
            l.append(g)
    return l

def snap_maker(data,t_batch):
    N = data.shape[1]
    T = data.shape[0]
    list=[]
    for i in range(T-t_batch):
        for j in range(N):
            list.append(data[i:i+t_batch,j,:,:])
            
    return list


def acc(data,pred):
    a=torch.abs(data-pred)
    d = torch.abs(data)
    return torch.mean(a.flatten()/d.flatten())

def acc_pos(data,pred,eps=1e-4):
    count = 0
    for i in  range(data.shape[0]):
        d = data[i,0,:]
        p = pred[i,0,:]
        if torch.linalg.vector_norm(p-d,dim=-1) < eps:
            print()
            print("correct point: {}".format(i))
            count+=1
        #else: 
        #    print(torch.linalg.vector_norm(p-d,dim=-1).item())
    return count/data.shape[0]

def create_pend1dof_graph_snapshots(snaps,src,dst):
    graphs=[]
    for snap in snaps:
        #print(snap.shape) 
        g = dgl.graph((src,dst))
        g.ndata["x"] = snap[:,:,0:2].transpose(0,1) 
        g.ndata["dx"] = snap[:,:,2:4].transpose(0,1)
        g.ndata["H"] =  snap[:,:,-1].transpose(0,1)
        graphs.append(g)
    return graphs

def create_pend2dof_graph_snapshots(snaps,src,dst):
    graphs=[]
    nodes = 2
    for snap in snaps:
        #print(snap.shape)
        tempx=torch.zeros(2,snap.shape[0],2) 
        tempx[0,:,0] = snap[:,0,0]
        tempx[1,:,0] = snap[:,0,1]
        tempx[0,:,1] = snap[:,0,2]
        tempx[1,:,1] = snap[:,0,3]
        tempdx=torch.zeros(2,snap.shape[0],2) 
        tempdx[0,:,0] = snap[:,0,4]
        tempdx[1,:,0] = snap[:,0,5]
        tempdx[0,:,1] = snap[:,0,6]
        tempdx[1,:,1] = snap[:,0,7]
        tempH=torch.zeros(2,snap.shape[0],1)
        tempH[0,:,0] = snap[:,0,8]
        tempH[1,:,0] = snap[:,0,8]
        
        g = dgl.graph((src,dst))
        g.ndata["x"] = tempx
        g.ndata["dx"] = tempdx
        g.ndata["H"] =  tempH
        graphs.append(g)
    return graphs
       
def create_pend3dof_graph_snapshots(snaps,src,dst):
    graphs=[]
    nodes = 3
    for snap in snaps:
        #print(snap.shape)
        tempx=torch.zeros(3,snap.shape[0],2) 
        tempx[0,:,0] = snap[:,0,0]
        tempx[1,:,0] = snap[:,0,1]
        tempx[2,:,0] = snap[:,0,2]
        tempx[0,:,1] = snap[:,0,3]
        tempx[1,:,1] = snap[:,0,4]
        tempx[2,:,1] = snap[:,0,5]
        tempdx=torch.zeros(3,snap.shape[0],2) 
        tempdx[0,:,0] = snap[:,0,6]
        tempdx[1,:,0] = snap[:,0,7]
        tempdx[2,:,0] = snap[:,0,8]
        tempdx[0,:,1] = snap[:,0,9]
        tempdx[1,:,1] = snap[:,0,10]
        tempdx[2,:,1] = snap[:,0,11]
        tempH=torch.zeros(3,snap.shape[0],1)
        tempH[0,:,0] = snap[:,0,12]
        tempH[1,:,0] = snap[:,0,12]
        tempH[2,:,0] = snap[:,0,12]
        
        g = dgl.graph((src,dst))
        g.ndata["x"] = tempx
        g.ndata["dx"] = tempdx
        g.ndata["H"] =  tempH
        graphs.append(g)
    return graphs 

def create_pend4dof_graph_snapshots(snaps,src,dst):
    graphs=[]
    nodes = 4
    for snap in snaps:
        #print(snap.shape)
        tempx=torch.zeros(4,snap.shape[0],2) 
        tempx[0,:,0] = snap[:,0,0]
        tempx[1,:,0] = snap[:,0,1]
        tempx[2,:,0] = snap[:,0,2]
        tempx[3,:,0] = snap[:,0,3]
        tempx[0,:,1] = snap[:,0,4]
        tempx[1,:,1] = snap[:,0,5]
        tempx[2,:,1] = snap[:,0,6]
        tempx[3,:,1] = snap[:,0,7]
        tempdx=torch.zeros(4,snap.shape[0],2) 
        tempdx[0,:,0] = snap[:,0,8]
        tempdx[1,:,0] = snap[:,0,9]
        tempdx[2,:,0] = snap[:,0,10]
        tempdx[3,:,0] = snap[:,0,11]
        tempdx[0,:,1] = snap[:,0,12]
        tempdx[1,:,1] = snap[:,0,13]
        tempdx[2,:,1] = snap[:,0,14]
        tempdx[3,:,1] = snap[:,0,15]
        tempH=torch.zeros(4,snap.shape[0],1)
        tempH[0,:,0] = snap[:,0,16]
        tempH[1,:,0] = snap[:,0,16]
        tempH[2,:,0] = snap[:,0,16]
        tempH[3,:,0] = snap[:,0,16]
        
        g = dgl.graph((src,dst))
        g.ndata["x"] = tempx
        g.ndata["dx"] = tempdx
        g.ndata["H"] =  tempH
        graphs.append(g)
    return graphs
    
        
def visualize_loss(title, loss_container):
    
    t = torch.linspace(0,loss_container.shape[1],loss_container.shape[1])
    fig = plt.figure()
    plt.title("{} logy".format(title))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.semilogy(t,loss_container[0,:].detach().numpy(),c="r")
    plt.semilogy(t,loss_container[1,:].detach().numpy(),c="b")
    plt.legend(["train","test"])
    fig.savefig("loss_"+title+".png")     