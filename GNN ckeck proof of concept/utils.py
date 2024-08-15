import torch
import dgl

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
        
        