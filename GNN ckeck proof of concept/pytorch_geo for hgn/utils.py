from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
def snap_maker(data,t_batch):
    N = data.shape[1]
    T = data.shape[0]
    list=[]
    for i in range(T-t_batch):
        for j in range(N):
            list.append(data[i:i+t_batch,j,:,:])
            
    return list

def transform_threbody(dataset):
    #print(dataset.shape)
    T = dataset.shape[0]
    B = dataset.shape[1]
    graphdata = torch.Tensor(T,B,3,4)
    for i in range(T):
        for j in range(B):
            graphdata[i,j,0,0:2] = dataset[i,j,0,0:2]
            graphdata[i,j,0,2:4] = dataset[i,j,0,6:8]
            graphdata[i,j,1,0:2] = dataset[i,j,0,2:4]
            graphdata[i,j,1,2:4] = dataset[i,j,0,8:10]
            graphdata[i,j,2,0:2] = dataset[i,j,0,4:6]
            graphdata[i,j,2,2:4] = dataset[i,j,0,10:12]
    return graphdata

def phasespace_show_threebody(data):
    fig, ax = plt.subplots(2,3)
    for i in range(3):
        for j in range(2):
            ax[j,i].set_title("qp plot body {}".format(i+1))
            ax[j,i].scatter(data[:,i,j],data[:,i,j+2])
            ax[j,i].set_xlabel("q")
            ax[j,i].set_ylabel("p")
            if j == 0:
                coor = "x"
            else:
                coor = "y"
            ax[j,i].legend(["qp trajectory {} coordinate".format(coor)])
    plt.show()
    
def phasespace_show_threebody_pred(data,pred):
    fig, ax = plt.subplots(2,3)
    for i in range(3):
        for j in range(2):
            ax[j,i].set_title("qp plot body {}".format(i+1))
            ax[j,i].scatter(data[:,i,j],data[:,i,j+2])
            ax[j,i].scatter(pred[:,i,j].detach().numpy(),pred[:,i,j+2].detach().numpy(),c="r")
            ax[j,i].set_xlabel("q")
            ax[j,i].set_ylabel("p")
            if j == 0:
                coor = "x"
            else:
                coor = "y"
            ax[j,i].legend(["qp trajectory {} coordinate".format(coor),"predicition"])
    plt.show()
    
def makePyG3dofDataset(src,dst,snaps,dsnaps):
    print("Welcome to PyG Dataset creator")
    data_list = []
    for snap,dsnap in zip(snaps,dsnaps):
        temp = Data(x = snap.transpose(0,1),dx = dsnap.transpose(0,1), edge_index=torch.tensor([src,dst],dtype=torch.long))
        data_list.append(temp)
    return data_list