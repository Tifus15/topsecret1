from dof3_pendelum_torch import *
from device_util import*
#from torch_geometric.data import Data
import dgl

def makeDGL3dofDataset(edge_list,dataset, t_batchsize):
    print("Welcome to DGL Dataset creator")
    nsnapshots = dataset.shape[0]
    nsamples = dataset.shape[1]
    nfeats = dataset.shape[3]
    graphs=[]
    print("with snaphots size: {}\nwe will get {} batches from the dataset".format(t_batchsize, nsamples*int(nsnapshots-t_batchsize)))
    for i in range(int(nsnapshots-t_batchsize)):
        for j in range(int(nsamples)):
            time_range_b = i
            time_range_e = i + t_batchsize
            sample = j
            y = transform_Data_3dof(dataset[time_range_b:time_range_e,sample,0,:])
            x = y[:,:,0]
            # print(x.shape)
            src = edge_list[0]
            dst = edge_list[1]
            g = dgl.graph((src,dst))
            g.ndata["x"] = x
            g.ndata["y"] = y
            graphs.append(g)

    return graphs

def transform_Data_3dof(y):
    out = torch.Tensor(3,2,y.shape[0])
    node1 = y[:,[0,3]].transpose(0,1)
    node2 = y[:,[1,4]].transpose(0,1)
    node3 = y[:,[2,5]].transpose(0,1)
    out[0,:,:] = node1
    out[1,:,:] = node2
    out[2,:,:] = node3
    return out
    


def make3dofBaseDataset(settings):
    dt = settings["dt"]
    T = settings["T"]
    t_samples = int(T/dt) + 1
    samples = settings["samples"]

    t = torch.linspace(0,T,t_samples)
    pi_b = settings["pi_range"][0]
    pi_e = settings["pi_range"][1]
    datamaker = pendelum3dof(1,1,1,1,1,1,9.81)
    #randomised
    inits = torch.cat(([pi_b + torch.rand(samples,1,3)*(pi_e-pi_b),torch.zeros(samples,1,3)]),dim=-1)
    dataset = torch.zeros(t_samples,samples,1,6).to(DEVICE)
    for i in tqdm(range(samples)):
        flag=False
        temp = inits[i,:,:]
        while(flag==False):
            print(" making trajectory ")
            temp_traj = dof3_trajectory(datamaker.to(DEVICE),t.to(DEVICE),temp.to(DEVICE))
            print(" trajectory check ")
            flag = perfect_sample(datamaker,temp_traj)
            if flag:
                print(" trajectory accepted ")
                dataset[:,i,:,:]=temp_traj
            else:
                print(" trajectory recycled ")
                temp = torch.cat(([pi_b + torch.rand(1,3)*(pi_e-pi_b),torch.zeros(1,3)]),dim=-1)
    return dataset.cpu().detach()

"""
def makePyG3dofDataset(edge_list,dataset, t_batchsize):
    print("Welcome to PyG Dataset creator")
    nsnapshots = dataset.shape[0]
    nsamples = dataset.shape[1]
    nfeats = dataset.shape[3] #there is time feature
    print("with snaphots size: {}\nwe will get {} batches from the dataset".format(t_batchsize, nsamples*int(nsnapshots-t_batchsize)))
    data_list = []
    for i in range(int(nsnapshots-t_batchsize)):
        for j in range(int(nsamples)):
            time_range_b = i
            time_range_e = i + t_batchsize
            sample = j
            y = transform_Data_3dof(dataset[time_range_b:time_range_e,sample,0,:])
            #print(y.shape)
            x = y[:,:,0]
            #print("{},{}".format(i,j))
            data = Data(x,edge_index=torch.tensor(edge_list,dtype=torch.int64),y=y)
            data_list.append(data)
    print("list is so big: {}".format(len(data_list)))
    return data_list 
"""