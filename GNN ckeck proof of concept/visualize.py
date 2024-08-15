import torch
import matplotlib.pyplot as plt


def phasespace_show(data):
    plt.figure()
    plt.title("qp plot")
    plt.scatter(data[:,0,0],data[:,0,1])
    plt.xlabel("q")
    plt.ylabel("p")
    plt.legend(["qp trajectory"])
    plt.show()
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
    plt.show(block=False)
    
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
    
def model_phasespace_show(data,pred):
    plt.figure()
    plt.title("qp plot")
    plt.scatter(data[:,0,0].detach().numpy(),data[:,0,1].detach().numpy(),c="b")
    plt.plot(pred[:,0,0].detach().numpy(),pred[:,0,1].detach().numpy(),c="r")
    plt.xlabel("q")
    plt.ylabel("p")
    plt.legend(["ground_truth","predicition"])
    plt.show()
def ham_show(t,H):
    plt.figure()
    plt.title("qp plot")
    plt.plot(t,H)
    plt.ylim([H[0]-H[0]*0.1,H[0]+H[0]*0.1])
    plt.xlabel("q")
    plt.ylabel("p")
    plt.show()