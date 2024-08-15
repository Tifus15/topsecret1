import yaml
from device_util import ROOT_PATH
from oscilator import *
from twobody import *
from threebody import *
from samples import SAMPLES

def visualize_loss(title, loss_container):
    t = torch.linspace(0,loss_container.shape[1],loss_container.shape[1])
    fig = plt.figure()
    plt.title("{}".format(title))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.semilogy(t,loss_container[0,:],c="r")
    plt.semilogy(t,loss_container[1,:],c="b")
    plt.legend(["train","test"])
    plt.show()  

def visualize_hamiltonian(H_ground, H,t):
    fig = plt.figure()
    plt.title("{} plot".format("hamiltonian"))
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.scatter(t,H.detach().numpy(),c="r")
    plt.scatter(t,H_ground.detach().numpy(),c="b")
    plt.legend(["prediction","ground truth"])
    plt.show()  

def viz_traj(ground,model_eval,type):
    ground = ground.squeeze()
    model_eval = model_eval.squeeze()
    if type == "osci":
        fig = plt.figure()
        plt.scatter(model_eval[:,0].detach().numpy(),model_eval[:,1].detach().numpy())
        plt.scatter(ground[:,0].detach().numpy(),ground[:,1].detach().numpy())
        plt.legend(["prediction","ground_truth"])
    if type == "twobody":
        fig, ax = plt.subplots(2,2)
        ax[0,0].scatter(model_eval[:,0].detach().numpy(),model_eval[:,1].detach().numpy())
        ax[0,0].scatter(ground[:,0].detach().numpy(),ground[:,1].detach().numpy())
        ax[0,0].legend(["prediction","ground_truth"])
        ax[0,1].scatter(model_eval[:,2].detach().numpy(),model_eval[:,3].detach().numpy())
        ax[0,1].scatter(ground[:,2].detach().numpy(),ground[:,3].detach().numpy())
        ax[0,1].legend(["prediction","ground_truth"])
        ax[1,0].scatter(model_eval[:,4].detach().numpy(),model_eval[:,5].detach().numpy())
        ax[1,0].scatter(ground[:,4].detach().numpy(),ground[:,5].detach().numpy())
        ax[1,0].legend(["prediction","ground_truth"])
        ax[1,1].scatter(model_eval[:,6].detach().numpy(),model_eval[:,7].detach().numpy())
        ax[1,1].scatter(ground[:,6].detach().numpy(),ground[:,7].detach().numpy())
        ax[1,1].legend(["prediction","ground_truth"])
    if type == "threebody":
        fig, ax = plt.subplots(2,3)
        ax[0,0].scatter(model_eval[:,0].detach().numpy(),model_eval[:,1].detach().numpy())
        ax[0,0].scatter(ground[:,0].detach().numpy(),ground[:,1].detach().numpy())
        ax[0,0].legend(["prediction","ground_truth"])
        ax[0,1].scatter(model_eval[:,2].detach().numpy(),model_eval[:,3].detach().numpy())
        ax[0,1].scatter(ground[:,2].detach().numpy(),ground[:,3].detach().numpy())
        ax[0,1].legend(["prediction","ground_truth"])
        ax[0,2].scatter(model_eval[:,4].detach().numpy(),model_eval[:,5].detach().numpy())
        ax[0,2].scatter(ground[:,4].detach().numpy(),ground[:,5].detach().numpy())
        ax[0,2].legend(["prediction","ground_truth"])
        ax[1,0].scatter(model_eval[:,6].detach().numpy(),model_eval[:,7].detach().numpy())
        ax[1,0].scatter(ground[:,6].detach().numpy(),ground[:,7].detach().numpy())
        ax[1,0].legend(["prediction","ground_truth"])
        ax[1,1].scatter(model_eval[:,8].detach().numpy(),model_eval[:,9].detach().numpy())
        ax[1,1].scatter(ground[:,8].detach().numpy(),ground[:,9].detach().numpy())
        ax[1,1].legend(["prediction","ground_truth"])
        ax[1,2].scatter(model_eval[:,10].detach().numpy(),model_eval[:,11].detach().numpy())
        ax[1,2].scatter(ground[:,10].detach().numpy(),ground[:,11].detach().numpy())
        ax[1,2].legend(["prediction","ground_truth"])
    plt.show()

def make_snapshots(data, TIME_SIZE):
    time_size =TIME_SIZE
    points = data.shape[0]
    print("{} snapshots".format((points-time_size)*data.shape[1]))
    list = []
    
    
    for i in range(data.shape[1]):
        for j in range(points-time_size):
            list.append(data[j:j+time_size,i,:,:])
                
    return list



def open_yamls(name):
    with open(ROOT_PATH+"/yamls/"+name) as stream:
        try:
            data_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data_dict

def make_dataset(dict):
    type = dict["type"]
    if type == "oscilator":
        print("OSCILATOR DATASET")
        points = dict["points"]
        samples = dict["samples"]
        data_maker = oscilator(dict["m"],dict["k"])
        H_span = dict["H_span"]
        #print(H_span)
        data ,ddata, t, H = data_maker.make_dataset(points,samples,(H_span[0],H_span[1]))
        func = data_maker.hamiltonian
        return data,ddata, t, H, func
    elif type == "twobody":
        print("TWOBODY DATASET")
        points = dict["points"]
        samples = dict["samples"]
        data_maker = twobody(dict["m1"],dict["m2"],dict["G"])
        data,ddata, t, H = data_maker.make_dataset(points,samples,T=dict["T"],r1_range=(dict["r1_range"][0],dict["r1_range"][1]))
        func =data_maker.hamiltonian
        return data,ddata, t, H, func
    elif type == "threebody":
        print("THREEBODY DATASET")
        points = dict["points"]
        samples = dict["samples"]
        sample = dict["sample"]
        data_maker = threebody(SAMPLES)
        data,ddata, t, H = data_maker.dataset_onekind(sample,samples,points,phi_span=(dict["phi_span"][0],dict["phi_span"][1]))
        func = data_maker.hamiltonian
        return data ,ddata, t, H, func
