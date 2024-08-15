import torch
from dof3_pendelum_torch import *
from dof2_pendelum_torch import *
from dof1_pendelum_torch import *
from dof4_pendelum_torch import *
from device_util import ROOT_PATH, DEVICE


def create_1dof_pendelum_dataset(samples,T,steps,data = [1,1,9.81],range_of_angles=[-torch.pi,torch.pi],filename="/traj_1dof.pt"):
    data_maker = pendelum1dof(data[0],data[1],data[2])
    t = torch.linspace(0,T,steps)
    dim = 1
    inits = torch.cat((range_of_angles[0]+ torch.rand(samples,1,dim)*(range_of_angles[1]-range_of_angles[0]),
                    torch.zeros(samples,1,dim)),dim=-1)
    for i in range(inits.shape[0]):
        H = dof1_hamiltonian_eval(data_maker,inits[i,:,:].unsqueeze(0))
        #print(H)
        while(torch.abs(H[0]) < 1):
            print("found under H 1: change")
            inits[i,:,0:dim]=range_of_angles[0]+ torch.rand(1,1,dim)*(range_of_angles[1]-range_of_angles[0])
            H = dof1_hamiltonian_eval(data_maker,inits[i,:,:].unsqueeze(0))
    trajectories = dof1_dataset(data_maker.to(DEVICE),t.to(DEVICE),inits.to(DEVICE))
    H_temp = torch.zeros(steps,samples,1,1)
    for i in range(trajectories.shape[1]):
        H_temp[:,i,0,:] = dof1_hamiltonian_eval(data_maker,trajectories[:,i,:,0:2])
    traj= torch.cat((trajectories,H_temp),dim=-1)
    print("made: {}".format(traj.shape))
    torch.save(traj,ROOT_PATH + filename)
    return traj

def create_2dof_pendelum_dataset(samples,T,steps,data = [1,1,1,1,9.81],range_of_angles=[-torch.pi,torch.pi],filename="/traj_2dof.pt"):
    data_maker = pendelum2dof(data[0],data[1],data[2],data[3],data[4])
    t = torch.linspace(0,T,steps)
    dim = 2
    inits = torch.cat((range_of_angles[0]+ torch.rand(samples,1,dim)*(range_of_angles[1]-range_of_angles[0]),
                    torch.zeros(samples,1,dim)),dim=-1)
    for i in range(inits.shape[0]):
        H = dof2_hamiltonian_eval(data_maker,inits[i,:,:].unsqueeze(0))
        #print(H)
        while(torch.abs(H[0]) < 1):
            print("found under H 1: change")
            inits[i,:,0:dim]=range_of_angles[0]+ torch.rand(1,1,dim)*(range_of_angles[1]-range_of_angles[0])
            H = dof2_hamiltonian_eval(data_maker,inits[i,:,:].unsqueeze(0))
    trajectories = dof2_dataset(data_maker.to(DEVICE),t.to(DEVICE),inits.to(DEVICE))
    H_temp = torch.Tensor(steps,samples,1,1)
    for i in range(trajectories.shape[1]):
        H_temp[:,i,0,:] = dof2_hamiltonian_eval(data_maker,trajectories[:,i,:,0:4])
    traj= torch.cat((trajectories,H_temp),dim=-1)
    print("made: {}".format(traj.shape))
    torch.save(traj,ROOT_PATH + filename)
    return traj
def create_3dof_pendelum_dataset(samples,T,steps,data = [1,1,1,1,1,1,9.81],range_of_angles=[-torch.pi,torch.pi],filename="/traj_3dof.pt"):
    data_maker = pendelum3dof(data[0],data[1],data[2],data[3],data[4],data[5],data[6])
    t = torch.linspace(0,T,steps)
    dim =3
    inits = torch.cat((range_of_angles[0]+ torch.rand(samples,1,dim)*(range_of_angles[1]-range_of_angles[0]),
                    torch.zeros(samples,1,dim)),dim=-1)
    for i in range(inits.shape[0]):
        H = dof3_hamiltonian_eval(data_maker,inits[i,:,:].unsqueeze(0))
        #print(H)
        while(torch.abs(H[0]) < 1):
            print("found under H 1: change")
            inits[i,:,0:dim]=range_of_angles[0]+ torch.rand(1,1,dim)*(range_of_angles[1]-range_of_angles[0])
            H = dof3_hamiltonian_eval(data_maker,inits[i,:,:].unsqueeze(0))
    trajectories = dof3_dataset(data_maker.to(DEVICE),t.to(DEVICE),inits.to(DEVICE))
    H_temp = torch.Tensor(steps,samples,1,1)
    for i in range(trajectories.shape[1]):
        H_temp[:,i,0,:] = dof3_hamiltonian_eval(data_maker,trajectories[:,i,:,0:6])
    traj= torch.cat((trajectories,H_temp),dim=-1)
    print("made: {}".format(traj.shape))
    torch.save(traj,ROOT_PATH + filename)
    return traj


def create_4dof_pendelum_dataset(samples,T,steps,data = [1,1,1,1,1,1,1,1,9.81],range_of_angles=[-torch.pi,torch.pi],filename="/traj_4dof.pt"):
    data_maker = pendelum4dof(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
    t = torch.linspace(0,T,steps)
    dim =4
    inits = torch.cat((range_of_angles[0]+ torch.rand(samples,1,dim)*(range_of_angles[1]-range_of_angles[0]),
                    torch.zeros(samples,1,dim)),dim=-1)
    for i in range(inits.shape[0]):
        H = dof4_hamiltonian_eval(data_maker,inits[i,:,:].unsqueeze(0))
        #print(H)
        while(torch.abs(H[0]) < 1):
            print("found under H 1: change")
            inits[i,:,0:dim]=range_of_angles[0]+ torch.rand(1,1,dim)*(range_of_angles[1]-range_of_angles[0])
            H = dof4_hamiltonian_eval(data_maker,inits[i,:,:].unsqueeze(0))
    trajectories = dof4_dataset(data_maker.to(DEVICE),t.to(DEVICE),inits.to(DEVICE))
    H_temp = torch.Tensor(steps,samples,1,1)
    for i in range(trajectories.shape[1]):
        H_temp[:,i,0,:] = dof4_hamiltonian_eval(data_maker,trajectories[:,i,:,0:8])
    traj= torch.cat((trajectories,H_temp),dim=-1)
    print("made: {}".format(traj.shape))
    torch.save(traj,ROOT_PATH + filename)
    return traj


def datasets_creator(samples,T,points):
    print("DOF1 started")
    traj1 = create_1dof_pendelum_dataset(samples,T,points)
    print("DOF1 finished")
    print("DOF2 started")
    traj2 = create_2dof_pendelum_dataset(samples,T,points)
    print("DOF2 finished")
    print("DOF3 started")
    traj3 = create_3dof_pendelum_dataset(samples,T,points)
    print("DOF3 finished")
    print("DOF4 started")
    traj4 = create_4dof_pendelum_dataset(samples,T,points)
    print("DOF4 finished")

if __name__ == "__main__":
    test = datasets_creator(300,1.27,128) #dt =0.01
    #traj1 = create_1dof_pendelum_dataset(25,2.5,251)
    #print(traj1.shape)
    #print(traj1[:,:,0,-1])
    #traj2 = create_2dof_pendelum_dataset(25,2.5,251)
    #print(traj2.shape)
    #print(traj2[:,:,0,-1])
    #traj3 = create_3dof_pendelum_dataset(1000,2.55,256)
    #print(traj3.shape)
    #print(traj3[:,:,0,-1])

