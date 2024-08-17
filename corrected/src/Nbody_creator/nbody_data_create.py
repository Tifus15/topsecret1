import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from nbody import simulate
import os
from tqdm import tqdm
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
def simulation_start(times=25, 
                     bodies = 15,
                     t=torch.linspace(0,1.27,128),
                     sim=True):
    for i in tqdm(range(times)):
        flag = False
        while not flag:
            p = np.random.randn(bodies,3)
            v = np.random.randn(bodies,3)
            p_extra = np.random.randn(1,3)
            v_extra = np.random.randn(1,3)
            p_new = np.concatenate((p,p_extra),0)
            v_new = np.concatenate((v,v_extra),0)
            name1,H1 = simulate(p,v,N=bodies, t= 0, tEnd= t[-1], dt = 0.01, r=0.4, softening= 0.01, G = 1, plotRealTime=sim)
            name2,H2 = simulate(p_new,v_new,N=bodies+1, t= 0, tEnd= t[-1], dt = 0.01, r=0.4, softening= 0.01, G = 1, plotRealTime=sim)               
            CRIT1 =torch.std(torch.tensor(H1))
            CRIT2 =torch.std(torch.tensor(H2))
            if CRIT1 > 0.1 and CRIT2 > 0.1:
                print("{} > 0.1, {} > 0.1,  FAILED".format(CRIT1,CRIT2))
                os.remove(name1)
                os.remove(name2)
                flag = False
            else:
                print("{} < 0.1, {} < 0.1,  PASSED".format(CRIT1,CRIT2))
                flag = True
        time.sleep(1.0)
    for i in range(2):

        arr = os.listdir()
        arr_new=list(filter(lambda k: str(bodies+i)+'.npy' in k, arr)) 
        print(len(arr_new))
        qp = []
        H = []
        out = []
        for npy in arr_new:
            with open(npy, 'rb') as f:
                pos = torch.tensor(np.load(f))
                vel = torch.tensor(np.load(f))
                acc = torch.tensor(np.load(f))
                t = torch.tensor(np.load(f))
                mass = np.load(f)
                K = torch.tensor(np.load(f))
                P = torch.tensor(np.load(f))
                print("K shape {}".format(K.shape))
                print("P shape {}".format(P.shape))            
            qp.append(torch.cat((pos.unsqueeze(1),vel.unsqueeze(1)),dim=-1))
            print("trajectory shape:{}".format(qp[-1].shape))
            H.append((K+P).reshape(-1,1,1))
            print("energy shape {}".format(H[-1].shape))
            time.sleep(1.0) 
            os.remove(npy)	
        out = torch.cat((qp),dim=1)
        H_out = torch.cat((H),dim=1)
        print("TRAJECTORY {}".format(out.shape))
        print("ENERGY {}".format(H_out.shape))
        filename = "nbody_" + str(bodies+i) + "_traj.pt"
        H_file = "nbody_" + str(bodies+i) + "_H.pt"
        torch.save(out,ROOT_PATH+"/"+filename)
        torch.save(H_out,ROOT_PATH+"/"+H_file)
        #print(H_out)



if __name__ == "__main__":
    #start if from the folder corrected
    simulation_start(500,12,t=torch.linspace(0,1.27,128),sim=False)