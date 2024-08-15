import torch
import math
import torch.nn as nn
from samples import get_3body_samples
from torchdiffeq import odeint_adjoint as odeint
torch.set_printoptions(precision=6) 
from tqdm import tqdm


class ham_func_3body(nn.Module):
    def __init__(self,M,G,device):
        super(ham_func_3body,self).__init__()
        self.A = torch.zeros(12,12).double().to(device)
        self.A[0,6]=1/M[0]
        self.A[1,7]=1/M[0]
        self.A[2,8]=1/M[1]
        self.A[3,9]=1/M[1]
        self.A[4,10]=1/M[2]
        self.A[5,11]=1/M[2]
        self.G = G
        self.M = M
    # update the matrix in every step    
    def updateA(self,r12,r23,r13):
        self.A[6,0] =-(self.G*self.M[0]*self.M[1]/(r12**3) + self.G*self.M[0]*self.M[2]/(r13**3))
        self.A[6,2] =self.G*self.M[0]*self.M[1]/(r12**3)
        self.A[6,4] =self.G*self.M[0]*self.M[2]/(r13**3)
        self.A[7,1] =-(self.G*self.M[0]*self.M[1]/(r12**3) + self.G*self.M[0]*self.M[2]/(r13**3))
        self.A[7,3] =self.G*self.M[0]*self.M[1]/(r12**3)
        self.A[7,5] =self.G*self.M[0]*self.M[2]/(r13**3)
        
        self.A[8,0] =self.G*self.M[0]*self.M[1]/(r12**3)
        self.A[8,2] =-(self.G*self.M[0]*self.M[1]/(r12**3) + self.G*self.M[1]*self.M[2]/(r23**3))
        self.A[8,4] =self.G*self.M[1]*self.M[2]/(r23**3)
        self.A[9,1] =self.G*self.M[0]*self.M[1]/(r12**3)
        self.A[9,3] =-(self.G*self.M[0]*self.M[1]/(r12**3) + self.G*self.M[1]*self.M[2]/(r23**3))
        self.A[9,5] =self.G*self.M[1]*self.M[2]/(r23**3)
        
        self.A[10,0] =self.G*self.M[0]*self.M[2]/(r13**3)
        self.A[10,2] =self.G*self.M[1]*self.M[2]/(r23**3)
        self.A[10,4] =-(self.G*self.M[0]*self.M[2]/(r13**3) + self.G*self.M[1]*self.M[2]/(r23**3))
        self.A[11,1] =self.G*self.M[0]*self.M[2]/(r13**3)
        self.A[11,3] =self.G*self.M[1]*self.M[2]/(r23**3)
        self.A[11,5] =-(self.G*self.M[0]*self.M[2]/(r13**3) + self.G*self.M[1]*self.M[2]/(r23**3))
    
    def forward(self,t,y):
        r12 = torch.norm(y[0,0:2]-y[0,2:4])
        r13 = torch.norm(y[0,0:2]-y[0,4:6])
        r23 = torch.norm(y[0,2:4]-y[0,4:6])
        #print(r12 , r23 , r13)
        self.updateA(r12,r23,r13)
        #print("shape y in creator: {}".format(y.shape))
        #print(self.A.shape)
        return torch.matmul(y,self.A.T)
   


class threebody:
    def __init__(self,samples_dict,G=1.0,device=torch.device("cuda")):
        self.data = samples_dict # the samples.py is one big dictonary of data 
        self.names = list(samples_dict.keys())
        self.device = device
        self.G = G
    def __str__(self):
        return "threebody : {}".format(self.names)
    
    # make a trajectory based on a key for example: fig8
    def make_sample(self,key,points,alpha):
        init_dict = self.data[key]
        y0 = torch.zeros(1,12).double().to(self.device)
        y0[0,0:2] =torch.Tensor(init_dict["q1"]).double()
        y0[0,2:4] =torch.Tensor(init_dict["q2"]).double()
        y0[0,4:6] =torch.Tensor(init_dict["q3"]).double()
        y0[0,6:8] =torch.Tensor(init_dict["v1"]).double()
        y0[0,8:10] =torch.Tensor(init_dict["v2"]).double()
        y0[0,10:12] = torch.Tensor(init_dict["v3"]).double()
        T = init_dict["T"]
        H = init_dict["H"]
        t = torch.linspace(0,T,points).to(self.device)
        F = ham_func_3body(init_dict["M"],self.G,self.device)
        if alpha != 0:
            q = (y0[:,0:2] + y0[:,2:4] + y0[:,4:6]).to(self.device)/3  # calculating mass middlepoint
            #print("alpha found!")
            R = torch.Tensor([[math.cos(alpha),-math.sin(alpha)],
                              [math.sin(alpha),math.cos(alpha)]]).double().to(self.device) # rotational matrix on 2D plane
            ## rotating the y0 around mass middlepoint
            y0[:,0:2] = y0[:,0:2] @ R.T - q @ R.T + q
            y0[:,2:4] = y0[:,2:4] @ R.T - q @ R.T + q
            y0[:,4:6] = y0[:,4:6] @ R.T - q @ R.T + q
            y0[:,6:8] = y0[:,6:8] @ R.T - q @ R.T + q
            y0[:,8:10] = y0[:,8:10] @ R.T - q @ R.T + q
            y0[:,10:12] = y0[:,10:12] @ R.T - q @ R.T + q
            
        x = odeint(F,y0,t,method="dopri5").to(self.device)
        dx = self.make_dx(F,x).to(self.device)
        #print(x.shape)
        #print(dx.shape)
        #print(" nans from threebody: {}".format(torch.sum(dx.isnan())))
        return t.float() , x.float(), dx.float() , H
    # hamiltonian checker
    def make_dx(self,F,x):
        dx = torch.Tensor(x.shape).to(self.device)
        for i in range(x.shape[0]):
            dx[i,:,:] = F(0,x[i,:,:]).to(self.device)
        return dx
    def hamiltonian(self,dataset):
        T = 0.5*(torch.norm(dataset[:,:,0,6:8],dim=-1)**2 + torch.norm(dataset[:,:,0,8:10],dim=-1)**2 + torch.norm(dataset[:,:,0,10:12],dim=-1)**2).to(self.device)
        r12 = torch.norm(dataset[:,:,0,0:2]-dataset[:,:,0,2:4],dim=-1).to(self.device)
        r13 = torch.norm(dataset[:,:,0,0:2]-dataset[:,:,0,4:6],dim=-1).to(self.device)
        r23 = torch.norm(dataset[:,:,0,2:4]-dataset[:,:,0,4:6],dim=-1).to(self.device)
        U = -(1/r12 + 1/r13 + 1/r23).to(self.device)
        return T + U
        
    def hamiltonian_check(self,x,H):
        
        T = 0.5*(torch.norm(x[:,0,6:8],dim=1)**2 + torch.norm(x[:,0,8:10],dim=1)**2 + torch.norm(x[:,0,10:12],dim=1)**2)
        r12 = torch.norm(x[:,0,0:2]-x[:,0,2:4],dim=1)
        r13 = torch.norm(x[:,0,0:2]-x[:,0,4:6],dim=1)
        r23 = torch.norm(x[:,0,2:4]-x[:,0,4:6],dim=1)
        U = -(1/r12 + 1/r13 + 1/r23)
        
        meanH = torch.mean(T+U).float()
        stdH = torch.std(T+U).float()
        """
        H_s=self.hamiltonian(x.unsqueeze(1))
        stdH = torch.std(H_s).float()
        meanH = torch.std(H_s).float()
        """
        if stdH/meanH < 0.01:
            print("VALID")
            print("H:    {}\nmean: {}\nstd:   {}".format(H,meanH,stdH))
            return True
        else:
            print("BAD")
            print("H:    {}\nmean: {}\nstd:   {}".format(H,meanH,stdH))
            return False
    # make a dataset only using one key
    def dataset_onekind(self,key,nsamples, points,phi_span=[0 ,2*math.pi]):
        dataset = torch.Tensor(points,nsamples,1,12)
        ddataset = torch.Tensor(points,nsamples,1,12)
        H = torch.Tensor(points,nsamples,1,1)
        for i in tqdm(range(nsamples)):
            t,x,dx,H_real = self.make_sample(key,points,phi_span[0]+torch.rand(1,)*(phi_span[1]-phi_span[0]))
            #print("from threebody: {}".format(torch.sum(dx.isnan())))
            h = self.hamiltonian(x.unsqueeze(1))
            #print(H.shape)
            dataset[:,i,:,:] = x
            ddataset[:,i,:,:] = dx
            H[:,i,0,:] = h 
            
        return t,dataset, ddataset, H
    ## make one dataset with mixed keys ""haven't been used""
    def dataset_mixed(self,points,keys,samples_pro_kind):
        nsamples = len(keys) * samples_pro_kind
        dataset = torch.Tensor(points,nsamples,1,13)
        for i in tqdm(range(nsamples)):
            print(int(i/len(keys)))
            t,x,H = self.make_sample(keys[int(i/len(keys))],points,torch.rand(1,)*2*torch.pi)
            dataset[:,i,:,0:12] = x
            dataset[:,i,0,12] = t
        return dataset
    ## make one trajectory      
    def dataset_one(self,points,key,alpha=0):
        dataset = torch.Tensor(points,1,1,13)
        t,x,H = self.make_sample(key,points,alpha)
        dataset[:,0,:,0:12] = x
        dataset[:,0,0,12] = t  
        return dataset
    ## test if trajectory obeys static hamiltonian value
    
    def make_dataset(self,key,nsamples, points,phi_span=[0 ,2*math.pi]):
        t,x,dx,H = self.dataset_onekind(key,nsamples, points,phi_span)
        return t,x, dx, H
    
    def TEST(self,points):
        print("SAMPLES_TEST\n")
        samples_list = list(self.data.keys())
    
        N = len(samples_list)
        TRIES = 25
        valid = 0
        for i in range(N):
            print(self.data[samples_list[i]]["name"])
            for j in tqdm(range(TRIES)):
                t,x,dx,H = self.make_sample(samples_list[i],points,torch.rand(1,)*2*torch.pi)
                flag = self.hamiltonian_check(x,H)
                if flag:
                    valid+=1
                    print("passed")
    
    
        if valid/(N*TRIES)<0.9:
            print("TEST_FAILED {}/{}".format(valid,N*TRIES))
        else:
            print("TEST_PASSED {}/{}".format(valid,N*TRIES))
    def new_test(self,points=256,batches=3,repeats=2):
        print("SAMPLES_TEST\n")
        samples_list = list(self.data.keys())
        acc = 0
        N = len(samples_list)
        for i in range(repeats):
            print("run {}".format(i+1))
            for sample in samples_list:
                print(sample)
                t,x,dx,H = self.make_dataset(sample,batches,points)
                H = H.flatten()
                H_mean = torch.mean(H)
                H_std = torch.std(H)
                if H_std >= 8e-5:
                    print("INVALID")
                    print("mean: {}\nstd:   {}".format(H_mean,H_std))
                else:
                    print("VALID")
                    print("mean: {}\nstd:   {}".format(H_mean,H_std))
                    acc+=1
        print("{}/{} == {}%".format(acc,N*repeats,(acc/(N*repeats))*100))
                
        


if __name__ == "__main__":
    SAMPLES = get_3body_samples()
    creator = threebody(SAMPLES)
    #keys = list(creator.data.keys())
    #creator.TEST(128)
    #dataset0 = creator.dataset_onekind("yarn",32,512)
    #dataset1 = creator.dataset_one(512,"yarn")
    #dataset2 = creator.dataset_mixed(512,keys,8)
    #t,x,dx,H = creator.make_dataset("fig8",10,128)
    #print(H)
    creator.new_test()
    #creator.TEST(128)
            
            
            
        

        
        
    

