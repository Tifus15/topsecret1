
import torch 
import torch.nn as nn 
import math
# import device_util as device_util # package which connects the modules to device
from torchdiffeq import odeint_adjoint as odeint



class oscigradH(nn.Module):
    """
    class is used for easier calculating the derivative of the oscilator 

    ...

    Attributes
    ----------
    k : float
        value of the coeficient
    m : float
        value of the mass
    A : torch.FloatTensor
        Matrix for which represents F(y) for calculating a derivative on position y
    

    Methods
    -------
    forward(t,y)
        method for calcualting the derivative
    """
    def __init__(self,k,m,device = torch.device("cuda")):
        """
        Parameters
        ----------
        k : float
            value of the coeficient
        m : float
            value of the mass
        device: torch.device, optional
            device on which is module or matrix A stored 
        """
        super(oscigradH,self).__init__() 
        self.A=torch.zeros(2,2).to(device) # matrix init 
        self.A[0,1]=1/m 
        self.A[1,0]=-k 
    def forward(self,t,y):
        """
        method for calcualting the derivative

        Parameters
        ----------
        t : float
            time position, it has usage for a odeint method from package torchdiffeq
        y: torch.Tensor
            position of the timestep  

        Returns
        -------
        torch.Tensor
            derivative at the position y
          
        """
        return torch.matmul(y,self.A.T) #dy_i = A @ y_i


class oscilator:  
    """
    A class used to represent an oscilator an its data

    ...

    Attributes
    ----------
    k : float
        value of the coeficient
    m : float
        value of the mass
    F : oscigradH
        function which represents derivatve calculation
    device: torch.device, optional
        device on which is module stored
    Methods
    -------
    make_inits(samples, H_span=[1,5])
        creates a initial points for the dataset
    
    hamiltonian(dataset)
        calculates hamiltonian energy of the dataset
    
    kinetic(dataset)
        calculates kinetic energy of the dataset
    
    potential(dataset)
        calculates potential energy of the dataset
    
    make_one(points, H_span=[1,5])
        (depricated) creates one trajectory of the dataset 
    
    make_dataset(points,samples,H_span=[1,5])
        creates whole dataset: trajectory derivatives and hamiltonian energy
    """  
    def __init__(self,m,k,device = torch.device("cuda")): # mass and coeficient
        """
        Attributes
        ----------
        k : float
            value of the coeficient
        m : float
            value of the mass
        F : oscigradH
            function which represents derivatve calculation
    
        """
        self.device = device
        self.k = k
        self.m = m
        self.F = oscigradH(k,m,device).to(self.device)
        
    
    def make_inits(self,samples,H_span=[1,5]):
        """creates a initial points for the dataset

        Parameters
        ----------
        samples : int
            Number of samples for the dataset initial positions
        H_span : tuple(float,float), optional
            region of the energy [min, max]
        device: torch.device, optional
            device on which is module stored
        Returns
        -------
        torch.Tensor
            3 dim FloatTensor of positions and moments (example of shape [samples, 1, 2]) 
       
        """
       
        Hs = H_span[0]*torch.rand(samples,)*(H_span[1]-H_span[0])# randomsing possible Energies
        a = torch.sqrt(2*self.m*Hs) # calculating a for elipse 
        b = torch.sqrt(2*Hs/self.k) # calculating b for elipse
        phis = torch.rand(samples,)*2*torch.pi # random angles 
        
        p = a * torch.cos(phis) # calculating moment
        p = torch.unsqueeze(p,dim=-1)
   
        x = b * torch.sin(phis) # calculating position
        x = torch.unsqueeze(x,dim=-1)

        return torch.cat((x,p),dim=-1) # bind positions with moments
    
    def hamiltonian(self,dataset):
        """ 
        method for hamiltonian energy calculation
        Hamiltonian energy is sum of kinetic energy and potential energy
        Parameters
        ----------
        dataset : torch.Tensor
            shapes - [len(t),batches,1,2], [len(t),1,2], [batches,1,2]
        
        Returns
        -------
        torch.Tensor
            hamiltonian energy
        
        """
        T = self.kinetic(dataset) # kinetic energy
        U = self.potential(dataset) # potential energy
        H = T+U
        return H
    
    def kinetic(self,dataset):
        """ 
        method for kinetic energy calculation
        
        Parameters
        ----------
        dataset : torch.Tensor
            shapes - [len(t),batches,1,2], [len(t),1,2], [batches,1,2]
        
        Returns
        -------
        torch.Tensor
            kinetic energy
        
        """
        if dataset.dim() > 3:
            dataset = dataset.squeeze()
        if dataset.dim() < 3:
            dataset = torch.unsqueeze(dataset,dim=1)
        p = dataset[:,:,1]
        T = p.square()/(2*self.m)
        return T
    
    def potential(self,dataset):
        """ 
        method for potential energy calculation
        
                        
        Parameters
        ----------
        dataset : torch.Tensor
            shapes - [len(t),batches,1,2], [len(t),1,2], [batches,1,2]
        
        Returns
        -------
        torch.Tensor
            potential energy
        
        """
        if dataset.dim() > 3:
            dataset = dataset.squeeze()
        if dataset.dim() < 3:
            dataset = torch.unsqueeze(dataset,dim=1)
        x = dataset[:,:,0]
        U = self.k*x.square()/2
        return U

    
    def _make_one(self,points, H_span=[1,5]):
        """::depricated::
        method to create one trajectory of the oscilator movement for whole period
        
        
        Parameters
        ----------
        points: int
            number of datapoints which represent one state in time
        H_span : tuple(float,float), optional
            region of the energy [min, max]

        Returns
        -------
        torch.Tensor 
            3dim Tensor of positions and moments (example of shape [points, 1, 2])
        
        torch.Tensor    
            1dim time space/discretisation 
            
        """
        omega = math.sqrt(self.k/self.m) # calculating omega
        T = 2*math.pi/omega # calculating period

        t = torch.linspace(0,T,points).to(self.device) # time space/discretisation
        y=self.make_inits(1,H_span=H_span).to(self.device) # creating initial position and moments
        data = odeint(self.F,y[0,:],t,method="rk4") # solve the differential equation dy = F(y) when y(0)=y0
        
        return data.unsqueeze(dim=1), t
    
    def make_dataset(self,points,samples,H_span=[1,5]):
        """
        method to create whole dataset for oscilator movement: trajectory, derivative and hamiltonian
        
        Parameters
        ----------
        points: int
            number of datapoints which represent one state in time
        samples : int
            Number of samples for the dataset initial positions
        H_span : tuple(float,float), optional
        
        Returns
        -------
        torch.Tensor 
            3dim Tensor of positions and moments (example of shape [points, 1, 2])
        
        torch.Tensor 
            3dim Tensor of derivatives of positions and moments (example of shape [points, 1, 2])
        
        torch.Tensor    
            1dim time space/discretisation 
        torch.Tensor
            2dim or 1dim Tensor of hamiltonian energy
            
        """
        omega = math.sqrt(self.k/self.m)
        T = 2*math.pi/omega
        t = torch.linspace(0,T,points).to(self.device)
        y=self.make_inits(samples,H_span=H_span).to(self.device)
        y=torch.unsqueeze(y,dim=1)
        #print(y.shape)
        data = odeint(self.F,y,t,method="rk4")
        ddata = self.F(0,data)
        H= self.hamiltonian(data)

        return data,ddata, t, H

    

if __name__ == "__main__":
    creator = oscilator(4,0.5)
    data,ddata,t,H = creator.make_dataset(100,10)
    print(creator.hamiltonian(data)[:,0])
    print(H[:,0])




    

    
