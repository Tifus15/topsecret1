import numpy as np
import matplotlib.pyplot as plt
zero = 1e-5
### extract data
from device_util import ROOT_PATH

def load_Nbody_data(filename):
	with open(filename, 'rb') as f:
		pos = np.load(f)
		vel = np.load(f)
		t = np.load(f)
		mass = np.load(f)
		K = np.load(f)
		P = np.load(f)
	return pos, vel, t, mass, K, P 
    
"""
with open('nbody_200_01152024172749.npy', 'rb') as f:
	pos = np.load(f)
	vel = np.load(f)
	t = np.load(f)
	mass = np.load(f)
	K = np.load(f)
	P = np.load(f)
	
print(pos.shape)
print(vel.shape)
print(t.shape)
print(mass.shape)
print(K.shape)
print(P.shape)
print(mass)

print(np.min(mass))
r_max = np.min(mass)**2/zero
print(r_max)


for v in range(len(t)):
	A = np.zeros((pos.shape[1],pos.shape[1]))
	for i in range(pos.shape[1]):
		for j in range(i,pos.shape[1]):
			temp = np.linalg.norm(pos[v,i,:]-pos[v,j,:])
			if temp > r_max:
				A[i,j]=0
				A[j,i]=0
			else:
				A[i,j]=1
				A[j,i]=1
	plt.figure()
	plt.matshow(A)
	plt.show()
		
			
"""
		
		
		
