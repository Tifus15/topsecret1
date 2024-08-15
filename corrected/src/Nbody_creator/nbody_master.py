import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def getAcc( pos, mass, G, softening ):
	"""
    Calculate the acceleration on each particle due to Newton's Law 
	pos  is an N x 3 matrix of positions
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	softening is the softening length
	a is N x 3 matrix of accelerations
	"""
	# positions r = [x,y,z] for all particles
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores 1/r^3 for all particle pairwise particle separations 
	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
	inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

	ax = G * (dx * inv_r3) @ mass
	ay = G * (dy * inv_r3) @ mass
	az = G * (dz * inv_r3) @ mass
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))

	return a
	
def getEnergy( pos, vel, mass, G ):
	"""
	Get kinetic energy (KE) and potential energy (PE) of simulation
	pos is N x 3 matrix of positions
	vel is N x 3 matrix of velocities
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	KE is the kinetic energy of the system
	PE is the potential energy of the system
	"""
	# Kinetic Energy:
	KE = 0.5 * np.sum(np.sum( mass * vel**2 ))


	# Potential Energy:

	# positions r = [x,y,z] for all particles
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores 1/r for all particle pairwise particle separations 
	inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
	inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

	# sum over upper triangle, to count each interaction only once
	PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
	
	return KE, PE;

def make_one(N = 15,
		 t = np.linspace(0,3.0,301),
		 G= 1.0,
		 eps = 1.0,
		 r = 1.0):
	dt = t[1]-t[0]
	mass1 = np.ones((N,1))
	pos1  = np.random.randn(N,3) * r
	vel1  = np.random.randn(N,3)
	extra_pos = np.random.randn(1,3)
	extra_vel = np.random.randn(1,3)
	mass2 = np.ones((N+1,1))
	## creating almost aquivalent second set
	vel1 -= np.mean(mass1 * vel1,0) / np.mean(mass1)
	pos2 = np.concatenate((pos1,extra_pos),0)
	vel2 = np.concatenate((vel1,extra_vel),0)
	# calculate initial gravitational accelerations
	acc1 = getAcc( pos1, mass1, G, eps)
	acc2 = getAcc( pos2, mass2, G, eps)
	
	# calculate initial energy of system
	KE1, PE1  = getEnergy( pos1, vel1, mass1, G )
	KE2, PE2  = getEnergy( pos2, vel2, mass2, G )
	Nt = int(np.ceil(t[-1]/(t[1]-t[0])))
	save_pos1=np.zeros((Nt+1,N,3))
	save_vel1=np.zeros((Nt+1,N,3))
	save_acc1=np.zeros((Nt+1,N,3))
	save_pos2=np.zeros((Nt+1,N+1,3))
	save_vel2=np.zeros((Nt+1,N+1,3))
	save_acc2=np.zeros((Nt+1,N+1,3))
	KE1_save = np.zeros(Nt+1)
	PE1_save = np.zeros(Nt+1)
	KE2_save = np.zeros(Nt+1)
	PE2_save = np.zeros(Nt+1)
	KE1_save[0] = KE1
	KE2_save[0] = KE2
	PE1_save[0] = PE1
	PE2_save[0] = PE2
	save_pos1[0,:,:] = pos1
	save_vel1[0,:,:] = vel1
	save_acc1[0,:,:] = acc1
	save_pos2[0,:,:] = pos2
	save_vel2[0,:,:] = vel2
	save_acc2[0,:,:] = acc2
	for i in range(Nt):
		acc1 = getAcc( pos1, mass1, G, eps )
		acc2 = getAcc( pos2, mass2, G, eps )
		if i == 0:
			#leapfrog DA
			pos1 = pos1 + vel1 *dt + acc1*(dt**2)/2
			acc1_2 = getAcc( pos1, mass1, G, eps)
			vel1 = vel1 + (acc1 + acc1_2)*dt/2
			#second 
			pos2 = pos2 + vel2 *dt + acc2*(dt**2)/2
			acc2_2 = getAcc( pos2, mass2, G, eps)
			vel2 = vel2 + (acc2 + acc2_2)*dt/2
		else:
			## beeman
			acc1i = getAcc( pos1, mass1, G, eps )
			acc1o = getAcc(save_pos1[i-1,:,:],mass1,G,eps)
			pos1 = pos1 + vel1*dt + (1/6)*(4*acc1i - acc1o)*(dt**2)
			acc1ii = getAcc( pos1, mass1, G, eps )
			vel1 = vel1 + (1/6)*(2*acc1ii+5*acc1i-acc1o)*dt
			#second
			acc2i = getAcc( pos2, mass2, G, eps )
			acc2o = getAcc(save_pos2[i-1,:,:],mass2,G,eps)
			pos2 = pos2 + vel2*dt + (1/6)*(4*acc2i - acc2o)*(dt**2)
			acc2ii = getAcc( pos2, mass2, G, eps )
			vel2 = vel2 + (1/6)*(2*acc2ii+5*acc2i-acc2o)*dt
		KE1, PE1  = getEnergy( pos1, vel1, mass1, G )
		KE2, PE2  = getEnergy( pos2, vel2, mass2, G )
		save_pos1[i+1,:,:]=pos1
		save_vel1[i+1,:,:]=vel1
		save_acc1[i+1,:,:]=acc1
		save_pos2[i+1,:,:]=pos2
		save_vel2[i+1,:,:]=vel2
		save_acc2[i+1,:,:]=acc2
		KE1_save[i+1] = KE1
		PE1_save[i+1] = PE1
		KE2_save[i+1] = KE2
		PE2_save[i+1] = PE2
	return 	(save_pos1, save_vel1, save_acc1, KE1_save,PE1_save),(save_pos2, save_vel2, save_acc2, KE2_save,PE2_save)
		
			
	

	
	


if __name__== "__main__":
	system1,system2 = make_one(N=10)
	print(system1[3]+system1[4])






