import numpy as np
import time
import os
import sys

from multiprocessing import Process, Pipe, cpu_count, Array
from ctypes import c_double
from tqdm import tqdm
from matplotlib import pyplot as plt

#own imports
from BH import Particle, Cell, Tree, BHF_kickstart, connection_receiveAndClose, processes_joinAndTerminate
from ODEInt import leapfrog
from Animator import AnimateOrbit
from GalGen import generate


def particles2arr(particles):
	r = np.array([p.r for p in particles])
	v = np.array([p.v for p in particles])
	return r,v

def updateparticles(r,v, particles):
	for indx,p in enumerate(particles):
		p.r = r[indx]
		p.v = v[indx]
	return particles

def GetSituation(r,colors):
	plt.figure(figsize=(10,10))
	plt.scatter([p.r[0] for p in particles],[p.r[1] for p in particles],color=colors,s=0.4)
	plt.grid()
	plt.ylim(-400,400)
	plt.xlim(-400,400)
	plt.show()


if __name__ == "__main__":
	'''

		1. Nparticles indicates the amount of particles in the simulation recommended is an amount between 1000 and 10000 = 1k-10k
		2. The amount of frames for test runs to observe any flaws should be between 70 and 200 to obtain reasonable computing time. (within 5mins to 1h)
		3. Theta indicates the BH performance or approximation. The higher θ, the faster the algorithm is but less accurate. Recommended is: θ=[0.5,0.8]
		4. The algorithm typically follows the following idea: GENERATE GALAXY + INITIAL CONDITIONS -> START COMPUTING FRAMES <--> (BARNES HUT ALGORITHM -> INTEGRATOR); --> MOVIE
		5. A mass of M = 1E9 to 1E12 is recommended.
		6. The program automatically detects the maximumum number of possible cpu cores on your computer and will maximize its usage. WINDOWS, LINUX, and MAC OS are supported.
		7. The potential used can be altered; Current options are: "plummer"
		8. The recommended timestep dt, based on obtaining smooth orbits, is recommended to be smaller than 0.01 Gyrs.
		9. r0 is directly related to the maximum radius of the galaxy and acts as a scaling radius.
		10. Dispersion (disp) is the random motions of the stars relative to the "systemic" velocity.

	'''
	Nparticles = 2000
	θ = 0.7
	dt = 0.005
	Mtot = 10 ** 12
	r0 = 5 # <--
	frames = 200
	disp = 1600

	r, v = generate(Nparticles, Mtot, r0, disp, type_="plummer") #'plummer2D' gives radial motion of particles outward.  'plummer' just remains
 #stationary
	#r = np.array([[-1,0.0001],[1,0.0001]])
	#v = np.array([[0,0],[0,0]])
	r = r[:,:2]
	v = v[:,:2]
	L = 200#2 * np.linalg.norm(r[-1])

	particles = [Particle(r[i], v[i], m=(Mtot * 10 ** (-2) / Nparticles)) for i in range(Nparticles)] #:,i
	colors = ['orange' if i == 10 else 'b' for i in range(Nparticles)]

	SDV = [v] # Storage of Data for V
	SDR = [r] # Storage of Data for R

	for frame in tqdm(range(frames)):
		#GetSituation(r,colors)	
		for k in range(2):
			# compute the location of the Center of Mass (COM) and total mass for the
			# ROOT cell
			Rgal_CM = np.sum([p.m * p.r for p in particles]) / np.sum([p.m for p in particles])
			Mgal = np.sum([p.m for p in particles])

			# initialize ROOT cell
			ROOT = Cell(np.array([0, 0]), L, parent=None, M=Mgal, R_CM=Rgal_CM)

			#BUILD TREE
			Tree(ROOT, particles)
		
			################################################
			##    COMPUTE FORCES USING MULTIPROCESSING ##
			################################################
			try:
				N_CPU = int(sys.argv[1])
			except:
				N_CPU = cpu_count() #get the number of CPU cores
			PLATFORM = sys.platform #get the patform on which this script is running

			#NN defines the slice ranges for the particle array.
			#We want to split the particles array in N_CPU-1 parts, i.e.  the number of
			#feasible subprocesses on this machine.
			NN = int(Nparticles / (N_CPU - 1))
			#If the platform is 'win32' we will use pipes.  The parent connector will be
			#stored in the connections list.
			if PLATFORM == 'win32':
				connections = []

			processes = [] #array with process instances

			#create a multiprocessing array for the force on each particle in shared
			#memory
			mp_Forces = Array(c_double, 2 * Nparticles)
			#create a 2D numpy array sharing its memory location with the
			#multiprocessing
			#array
			Forces = np.frombuffer(mp_Forces.get_obj(), dtype=c_double).reshape((Nparticles, 2))

			#spawn the processes
			for i in range(N_CPU - 1):
				#ensure that the last particle is also included when Nparticles / (N_CPU -
				#1) is not an integer
				if i == N_CPU - 2:
					if PLATFORM == 'win32':
						parent_conn, child_conn = Pipe() #create a duplex Pipe
						p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:], Mtot, r0), kwargs=dict(θ=θ, conn=child_conn)) #spawn process
						p.start() #start process
						parent_conn.send(Forces[i * NN:]) #send Forces array through Pipe
						connections.append(parent_conn)
					else:
						p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:], Mtot, r0), kwargs=dict(Forces=Forces[i * NN:], θ=θ)) #spawn process
						p.start() #start process
				else:
					if PLATFORM == 'win32':
						parent_conn, child_conn = Pipe() #create a duplex Pipe
						p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:(i + 1) * NN], Mtot, r0), kwargs=dict(θ=θ, conn=child_conn)) #spawn process
						p.start() #start process
						parent_conn.send(Forces[i * NN:(i + 1) * NN]) #send Forces array through Pipe
						connections.append(parent_conn)
					else:
						p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:(i + 1) * NN], Mtot, r0), kwargs=dict(Forces=Forces[i * NN:(i + 1) * NN], θ=θ)) #spawn process
						p.start() #start process
				processes.append(p)

			#if platform is 'win32' => receive filled Forces arrays through Pipe
			if PLATFORM == 'win32':
				Forces = np.concatenate([connection_receiveAndClose(conn) for conn in connections], axis=0)

			#join and terminate all processes
			processes_joinAndTerminate(processes)

			r,v = particles2arr(particles)
			if k == 0:
				v12 = v + dt / 2 * Forces
				r = r + dt * v12
			else:
				v = v12 + dt / 2 * Forces

			particles = updateparticles(r, v, particles) #added this here!

		if frame % 2 == 0:
			SDR.append(r)
			SDV.append(v)
			

	outfile = os.path.dirname(os.path.abspath(__file__)) + "/Data.npz"
	np.savez(outfile,r=np.array(SDR, dtype=object))
	AnimateOrbit(outfile, len(SDR))