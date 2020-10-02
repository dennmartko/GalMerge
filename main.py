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
	plt.scatter([p.r[0] for p in particles],[p.r[1] for p in particles],color=colors)
	plt.grid()
	plt.ylim(-15,15)
	plt.xlim(-15,15)
	plt.show()

if __name__ == "__main__":
	Nparticles = 1000
	
	r, v = generate(Nparticles)
	r = r[:,:2]
	v = v[:,:2]
	#x = 20 * (2*np.random.random(size=Nparticles) - 1)
	#y = 20 * (2*np.random.random(size=Nparticles) - 1)
	#vx = 20 * (2*np.random.random(size=Nparticles) - 1)
	#vy = 20 * (2*np.random.random(size=Nparticles) - 1)

	#r = np.array([x, y])
	#v = np.array([vx, vy])

	particles = [Particle(r[i], v[i]) for i in range(Nparticles)] #:,i
	colors = ['orange' if i== 10 else 'b' for i in range(Nparticles)]

	L = 40
	frames = 10

	SDV = [v]
	SDR = [r]
	for frame in tqdm(range(frames)):
		#GetSituation(particles,colors)
		# compute the location of the Center of Mass (COM) and total mass for the
		# ROOT cell
		Rgal_CM = np.sum([p.m * p.r for p in particles]) / np.sum([p.m for p in particles])
		Mgal = np.sum([p.m for p in particles])

		# initialize ROOT cell
		ROOT = Cell(np.array([0, 0]), L, parent=None, M=Mgal, R_CM=Rgal_CM)

		#BUILD TREE
		Tree(ROOT, particles)
		
		################################################
		##    COMPUTE FORCES USING MULTIPROCESSING    ##
		################################################
		N_CPU = cpu_count() #get the number of CPU cores
		PLATFORM = sys.platform #get the patform on which this script is running

		#NN defines the slice ranges for the particle array.
		#We want to split the particles array in N_CPU-1 parts, i.e. the number of feasible subprocesses on this machine.
		NN = int(Nparticles / (N_CPU - 1))

		#If the platform is 'win32' we will use pipes. The parent connector will be stored in the connections list.
		if PLATFORM == 'win32':
			connections = []
		processes = [] #array with process instances

		#create a multiprocessing array for the force on each particle in shared memory
		mp_Forces = Array(c_double, 2*Nparticles)
		#create a 2D numpy array sharing its memory location with the multiprocessing array
		Forces = np.frombuffer(mp_Forces.get_obj(), dtype=c_double).reshape((Nparticles, 2))

		#spawn the processes
		for i in range(N_CPU - 1):
			#ensure that the last particle is also included when Nparticles / (N_CPU - 1) is not an integer
			if i == N_CPU - 2:
				if PLATFORM == 'win32':
					parent_conn, child_conn = Pipe() #create a duplex Pipe
					p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:]), kwargs=dict(θ=0.5, conn=child_conn)) #spawn process
					p.start() #start process
					parent_conn.send(Forces[i*NN:]) #send Forces array through Pipe
					connections.append(parent_conn)
				else:
					p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:]), kwargs=dict(Forces=Forces[i*NN:], θ=0.5)) #spawn process
					p.start() #start process
			else:
				if PLATFORM == 'win32':
					parent_conn, child_conn = Pipe() #create a duplex Pipe
					p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:(i + 1) * NN]), kwargs=dict(θ=0.5, conn=child_conn)) #spawn process
					p.start() #start process
					parent_conn.send(Forces[i * NN:(i + 1) * NN]) #send Forces array through Pipe
					connections.append(parent_conn)
				else:
					p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:(i + 1) * NN]), kwargs=dict(Forces=Forces[i * NN:(i + 1) * NN], θ=0.5)) #spawn process
					p.start() #start process
			processes.append(p)

		#if platform is 'win32' => receive filled Forces arrays through Pipe
		if PLATFORM == 'win32':
			Forces = np.concatenate([connection_receiveAndClose(conn) for conn in connections], axis=0)

		#join and terminate all processes
		processes_joinAndTerminate(processes)

		r,v = particles2arr(particles)

		if frame == 0:
			r, v, dummy = leapfrog(r, Forces, v, dt=0.001, init=True)
		else:
			if frame % 1 == 0:
				r, v, vstore = leapfrog(r, Forces, v, dt=0.001)
				SDR.append(r)
				SDV.append(vstore)
			else:
				r, v, vstore = leapfrog(r,Forces, v, dt=0.001)

		particles = updateparticles(r,v, particles)

	print(SDR[1], SDV[1])
	print(len(SDV))
	outfile = os.path.dirname(os.path.abspath(__file__)) + "/Data.npz"
	np.savez(outfile,r=np.array(SDR, dtype=object))
	AnimateOrbit(outfile, len(SDR))