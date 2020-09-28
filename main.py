import numpy as np
import time

import sys
from multiprocessing import Process, Pipe, cpu_count, Array
from ctypes import c_double
from tqdm import tqdm

#own imports
from BH import Particle, Cell, Tree, BHF_kickstart, connection_receiveAndClose, processes_joinAndTerminate
from ODEInt import leapfrog

def particles2arr(particles):
	r = np.array([p.r for p in particles])
	v = np.array([p.v for p in particles])
	return r,v

def updateparticles(r,v, particles):
	for indx,p in enumerate(particles):
		p.r = r[indx]
		p.v = v[indx]
	return particles

if __name__ == "__main__":
	time_arr1 = []
	time_arr2 = []

	Nparticles = 100000
	
	x = 20 * np.random.random(size=Nparticles) - 10
	y = 20 * np.random.random(size=Nparticles) - 10
	vx = 200 * np.random.random(size=Nparticles)
	vy = 200 * np.random.random(size=Nparticles)

	r = np.array([x, y])
	v = np.array([vx, vy])

	particles = [Particle(r[:,i], v[:,i]) for i in range(Nparticles)]

	obj = []
	L = 20
	frames = 20
	for frame in tqdm(range(frames)):
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

		SDV = np.empty((0,2))
		SDR = np.empty((0,2))

		r,v = particles2arr(particles)

		if frame == 0:
			r, v, SDR, SDV = leapfrog(r, Forces, v, SDV,SDR, dt=0.001, init=True)
		else:
			if frame % 5 == 0:
				r, v, SDR, SDV = leapfrog(r, Forces, v, SDV, SDR, dt=0.001, store='yes')
			else:
				r, v, SDR, SDV = leapfrog(r, Forces, v, SDV, SDR, dt=0.001)

		particles = updateparticles(r,v, particles)
		print(r)
	print(SDR, SDV)