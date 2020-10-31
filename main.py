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


class BlackHole:
	__slots__ = ('r', 'v', 'ε', 'm')
	def __init__(self, r, v, ε, m):
		# Position, velocity and mass
		self.r = r
		self.v = v
		self.m = m

		#softening parameter for force calculation
		self.ε = ε
		
		

def setup_Galaxy(Nparticles, Mtot, r0, R0, Vsys, Msmbh, ζ=1, type_="plummer", kind="3d"):
	'''
		Nparticles : number of stars in the Galaxy
		Mtot : total mass of stars in the Galaxy
		r0 : scaling radius of the Galaxy
		R0 : offset from origin
		Vsys : systemic velocity of the Galaxy in kpc/Gyr ~ 0.9784619419 km/s
		Msmbh : mass of the supermassive black hole (SMBH) at the center of the Galaxy
		ζ : parameter defining the rotation direction of stars in the Galaxy:
			1 : anti-clockwise rotation
			-1 : clockwise rotation
			0 : each star rotates in a random direction
			(defaults to '1', i.e. anti-clockwise)
		type_ : type of Galactic model to use (defaults to 'plummer')
		kind : defines the dimensionality of setup (either '2d' or '3d')
	'''
	r, v = generate(Nparticles, Mtot, r0, ζ=ζ, type_=type_) # generate stellar positions and velocities
	m = np.full(Nparticles, 1) #generate mass array where all stars have the same mass #(Mtot - Msmbh) / Nparticles

	# if a 2D Galaxy needs to be generated slice off one dimension from de r and v
	# arrays
	if kind == "2d":
		r = r[:,:2]
		v = v[:,:2]

		#Add systemic velocities and offset from the origin
		r += R0[:2].reshape(1, 2)
		v += Vsys[:2].reshape(1, 2)
	else:
		r += R0.reshape(1, 3)
		v += Vsys.reshape(1, 3)
		

	#add SMBH to the Galactic center
	SMBH = [BlackHole(R0, Vsys, r0, Msmbh)]

	#generate particle objects
	particles = [Particle(r[i], v[i], m=m[i]) for i in range(Nparticles)]

	return particles, r, v, SMBH


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
	plt.ylim(-50,50)
	plt.xlim(-50,50)
	plt.show()


if __name__ == "__main__":
	'''

		1. Nparticles indicates the amount of particles in the simulation recommended is an amount between 1000 and 10000 = 1k-10k
		2. The amount of frames for test runs to observe any flaws should be between 70 and 200 to obtain reasonable computing time. (within 5mins to 1h)
		3. Theta indicates the BH performance or approximation. The higher θ, the faster the algorithm is but less accurate. Recommended is: θ=[0.5,0.8]
		4. The recommended timestep dt, based on obtaining smooth orbits, is recommended to be smaller than 0.01 Gyrs. This requirement is substantiated by the crossing time of the Galaxy.
		5. The algorithm typically follows the following idea: GENERATE GALAXY + INITIAL CONDITIONS -> START COMPUTING FRAMES <--> (BARNES HUT ALGORITHM -> INTEGRATOR); --> MOVIE
		5. A total stellar mass of M = 1E9 to 1E12 is recommended.
		6. The program automatically detects the maximumum number of possible cpu cores on your computer and will maximize its usage. WINDOWS, LINUX, and MAC OS are supported.
		   If you wish to set this number manually you can provide it as an argument to the run, e.g. 'python3 main.py 2' to use two cores.
		7. The Galactic model used to generate a Galaxy can be altered. Current options are: "plummer", "jaffe" and "hernquist".
		9. r0 is the scaling radius of the Galaxy

	'''
	frames = 100  #600
	θ = 0.7
	dt = 0.005
	L = 300 * 2

	#Galaxy specific parameters
	Nparticles = 1000 #3000
	Msmbh = 10 ** 8
	Mtot = Msmbh # stars contribute 10^8Msol
	r0 = 20
	R0 = np.array([0, 0, 0])
	Vsys = np.array([10, 0, 0])

	particles, r, v, SMBH = setup_Galaxy(Nparticles, Mtot, r0, R0, Vsys, Msmbh, type_="hernquist")

	SMBHS = SMBH
	#Nparticles += #don't forget this when adding more galaxies!!!

	colors = ['orange' if i == 10 else 'b' for i in range(Nparticles)]

	SDV = [v] # Storage of Data for V
	SDR = [r] # Storage of Data for R

	for frame in tqdm(range(frames)):
		# debugger code:
		#GetSituation(r,colors)
		if frame == 0:
			try:
				debug = str(sys.argv[2]) == "--debug"
				debugfile = os.path.dirname(os.path.abspath(__file__)) + '/' + "debug_log.txt"
				with open(debugfile, 'w') as f:
					f.write("START\n")
				t_start = time.time()
				t_end = t_start
			except:
				debug = False

		if debug and frame % 1 == 0:
			t_end = time.time()
			r, v = particles2arr(particles)
			Np_in_frame = sum([1 if (abs(rr[0]) < L / 2 and abs(rr[1]) < L / 2 and abs(rr[2]) < L / 2) else 0 for rr in r])

			#GetSituation(r,colors)

			with open(debugfile, 'a') as f:
				f.write("Np : {}, T : {} s\n".format(Np_in_frame, t_end - t_start))
			t_start = time.time()



		# compute the location of the Center of Mass (COM) and total mass for the
		# ROOT cell
		Rgal_CM = np.sum([p.m * p.r for p in particles]) / np.sum([p.m for p in particles])
		Mgal = np.sum([p.m for p in particles])

		# initialize ROOT cell
		ROOT = Cell(np.array([0, 0, 0]), L, parent=None, M=Mgal, R_CM=Rgal_CM)

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
		mp_Forces = Array(c_double, 3 * Nparticles)
		#create a 3D numpy array sharing its memory location with the
		#multiprocessing
		#array
		Forces = np.frombuffer(mp_Forces.get_obj(), dtype=c_double).reshape((Nparticles, 3))

		#spawn the processes
		for i in range(N_CPU - 1):
			#ensure that the last particle is also included when Nparticles / (N_CPU -
			#1) is not an integer
			if i == N_CPU - 2:
				if PLATFORM == 'win32':
					parent_conn, child_conn = Pipe() #create a duplex Pipe
					p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:]), kwargs=dict(θ=θ, conn=child_conn, SMBHS=SMBHS)) #spawn process
					p.start() #start process
					parent_conn.send(Forces[i * NN:]) #send Forces array through Pipe
					connections.append(parent_conn)
				else:
					p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:]), kwargs=dict(Forces=Forces[i * NN:], θ=θ, SMBHS=SMBHS)) #spawn process
					p.start() #start process
			else:
				if PLATFORM == 'win32':
					parent_conn, child_conn = Pipe() #create a duplex Pipe
					p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:(i + 1) * NN]), kwargs=dict(θ=θ, conn=child_conn, SMBHS=SMBHS)) #spawn process
					p.start() #start process
					parent_conn.send(Forces[i * NN:(i + 1) * NN]) #send Forces array through Pipe
					connections.append(parent_conn)
				else:
					p = Process(target=BHF_kickstart, args=(ROOT, particles[i * NN:(i + 1) * NN]), kwargs=dict(Forces=Forces[i * NN:(i + 1) * NN], θ=θ, SMBHS=SMBHS)) #spawn process
					p.start() #start process
			processes.append(p)

		#if platform is 'win32' => receive filled Forces arrays through Pipe
		if PLATFORM == 'win32':
			Forces = np.concatenate([connection_receiveAndClose(conn) for conn in connections], axis=0)

		#join and terminate all processes
		processes_joinAndTerminate(processes)

		if frame % 1 == 0 and frame != 0:
			SDR.append(r)
			#resync v and store
			SDV.append(v + Forces * dt / 2)
		
		#integrate using leapfrog (assuming v is half a step out of sync)
		if frame == 0:
			#kickstart v by moving it half a timestep backwards
			v = v + Forces * dt / 2
			r, v = leapfrog(r, Forces, v, dt)

			#kickstart leapfrog for the black holes
			for SMBH in SMBHS:
				SMBH.v = SMBH.v + 0 * dt / 2
				SMBH.r, _ = leapfrog(SMBH.r, 0, SMBH.v, dt)
		else:
			r, v = leapfrog(r, Forces, v, dt)

			#update location and velocity corresponding to the SMBHS
			for SMBH in SMBHS:
				SMBH.r, _ = leapfrog(SMBH.r, 0, SMBH.v, dt)
				

		particles = updateparticles(r, v, particles)
			

	outfile = os.path.dirname(os.path.abspath(__file__)) + "/Data.npz"
	np.savez(outfile,r=np.array(SDR, dtype=object))
	AnimateOrbit(outfile, len(SDR))