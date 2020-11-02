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
from GalGen import generate, rotate


class BlackHole:
	__slots__ = ('r', 'v', 'ε', 'm')
	def __init__(self, r, v, ε, m):
		# Position, velocity and mass
		self.r = r
		self.v = v
		self.m = m

		#softening parameter for force calculation
		self.ε = 15
		
		

def setup_Galaxies(gal1, gal2=None):
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
	'''

	if gal2 == None:
		particles = []
		for i in ["Bulge", "Disk"]:
			if gal1[i][1] != 0:
				r, v = generate(gal1[i][1], gal1["globals"]["M0"], gal1[i][2], ζ=gal1[i][3], type_=gal1[i][4])
				if gal1["globals"]["θ"] is not None:
					#rotate r and v
					r = rotate(rotate(rotate(r, gal1["globals"]["θ"][0], axis='x'), gal1["globals"]["θ"][1], axis='y'), gal1["globals"]["θ"][2], axis='z')
					v = rotate(rotate(rotate(v, gal1["globals"]["θ"][0], axis='x'), gal1["globals"]["θ"][1], axis='y'), gal1["globals"]["θ"][2], axis='z')

				m = np.full(gal1[i][1], gal1[i][0] * gal1["globals"]['M0'] / gal1[i][1]).tolist()
				particles += [Particle(r[j] + gal1["globals"]["R0"], v[j] + gal1["globals"]["Vsys"], m=m[j]) for j in range(gal1[i][1])]
		
		#generate dark matter particles
		r, v = gal1["DM"][2]
		if gal1["globals"]["θ"] is not None:
			#rotate r and v
			r = rotate(rotate(rotate(r, gal1["globals"]["θ"][0], axis='x'), gal1["globals"]["θ"][1], axis='y'), gal1["globals"]["θ"][2], axis='z')
			v = rotate(rotate(rotate(v, gal1["globals"]["θ"][0], axis='x'), gal1["globals"]["θ"][1], axis='y'), gal1["globals"]["θ"][2], axis='z')

		m = np.full(gal1["DM"][1], gal1["DM"][0] * gal1["globals"]["M0"] / gal1["DM"][1])
		particles += [Particle(r[j] + gal1["globals"]["R0"], v[j] + gal1["globals"]["Vsys"], m=m[j]) for j in range(gal1["DM"][1])]

		SMBH = [BlackHole(gal1["globals"]["R0"], gal1["globals"]["Vsys"], 20 , gal1["SMBH"][0] * gal1["globals"]["M0"] / gal1["SMBH"][1])]

	return particles, SMBH


def particles2arr(particles):
	r = np.array([p.r for p in particles])
	v = np.array([p.v for p in particles])
	return r, v

def updateparticles(r,v, particles):
	for indx,p in enumerate(particles):
		p.r = r[indx]
		p.v = v[indx]
	return particles

def GetSituation(r,colors):
	plt.figure(figsize=(10,10))
	plt.scatter([p.r[0] for p in particles],[p.r[1] for p in particles],color=colors,s=0.4)
	plt.grid()
	plt.ylim(-25,25)
	plt.xlim(-25,25)
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
	######################
	#

	DM_r = np.array([[10,0,0],[0,10,0],[-10,0,0],[0,-10,0]])
	DM_v = np.array([[0,30,0],[-30,0,0],[0,-30,0],[30,0,0]])

	#syntax: "component" : (mass fraction, N, r0, ζ, model) / "globals" defines
	#the global variables corresponding to the galaxy: "M0" = total mass; "R0" =
	#location; "Vsys" = systemic velocity; "θ" = rotation angles around (x, y, z)
	#respectively
	Gal1 = { "Bulge" : (0.125, 800, 3.5, 1, "plummer"),
			 "Disk": (0.375, 4000, [5, 20], 1, "disk"),
			 "DM": (0.02, len(DM_r), [DM_r, DM_v], None, None),
			 "SMBH": (0.48, 1, None, None, None),
			 "globals" : {"M0" : 2 * 10 ** 8, "R0" : np.array([0, 0, 0]), "Vsys" : np.array([0, 0, 5]), "θ" : (np.pi / 4, np.pi / 4, np.pi / 4)}
			}

	#Runtime variables
	frames = 100 #600
	θ = 0.8
	dt = 0.005
	L = 300

	#
	######################

	particles, SMBH = setup_Galaxies(Gal1)
	Nparticles = len(particles)

	SMBHS = SMBH

	colors = ['orange' if i == 10 else 'b' for i in range(Nparticles)]

	r, v = particles2arr(particles)
	SDV = [v] # Storage of Data for V
	SDR = [r] # Storage of Data for R
	SDC = []
	for frame in tqdm(range(frames)):
		# debugger code:
		if frame == 0:
			try:
				debug = str(sys.argv[2]) == "--debug"
				if not os.isdir(os.path.dirname(os.path.abspath(__file__)) + '/' + "logs"):
					os.mkdir(os.path.dirname(os.path.abspath(__file__)) + '/' + "logs")
				debugfile = os.path.dirname(os.path.abspath(__file__)) + '/' + "logs/debug_log.txt"
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
		obj = []
		Tree(ROOT, particles, obj)
		SDC.append(obj)

				
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
			
	cellfile = os.path.dirname(os.path.abspath(__file__)) + "/Cells.npz"
	np.savez(cellfile, cells=np.array(SDC, dtype=object))
	outfile = os.path.dirname(os.path.abspath(__file__)) + "/Data.npz"
	np.savez(outfile,r=np.array(SDR, dtype=object))
	AnimateOrbit(outfile, len(SDR))