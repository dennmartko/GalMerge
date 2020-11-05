import numpy as np
import time
import os
import sys
import constants as const

from multiprocessing import Process, Pipe, cpu_count, Array
from ctypes import c_double
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

#own imports
from utils import argv_parser, check_and_generate_fname, debugmsg
from BH import Particle, Cell, Tree, BHF_kickstart, connection_receiveAndClose, processes_joinAndTerminate
from ODEInt import leapfrog
from Animator import AnimateOrbit
from GalGen import generate, rotate, GeneratorPlot


class BlackHole:
	__slots__ = ('r', 'v', 'ε', 'm')
	def __init__(self, r, v, ε, m):
		# Position, velocity and mass
		self.r = r
		self.v = v
		self.m = m

		#softening parameter for force calculation
		self.ε = ε
		
		

def setup_Galaxies(galaxy):
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

	particles = []
	for i in ["Bulge", "Disk"]:
		if galaxy[i][1] != 0:
			r, v = generate(galaxy[i][1], galaxy["globals"]["M0"], galaxy[i][2], ζ=galaxy[i][3], type_=galaxy[i][4])
			if galaxy["globals"]["θ"] is not None:
				#rotate r and v
				r = rotate(rotate(rotate(r, galaxy["globals"]["θ"][0], axis='x'), galaxy["globals"]["θ"][1], axis='y'), galaxy["globals"]["θ"][2], axis='z')
				v = rotate(rotate(rotate(v, galaxy["globals"]["θ"][0], axis='x'), galaxy["globals"]["θ"][1], axis='y'), galaxy["globals"]["θ"][2], axis='z')

			m = np.full(galaxy[i][1], galaxy[i][0] * galaxy["globals"]['M0'] / galaxy[i][1]).tolist()
			particles += [Particle(r[j] + galaxy["globals"]["R0"], v[j] + galaxy["globals"]["Vsys"], m=m[j]) for j in range(galaxy[i][1])]
		
	#generate dark matter particles
	r, v = galaxy["DM"][2]
	if galaxy["globals"]["θ"] is not None:
		#rotate r and v
		r = rotate(rotate(rotate(r, galaxy["globals"]["θ"][0], axis='x'), galaxy["globals"]["θ"][1], axis='y'), galaxy["globals"]["θ"][2], axis='z')
		v = rotate(rotate(rotate(v, galaxy["globals"]["θ"][0], axis='x'), galaxy["globals"]["θ"][1], axis='y'), galaxy["globals"]["θ"][2], axis='z')

	m = np.full(galaxy["DM"][1], galaxy["DM"][0] * galaxy["globals"]["M0"] / galaxy["DM"][1])
	particles += [Particle(r[j] + galaxy["globals"]["R0"], v[j] + galaxy["globals"]["Vsys"], m=m[j]) for j in range(galaxy["DM"][1])]

	SMBH = [BlackHole(galaxy["globals"]["R0"], galaxy["globals"]["Vsys"], 15 , galaxy["SMBH"][0] * galaxy["globals"]["M0"] / galaxy["SMBH"][1])]

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

def GetSituation(R, colors, window=None):
	plt.style.use("dark_background")
	
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111, projection="3d")
	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False

	for i, r in enumerate(R):
		ax.scatter(*r, color=colors[i], s=0.4)
	
	ax.grid()

	if window is not None:
		lim = window
	else:
		Max = np.max([np.max(R[:,i]) for i in range(R.shape[1])])
		Min = np.min([np.min(R[:,i]) for i in range(R.shape[1])])
		lim = (Min, Max)

	ax.set(xlim=lim, ylim=lim, zlim=lim, xlabel=r"$x$", ylabel=r"$y$", zlabel=r"$z$")

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
	#parse command line arguments
	N_CPU, fname, outpath, debug, verbose = argv_parser()

	#create logfile for debugging purposes
	if debug:
		debugfile = "debug_log.txt"
		debugpath = os.path.dirname(os.path.abspath(__file__)) + '/' + "logs"
			
		#IF the "logs" folder doesn't exist create it!
		if not os.path.isdir(debugpath):
				os.mkdir(debugpath)
		else:
			#if the a file with name of debugfile already exists generate a new name!
			debugfile = check_and_generate_fname(debugfile, path=debugpath)

		#write a debug message
		debugmsg(os.path.join(debugpath, debugfile), "#####    START    #####", write_mode='w', verbose=verbose)

	######################
	#

	#location and velocity of MACHO's
	DM_r = np.array([[10,0,0],[0,10,0],[-10,0,0],[0,-10,0]])
	DM_v = np.array([[0,30,0],[-30,0,0],[0,-30,0],[30,0,0]])

	#syntax: "component" : (mass fraction, N, r0, ζ, model) / "globals" defines
	#the global variables corresponding to the galaxy: "M0" = total mass; "R0" =
	#location; "Vsys" = systemic velocity; "θ" = rotation angles around (x, y, z)
	#respectively
	Gal1 = { "Bulge" : (0.125, 900, 3.5, 1, "plummer"),
			 "Disk": (0.375, 3000, [5, 20], 1, "disk"),
			 "DM": (0.02, len(DM_r), [DM_r, DM_v], None, None),
			 "SMBH": (0.48, 1, None, None, None),
			 "globals" : {"M0" : 2 * 10 ** 8, "R0" : np.array([0, 0, 0]), "Vsys" : np.array([5, 5, 0]), "θ" : (0, 0, 0)}
			}

	Gal2 = { "Bulge" : (0.125, 750, 3.5, 1, "plummer"),
			 "Disk": (0.375, 1500, [4, 11], 1, "disk"),
			 "DM": (0.02, len(DM_r), [DM_r, DM_v], None, None),
			 "SMBH": (0.48, 1, None, None, None),
			 "globals" : {"M0" : 8 * 10 ** 7, "R0" : np.array([50, 25, 25]), "Vsys" : np.array([-10, -5, -5]), "θ" : (np.pi/4, -np.pi/4, 0)}
			}
	
	#Runtime variables
	frames = 5 #2000
	θ = 0.75
	dt = 0.005
	L = 300

	#
	######################
	
	particles = []
	SMBHS = []
	
	if debug: debugmsg(os.path.join(debugpath, debugfile), "Setting up Galaxies...", write_mode='a', verbose=verbose) #write debug message
	for Gal in [Gal1,Gal2]:
		setup_out = setup_Galaxies(Gal)
		particles += setup_out[0]
		SMBHS += setup_out[1]

	Nparticles = len(particles)

	r, v = particles2arr(particles)
	SDV = [v] # Storage of Data for V
	SDR = [r] # Storage of Data for R
	SDF = [] # Storage of Data for F
	SDC = [] # Storage of Cell objects
	
	#some properties
	Ncells_in_frame = []
	Np_in_frame = []

	#GeneratorPlot(r, type_="spatial", histograms=True, outpath=".")
	#GeneratorPlot(v, type_="velocity", outpath=".")
	#sys.exit()

	if debug: debugmsg(os.path.join(debugpath, debugfile), "Starting frame iteration...", write_mode='a', verbose=verbose) #write debug message
	for frame in tqdm(range(frames)):
		
		Np_in_frame += [sum([1 if (abs(rr[0]) < L / 2 and abs(rr[1]) < L / 2 and abs(rr[2]) < L / 2) else 0 for rr in r])]
		
		# debugging code:
		if debug and verbose:
			colors = ['r' if i == 10 else 'w' for i in range(Nparticles)]
			GetSituation(r,colors, window=(-25, 65))

		if debug and frame == 0:
			debugmsg(os.path.join(debugpath, debugfile), "frame : {}, Np : {}, T : {} s".format(frame, Np_in_frame[frame], 0), write_mode='a', verbose=verbose, writer=tqdm.write) #write debug message
			t_start = time.time()

		if debug and frame % 1 == 0 and frame != 0:
			t_end = time.time()
			r, v = particles2arr(particles)
			debugmsg(os.path.join(debugpath, debugfile), "frame : {}, Np : {}, T : {} s".format(frame, Np_in_frame[frame], t_end - t_start), write_mode='a', verbose=verbose, writer=tqdm.write) #write debug message
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
		Ncells_in_frame += [len(obj)]
		
		#store cell geometry
		if frame < 500:
			SDC.append(obj)
		#delete list of Cell objects (Cells can still be recovered by recursively stepping through the cells from ROOT)
		del obj

				
		################################################
		##    COMPUTE FORCES USING MULTIPROCESSING ##
		################################################
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

		####################
		##    LEAPFROG    ##
		####################
		#storage
		if frame % 1 == 0 and frame != 0:
			SDR.append(r) #x_{i+1}
			#resync v and store
			SDV.append(v + Forces * dt / 2) #v_{i+1} = v_{i+1/2} + a_{i+1}*Δt/2
			#store Forces
			SDF.append(Forces)
		elif frame == 0:
			#store Forces
			SDF.append(Forces)

		for i in SMBHS:
			#compute the force on supermassive black hole 'i'
			Fg = np.zeros(3)
			for j in SMBHS:
				if i != j:
					R = i.r - j.r
					Fg -= (const.G_ * j.m) * R / (np.linalg.norm(R) ** 2 + j.ε ** 2) ** (3 / 2)
			
			if frame == 0:
				#kickstart leapfrog for the black holes by moving v half a step forward
				i.v = i.v + Fg * dt / 2 #v_{i+1/2} = v_{i} + a_{i}*Δt/2
				i.r = i.r + i.v * dt #x_{i+1} = x_{i} + v_{i+1/2}*Δt
			else:
				#update location and velocity corresponding to the SMBHS
				# v : v_{i+3/2} = v_{i+1/2} + a_{i+1}*Δt
				# r : x_{i+2} = x_{i+1} + v_{i+3/2}*Δt
				i.r, i.v = leapfrog(i.r, Fg, i.v, dt)


		if frame == 0:
			#kickstart v by moving it half a timestep forward
			v = v + Forces * dt / 2 #v_{i+1/2} = v_{i} + a_{i}*Δt/2
			r = r + v * dt #x_{i+1} = x_{i} + v_{i+1/2}*Δt
		else:
			# v : v_{i+3/2} = v_{i+1/2} + a_{i+1}*Δt
			# r : x_{i+2} = x_{i+1} + v_{i+3/2}*Δt
			r, v = leapfrog(r, Forces, v, dt)
				

		particles = updateparticles(r, v, particles)

	
	#save file with properties of the run
	if debug: debugmsg(os.path.join(debugpath, debugfile), "Saving properties...", write_mode='a', verbose=verbose) #write debug message
	propertiesfile = outpath + "/Properties.npz"
	np.savez(propertiesfile, θ=θ, dt = dt , NPinFrame=np.array(Np_in_frame, dtype=object), NCinFrame=np.array(Ncells_in_frame, dtype=object))
	#delete large lists
	del Np_in_frame
	del Ncells_in_frame
	
	#save file with r data
	if debug: debugmsg(os.path.join(debugpath, debugfile), "Saving particle data...", write_mode='a', verbose=verbose) #write debug message
	outfile = outpath + "/Data.npz"
	np.savez(outfile, r=np.array(SDR, dtype=object), v=np.array(SDV, dtype=object), F=np.array(SDF, dtype=object))

	#save file with Cell objects for each frame
	if debug: debugmsg(os.path.join(debugpath, debugfile), "Saving cells...", write_mode='a', verbose=verbose) #write debug message
	cellfile = outpath + "/Cells.npz"
	np.savez(cellfile, cells=np.array(SDC, dtype=object))
	#delete large dict
	del SDC
	
	if debug: debugmsg(os.path.join(debugpath, debugfile), "Producing animation...", write_mode='a', verbose=verbose) #write debug message
	AnimateOrbit(outpath, len(SDR), filename=fname, window=(-25, 65), debug=debug, verbose=verbose)

	if debug: debugmsg(os.path.join(debugpath, debugfile), "#####    END    #####", write_mode='a', verbose=verbose) #write debug message