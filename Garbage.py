import numpy as np
import time

import sys
from multiprocessing import Process, Pipe, cpu_count, Array
from ctypes import c_double

#imports from own modules
import constants as const
from BH_utils.OFuncs import Tree_template_init, CM_Handler, NewCellGeom, get_condr, GForce, BHF_handler
from BH_utils.PFuncs import CellPlotter
from GalGen import generate


#Prototype of a cell object
class Cell:
	__slots__ = ('midR', 'L', 'parent', 'M', 'R_CM', 'daughters')
	def __init__(self, midR, L, parent=None, M=None, R_CM=None):
		self.parent = parent #parent of the current cell
		self.daughters = None #daughters of the current cell
		
		#Physical quantities
		self.M = M #total mass contained within the cell
		self.R_CM = R_CM #location of the center of mass of the cell

		#Geometrical quantities
		self.midR = midR #coordinate location of the cell's center
		self.L = L #length of the cell's sides


#Prototype of a particle object
class Particle:
	__slots__ = ('r', 'v', 'm')
	def __init__(self, r, v, m=None):
		# Position, velocity and mass
		self.r = r
		self.v = v

		if m is None:
			self.m = 1 #give the particle the mass of the Sun if m is not provided
		else:
			self.m = m


def rmParticles(rdd1, rdd2, rdd3, rdd4, particles1, particles2, particles3, particles4):
	# np.delete() does not work with empty lists
	if len(rdd1) != 0:
		particles1 = np.delete(particles1, rdd1, axis=0)

	if len(rdd2) != 0:
		particles2 = np.delete(particles2, rdd2, axis=0)

	if len(rdd3) != 0:
		particles3 = np.delete(particles3, rdd3, axis=0)

	if len(rdd4) != 0:
		particles4 = np.delete(particles4, rdd4, axis=0)

	return particles1, particles2, particles3, particles4 


def Tree(node, particles):
	obj.append(node) # append the created node

	# Hard copy the particle array
	particles1 = particles.copy()
	particles2 = particles.copy()
	particles3 = particles.copy()
	particles4 = particles.copy()

	# Redundant particles for each quadrant (the number in the variable name
	# refers to the ith quadrant)
	rdd1 = []
	rdd1app = rdd1.append
	rdd2 = []
	rdd2app = rdd2.append
	rdd3 = []
	rdd3app = rdd3.append
	rdd4 = []
	rdd4app = rdd4.append

	num1, num2, num3, num4, M1, M2, M3, M4 = Tree_template_init()

	node.daughters = []

	# Initialize particle counter
	pcount = 0

	# Check if more than 1 particles inside square
	for indx, p in enumerate(particles):
		r = p.r
		m = p.m
		condr = get_condr(r, node.L, node.midR) #condition r
		if 1 > condr[0] > 0 and 1 > condr[1] > 0:
			pcount += 1
			rdd2app(indx)
			rdd3app(indx)
			rdd4app(indx)

			num1, M1 = CM_Handler(num1,r,m,M1)
		elif -1 < condr[0] < 0 and 1 > condr[1] > 0:
			pcount += 1
			rdd1app(indx)
			rdd3app(indx)
			rdd4app(indx)

			num2, M2 = CM_Handler(num2,r,m,M2)
		elif -1 < condr[0] < 0 and -1 < condr[1] < 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd4app(indx)

			num3, M3 = CM_Handler(num3,r,m,M3)
		elif 1 > condr[0] > 0 and -1 < condr[1] < 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd3app(indx)

			num4, M4 = CM_Handler(num4,r,m,M4)

	# If theres more than one particle in a node, we can create new nodes!
	if pcount > 1:
		#remove redundant particles from particles arrays
		particles1, particles2, particles3, particles4 = rmParticles(np.array(rdd1), np.array(rdd2), np.array(rdd3), np.array(rdd4), particles1, particles2, particles3, particles4)

		# if a potential cell's mass is nonzero create it!
		if M1 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 1)
			D1 = Cell(newmidR, node.L / 2, parent=node, M = M1, R_CM = num1 / M1)
			node.daughters.append(D1)
			Tree(D1, particles1)
		if M2 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 2)
			D2 = Cell(newmidR, newL, parent=node, M = M2, R_CM = num2 / M2)
			node.daughters.append(D2)
			Tree(D2, particles2)
		if M3 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 3)
			D3 = Cell(newmidR, newL, parent=node, M = M3, R_CM = num3 / M3)
			node.daughters.append(D3)
			Tree(D3, particles3)
		if M4 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 4)
			D4 = Cell(newmidR, newL, parent=node, M = M4, R_CM = num4 / M4)
			node.daughters.append(D4)
			Tree(D4, particles4)


# Functions for computing the gravitational force on a single particle
def BHF(node, rp, force_arr, θ):
	daughters = node.daughters
	#print(daughters)

	if BHF_handler(rp, node.R_CM, node.L,θ):
		force_arr.append(GForce(node.M, rp, node.R_CM))
	else:
		for i in range(len(daughters)):
			BHF(daughters[i], rp, force_arr, θ)

def BHF_kickstart(ROOT, particles, Forces=None, θ=0.5, conn=None):
	#Forces will be None if the platform is 'win32'. In that case we should receive Forces through a duplex Pipe.
	if Forces is None and conn is not None:
		Forces = conn.recv() #waits until there's something to receive

	for i, p in enumerate(particles):
		force_arr = []
		BHF(ROOT, p.r, force_arr, θ)
		Fg = np.sum(np.array(force_arr) * p.m, axis=0)
		Forces[i,:] = Fg.astype(dtype=c_double)

	#send the updated Forces array back through the Pipe
	if conn is not None:
		conn.send(Forces)
		conn.close()

#receive something via a connection and close the connection
def connection_receiveAndClose(conn):
	tmp = conn.recv()
	conn.close()
	return tmp

#function to join and terminate the processes
def processes_joinAndTerminate(processes):
	#join processes
	for p in processes:
		p.join()

	#terminate processes
	for p in processes:
		p.terminate()



if __name__ == "__main__":
	time_arr1 = []
	time_arr2 = []

	for i in range(1):
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

		obj = []
		L = 2*np.linalg.norm(r[-1])

		# compute the location of the Center of Mass (COM) and total mass for the
		# ROOT cell
		Rgal_CM = np.sum([p.m * p.r for p in particles]) / np.sum([p.m for p in particles])
		Mgal = np.sum([p.m for p in particles])

		# initialize ROOT cell
		ROOT = Cell(np.array([0, 0]), L, parent=None, M=Mgal, R_CM=Rgal_CM)

		#BUILD TREE
		start = time.time()
		Tree(ROOT, particles)
		end = time.time()

		print("\nTOTAL AMOUNT OF CELLS: ",len(obj))

		duration = end - start
		time_arr1.append(duration)
		print("TOTAL TREE BUILDING TIME TAKEN FOR ",len(particles), " PARTICLES IS: ",duration, " SECONDS!")

		
		################################################
		##    COMPUTE FORCES USING MULTIPROCESSING    ##
		################################################
		N_CPU = cpu_count() #get the number of CPU cores
		PLATFORM = sys.platform #get the patform on which this script is running

		#NN defines the slice ranges for the particle array.
		#We want to split the particles array in N_CPU-1 parts, i.e. the number of feasible subprocesses on this machine.
		NN = int(Nparticles / (N_CPU - 1))

		start = time.time()

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

		end = time.time()
		duration = end - start
		time_arr2.append(duration)
		print(f"TOTAL TIME TAKEN FOR COMPUTING THE FORCES: {duration} SECONDS!")

		#PLOT CELLS
		CellPlotter(obj, particles)
	
	print("mean time taken for tree building: ",np.mean(time_arr1[1:]), "s")
	print("mean time taken for force calculation: ",np.mean(time_arr2[1:]), "s")