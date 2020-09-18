import numpy as np
import time

from numba import jit,njit
from multiprocessing import Process, Manager, Queue
#imports from own modules
import constants as const
from BH_utils.OFuncs import Tree_template_init, CM_Handler, NewCellGeom, get_condr


#Prototype of a cell object
class Cell:
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
	def __init__(self, r, v, m=None):
		# Position, velocity and mass
		self.r = r
		self.v = v

		if m is None:
			self.m = 2 #const.Msol #give the particle the mass of the Sun if m is not provided
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
	rdd1 = []; rdd1app = rdd1.append;
	rdd2 = []; rdd2app = rdd2.append;
	rdd3 = []; rdd3app = rdd3.append;
	rdd4 = []; rdd4app = rdd4.append;

	num1, num2, num3, num4, M1, M2, M3, M4 = Tree_template_init()

	node.daughters = []

	# Init
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
@njit
def GForce(M, rp, Rcm):
	r = rp - Rcm
	Fg = (const.G * M)/(r[0]**2 + r[1]**2)**(3/2) * (r)
	return Fg

@njit
def BHF_handler(rp,Rcm,L,θ):
	r = rp - Rcm
	D = (r[0]**2 + r[1]**2)**(1/2)

	if D == 0:
		return False
	elif L / D <= θ:
		return True
	else:
		return False

def BHF(node,rp,θ=0.5):
	daughters = node.daughters
	
	if BHF_handler(rp,node.R_CM,node.L,θ):
		force_arr.append(GForce(node.M,rp,node.R_CM))
	else:
		for i in range(len(daughters)):
			BHF(daughters[i],rp,θ)


if __name__ == "__main__":
	time_arr1 = []
	time_arr2 = []

	for i in range(5):
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

		
		
		#COMPUTE FORCES
		start = time.time()
		'''
		for p in particles:
			GForce_handler(ROOT, p)
		'''

		for p in particles:
			force_arr = []
			BHF(ROOT, p.r, θ=0.5)
			Fg = np.array(force_arr) * p.m
		end = time.time()


		duration = end - start
		time_arr2.append(duration)
		print(f"TOTAL TIME TAKEN FOR COMPUTING THE FORCES: {duration} SECONDS!")
		
		#PLOT CELLS
		#CellPlotter(obj, particles)
	
	print("mean time taken for tree building: ",np.mean(time_arr1[1:]), "s")
	print("mean time taken for force calculation: ",np.mean(time_arr2[1:]), "s")