import numpy as np
import time

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show
from numba import jit,njit
from multiprocessing import Process

#imports from own modules
import constants as const


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
			self.m = const.Msol #give the particle the mass of the Sun if m is not provided
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

@jit(nopython=True)
def CM_components():
	# Temporary memory where we store numerator of R_CM
	num1, num2, num3, num4 = np.zeros((4,2), dtype=np.float64)

	# Total mass of each cell
	M1 = M2 = M3 = M4 = 0

	return num1, num2, num3, num4, M1, M2, M3, M4

@jit(nopython=True)
def alterCM_components(num,r,m,M):
	return (num + m*r, M + m)

@jit(nopython=True)
def NewCellGeom(midR,L,order):
	if order == 1:
		newmidR = midR + np.array([L / 4, L / 4])
	if order == 2:
		newmidR = midR + np.array([-L / 4, L / 4])
	if order == 3:
		newmidR = midR + np.array([-L / 4, -L / 4])
	if order == 4:
		newmidR = midR + np.array([L / 4, -L / 4])
	
	newL = L / 2
	return newmidR, newL

@jit(nopython=True)
def get_condr(r, L, midR):
	return 2*(r-midR)/L

# Create a Tree = 1/4
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

	num1, num2, num3, num4, M1, M2, M3, M4 = CM_components()

	node.daughters = []

	# Init
	pcount = 0
	# Check if more than 1 particles inside square
	for indx, p in enumerate(particles):
		r = p.r
		m = p.m
		condr = get_condr(r, L, node.midR) #condition r
		if 1 > condr[0] > 0 and 1 > condr[1] > 0:
			pcount += 1
			rdd2app(indx)
			rdd3app(indx)
			rdd4app(indx)

			num1, M1 = alterCM_components(num1,r,m,M1)
		elif -1 < condr[0] < 0 and 1 > condr[1] > 0:
			pcount += 1
			rdd1app(indx)
			rdd3app(indx)
			rdd4app(indx)

			num2, M2 = alterCM_components(num2,r,m,M2)
		elif -1 < condr[0] < 0 and -1 < condr[1] < 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd4app(indx)

			num3, M3 = alterCM_components(num3,r,m,M3)
		elif 1 > condr[0] > 0 and -1 < condr[1] < 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd3app(indx)

			num4, M4 = alterCM_components(num4,r,m,M4)

	# If theres more than one particle in a node, we can create new nodes!
	if pcount > 1:
		#remove redundant particles from particles arrays
		particles1, particles2, particles3, particles4 = rmParticles(np.array(rdd1), np.array(rdd2), np.array(rdd3), np.array(rdd4), particles1, particles2, particles3, particles4)

		# if a potential cell's mass is nonzero create it!
		if M1 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 1)
			D1 = Cell(newmidR, node.L / 2, parent=node, M = M1, R_CM = num1 / M1)
			node.daughters.append(D1)
			if len(obj) == 0:
				p1 = Process(target=Tree, args=(D1,particles1,))
				p1.start()
			else:
				Tree(D1, particles1)
		if M2 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 2)
			D2 = Cell(newmidR, newL, parent=node, M = M2, R_CM = num2 / M2)
			node.daughters.append(D2)
			if len(obj) == 0:
				p2 = Process(target=Tree, args=(D2,particles2,))
				p2.start()
			else:
				Tree(D2, particles2)
		if M3 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 3)
			D3 = Cell(newmidR, newL, parent=node, M = M3, R_CM = num3 / M3)
			node.daughters.append(D3)
			if len(obj) == 0:
				p3 = Process(target=Tree, args=(D3,particles3,))
				p1.start()
			else:
				Tree(D3, particles3)
		if M4 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 4)
			D4 = Cell(newmidR, newL, parent=node, M = M4, R_CM = num4 / M4)
			node.daughters.append(D4)
			Tree(D4, particles4)

		if len(obj) == 1:
			p1.join();p2.start();p3.start();

def CellPlotter(cells, particles):
	rectStyle = dict(fill=False, ec='lightgrey', lw=2, zorder=1)
	scatterStyle = dict(color='k', s=2, zorder=2)

	fig = figure(figsize=(10, 10))
	frame = fig.add_subplot(111)
	frame.set_xlim(-10, 10)
	frame.set_ylim(-10, 10)
	frame.scatter([p.r[0] for p in particles], [p.r[1] for p in particles], **scatterStyle)

	for o in cells:
		rect = matplotlib.patches.Rectangle((o.midR[0] - o.L / 2,o.midR[1] - o.L / 2), width=o.L, height=o.L, **rectStyle)
		frame.add_patch(rect)

	frame.set_xlabel(r"$x$", fontsize=16)
	frame.set_ylabel(r"$y$", fontsize=16)
	show()


if __name__ == "__main__":
	time_arr = []

	for i in range(10):
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

		start = time.time()
		Tree(ROOT, particles)
		end = time.time()

		duration = end - start
		print("\nTOTAL AMOUNT OF CELLS: ",len(obj))
		print("TOTAL TIME TAKEN FOR",len(particles), " PARTICLES IS: ",duration, "SECONDS!")

		time_arr.append(duration)

	print("mean time taken: ",np.mean(duration), "s")