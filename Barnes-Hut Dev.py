import numpy as np
import time

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show
from numba import jit,njit

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

@jit(nopython=True)
def Optimizer1(num,x,y,m,M):
	return (num + np.array([x * m,y * m]),M + m)

def Optimizer2(rdd1,rdd2,rdd3,rdd4,particles1,particles2,particles3,particles4):
	# np.delete() does not work with empty lists
	if len(rdd1) != 0:
		particles1 = np.delete(particles1, rdd1, axis=0)

	if len(rdd2) != 0:
		particles2 = np.delete(particles2, rdd2, axis=0)

	if len(rdd3) != 0:
		particles3 = np.delete(particles3, rdd3, axis=0)

	if len(rdd4) != 0:
		particles4 = np.delete(particles4, rdd4, axis=0)

	return particles1,particles2,particles3,particles4 

@jit(nopython=True)
def Optimizer3(midR,L,order):
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

jit(nopython=True)
def Optimizer4():
	# Temporary memory where we store numerator of R_CM
	num1 = np.array([0,0], dtype=np.float64)
	num2 = np.array([0,0], dtype=np.float64)
	num3 = np.array([0,0], dtype=np.float64)
	num4 = np.array([0,0], dtype=np.float64)

	# Total mass of each cell
	M1 = M2 = M3 = M4 = 0

	return num1,num2,num3,num4,M1,M2,M3,M4

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

	num1,num2,num3,num4,M1,M2,M3,M4 = Optimizer4()

	node.daughters = []

	# Init
	pcount = 0
	# Check if more than 1 particles inside square
	for indx, p in enumerate(particles):
		x, y = p.r
		m = p.m
		if (node.midR + node.L / 2)[0] > x > node.midR[0] and (node.midR + node.L / 2)[1] > y > node.midR[1]:
			pcount += 1
			rdd2app(indx)
			rdd3app(indx)
			rdd4app(indx)

			num1, M1 = Optimizer1(num1,x,y,m,M1)
		elif (node.midR - node.L / 2)[0] < x < node.midR[0] and (node.midR + node.L / 2)[1] > y > node.midR[1]:
			pcount += 1
			rdd1app(indx)
			rdd3app(indx)
			rdd4app(indx)

			num2, M2 = Optimizer1(num2,x,y,m,M2)
		elif (node.midR - node.L / 2)[0] < x < node.midR[0] and (node.midR - node.L / 2)[1] < y < node.midR[1]:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd4app(indx)

			num3, M3 = Optimizer1(num3,x,y,m,M3)
		elif (node.midR + node.L / 2)[0] > x > node.midR[0] and (node.midR - node.L / 2)[1] < y < node.midR[1]:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd3app(indx)

			num4, M4 = Optimizer1(num4,x,y,m,M4)

	# If theres more than one particle in a node, we can create new nodes!
	if pcount > 1:
		# np.delete() does not work with empty lists
		particles1,particles2,particles3,particles4 = Optimizer2(np.array(rdd1),np.array(rdd2),np.array(rdd3),np.array(rdd4),particles1,particles2,particles3,particles4)

		# if a potential cell's mass is nonzero create it!
		if M1 != 0:
			newmidR, newL = Optimizer3(node.midR,node.L,1)
			D1 = Cell(newmidR, node.L / 2, parent=node, M = M1, R_CM = num1 / M1)
			node.daughters.append(D1)
			Tree(D1, particles1)
		if M2 != 0:
			newmidR, newL = Optimizer3(node.midR,node.L,2)
			D2 = Cell(newmidR, newL, parent=node, M = M2, R_CM = num2 / M2)
			node.daughters.append(D2)
			Tree(D2, particles2)
		if M3 != 0:
			newmidR, newL = Optimizer3(node.midR,node.L,3)
			D3 = Cell(newmidR, newL, parent=node, M = M3, R_CM = num3 / M3)
			node.daughters.append(D3)
			Tree(D3, particles3)
		if M4 != 0:
			newmidR, newL = Optimizer3(node.midR,node.L,4)
			D4 = Cell(newmidR, newL, parent=node, M = M4, R_CM = num4 / M4)
			node.daughters.append(D4)
			Tree(D4, particles4)


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

		start = time.time()
		Tree(ROOT, particles)
		end = time.time()

		duration = end - start
		print("\nTOTAL AMOUNT OF CELLS: ",len(obj))
		print("TOTAL TIME TAKEN FOR",len(particles), " PARTICLES IS: ",duration, "SECONDS!")

		time_arr.append(duration)

	print("mean time taken: ",np.mean(duration), "s")