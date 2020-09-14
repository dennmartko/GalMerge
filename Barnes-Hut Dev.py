import numpy as np
import time
import matplotlib

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show

#imports from own modules
import constants as const

#Prototype of a cell object
class Cell:
    def __init__(self, midR, L, parent=None, M=None, R_CM=None):
        self.parent = parent #parent of the current cell
        
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
    rdd1 = []
    rdd2 = []
    rdd3 = []
    rdd4 = []

    # Temporary memory where we store numerator of R_CM
    num1 = np.array([0,0], dtype=np.float64)
    num2 = np.array([0,0], dtype=np.float64)
    num3 = np.array([0,0], dtype=np.float64)
    num4 = np.array([0,0], dtype=np.float64)

    # Total mass of each cell
    M1 = M2 = M3 = M4 = 0


    # Init
    pcount = 0
    # Check if more than 1 particles inside square
    for indx, p in enumerate(particles):
        x, y = p.r
        m = p.m
        if (node.midR + node.L / 2)[0] > x > node.midR[0] and (node.midR + node.L / 2)[1] > y > node.midR[1]:
            pcount += 1
            rdd2.append(indx)
            rdd3.append(indx)
            rdd4.append(indx)

            num1[0] += x * m; num1[1] += y * m
            M1 += m
        elif (node.midR - node.L / 2)[0] < x < node.midR[0] and (node.midR + node.L / 2)[1] > y > node.midR[1]:
            pcount += 1
            rdd1.append(indx)
            rdd3.append(indx)
            rdd4.append(indx)

            num2[0] += x * m; num2[1] += y * m
            M2 += m
        elif (node.midR - node.L / 2)[0] < x < node.midR[0] and (node.midR - node.L / 2)[1] < y < node.midR[1]:
            pcount += 1
            rdd1.append(indx)
            rdd2.append(indx)
            rdd4.append(indx)

            num3[0] += x * m; num3[1] += y * m
            M3 += m
        elif (node.midR + node.L / 2)[0] > x > node.midR[0] and (node.midR - node.L / 2)[1] < y < node.midR[1]:
            pcount += 1
            rdd1.append(indx)
            rdd2.append(indx)
            rdd3.append(indx)

            num4[0] += x * m; num4[1] += y * m
            M4 += m

    # If theres more than one particle in a node, we can create new nodes!
    if pcount > 1:
        # np.delete() does not work with empty lists
        if len(rdd1) != 0:
            particles1 = np.delete(particles1, rdd1, axis=0)

        if len(rdd2) != 0:
            particles2 = np.delete(particles2, rdd2, axis=0)

        if len(rdd3) != 0:
            particles3 = np.delete(particles3, rdd3, axis=0)

        if len(rdd4) != 0:
            particles4 = np.delete(particles4, rdd4, axis=0)

        # if a potential cell's mass is nonzero create it!
        if M1 != 0:
            Tree(Cell(node.midR + np.array([node.L / 4, node.L / 4]), node.L / 2, parent=node, M = M1, R_CM = num1 / M1), particles1)
        if M2 != 0:
            Tree(Cell(node.midR + np.array([-node.L / 4, node.L / 4]), node.L / 2, parent=node, M = M2, R_CM = num2 / M2), particles2)
        if M3 != 0:
            Tree(Cell(node.midR + np.array([-node.L / 4, -node.L / 4]), node.L / 2, parent=node, M = M3, R_CM = num3 / M3), particles3)
        if M4 != 0:
            Tree(Cell(node.midR + np.array([node.L / 4, -node.L / 4]), node.L / 2, parent=node, M = M4, R_CM = num4 / M4), particles4)


def CellPlotter(cells, particles):
    rectStyle = dict(fill=False, ec='k', lw=2)
    scatterStyle = dict(color='k', s=2)

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

    # compute the location of the Center of Mass (COM) and total mass for the ROOT cell
    Rgal_CM = np.sum([p.m * p.r for p in particles]) / np.sum([p.m for p in particles])
    Mgal = np.sum([p.m for p in particles])

    # initialize ROOT cell
    ROOT = Cell(np.array([0, 0]), L, parent=None, M=Mgal, R_CM=Rgal_CM)

    start = time.time()
    Tree(ROOT, particles)
    end = time.time()

    print("\nTOTAL AMOUNT OF CELLS: ",len(obj))
    
    # Sort the obj array for nodes up to root
    obj.sort(key=lambda o: o.L)
    lengths = [o.L for o in obj]
    print("MINIMUM LENGTH IS: ",np.min(lengths))
    print("TOTAL TIME TAKEN FOR",len(particles), " PARTICLES IS: ",end - start, "SECONDS!")

    # TURN OFF IF SPAMMY

    #print("\nPROOF THAT THE TREE IS SORTED: ",lengths)
    #CellPlotter(obj, particles)