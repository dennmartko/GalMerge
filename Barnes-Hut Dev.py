import numpy as np
import time
import matplotlib

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show

#Prototype of a cell object
class Cell:
    def __init__(self,midR,L,parent=None, M=None, R_CM=None):
        self.parent = parent #parent of the current cell
        
        #Physical quantities
        self.M = M #total mass contained within the cell
        self.R_CM = R_CM #location of the center of mass of the cell

        #Geometrical quantities
        self.midR = midR #coordinate location of the cell's center
        self.L = L #length of the cell's sides

class Particle:
    def __init__(self,x,y,vx,vy):
        # Position and velocity
        self.x = x
        self.y = y

        self.vx = vx
        self.vy = vy

        # Physical constants
        self.m = 1.9891 * 10 ** (30)
        self.G = 6.67408 * 10 ** (-11) # m3 kg-1 s-2

# Create a Tree = 1/4
def Tree(node, particles):
    obj.append(node) # append the created node

    # Hard copy
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

    # Temporary memory where we store nominator of R_CM calculation
    nom1 = np.array([0,0], dtype=np.float64)
    nom2 = np.array([0,0], dtype=np.float64)
    nom3 = np.array([0,0], dtype=np.float64)
    nom4 = np.array([0,0], dtype=np.float64)

    # Total mass of each cell
    M1 = M2 = M3 = M4 = 0


    # Init
    pcount = 0

    # Check if more than 1 particles inside square
    for indx, p in enumerate(particles):
        x = p.x
        y = p.y
        m = p.m
        if (node.midR + node.L / 2)[0] > x > node.midR[0] and (node.midR + node.L / 2)[1] > y > node.midR[1]:
            pcount += 1
            rdd2.append(indx)
            rdd3.append(indx)
            rdd4.append(indx)

            nom1[0]+=x * m
            nom1[1]+=y * m
            M1+=m
        elif (node.midR - node.L / 2)[0] < x < node.midR[0] and (node.midR + node.L / 2)[1] > y > node.midR[1]:
            pcount += 1
            rdd1.append(indx)
            rdd3.append(indx)
            rdd4.append(indx)

            nom2[0]+=x * m
            nom2[1]+=y * m
            M2+=m
        elif (node.midR - node.L / 2)[0] < x < node.midR[0] and (node.midR - node.L / 2)[1] < y < node.midR[1]:
            pcount += 1
            rdd1.append(indx)
            rdd2.append(indx)
            rdd4.append(indx)

            nom3[0]+=x * m
            nom3[1]+=y * m
            M3+=m
        elif (node.midR + node.L / 2)[0] > x > node.midR[0] and (node.midR - node.L / 2)[1] < y < node.midR[1]:
            pcount += 1
            rdd1.append(indx)
            rdd2.append(indx)
            rdd3.append(indx)

            nom4[0]+=x * m
            nom4[1]+=y * m
            M4+=m

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

        if M1 != 0:
            Tree(Cell(node.midR + np.array([node.L / 4, node.L / 4]), node.L / 2, parent=node, M = M1, R_CM = nom1/M1), particles1)
        if M2 != 0:
            Tree(Cell(node.midR + np.array([-node.L / 4, node.L / 4]), node.L / 2, parent=node, M = M2, R_CM = nom2/M2), particles2)
        if M3 != 0:
            Tree(Cell(node.midR + np.array([-node.L / 4, -node.L / 4]), node.L / 2, parent=node, M = M3, R_CM = nom3/M3), particles3)
        if M4 != 0:
            Tree(Cell(node.midR + np.array([node.L / 4, -node.L / 4]), node.L / 2, parent=node, M = M4, R_CM = nom4/M4), particles4)


        '''
        # CREATE THE NODES!  and assign the correct particles to the nodes
        Tree(Cell(node.midR + np.array([node.L / 4, node.L / 4]), node.L / 2, parent=node, M = M1, R_CM = nom1/M1), particles1)
        Tree(Cell(node.midR + np.array([-node.L / 4, node.L / 4]), node.L / 2, parent=node, M = M2, R_CM = nom2/M2), particles2)
        Tree(Cell(node.midR + np.array([-node.L / 4, -node.L / 4]), node.L / 2, parent=node, M = M3, R_CM = nom3/M3), particles3)
        Tree(Cell(node.midR + np.array([node.L / 4, -node.L / 4]), node.L / 2, parent=node, M = M4, R_CM = nom4/M4), particles4)
        '''


def CellPlotter(cells, particles):
    rectStyle = dict(fill=False, ec='k', lw=2)
    scatterStyle = dict(color='k', s=2)

    fig = figure(figsize=(10, 10))
    frame = fig.add_subplot(111)
    frame.set_xlim(-10, 10)
    frame.set_ylim(-10, 10)
    frame.scatter([p.x for p in particles], [p.y for p in particles], **scatterStyle)

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

    particles = [Particle(x[i],y[i],vx[i],vy[i]) for i in range(0,Nparticles)]

    obj = []
    L = 20

    ROOT = Cell(np.array([0,0]),L,parent=None)
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
    #
    #print("\nPROOF THAT THE TREE IS SORTED: ",lengths)
    #CellPlotter(obj, particles)
