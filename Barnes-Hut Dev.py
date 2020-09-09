import numpy as np
import time
from matplotlib.pyplot import figure, show

#Prototype of a cell object
class Cell:
    def __init__(self,midR,L,parent=None,):
        self.parent = parent #parent of the current cell
        
        #Physical quantities
        self.M = None #total mass contained within the cell
        self.R_CM = None #location of the center of mass of the cell

        #Geometrical quantities
        self.midR = midR #coordinate location of the cell's center
        self.L = L #length of the cell's sides

# Create a Tree = 1/4
def Tree(root, particles):
    obj.append(root) # append the created node

    # Hard copy
    particles1 = particles.copy()
    particles2 = particles.copy()
    particles3 = particles.copy()
    particles4 = particles.copy()

    # Redundant particles for each quadrant
    rdd1 = []
    rdd2 = []
    rdd3 = []
    rdd4 = []

    # When we start we just create 4 nodes straight away.
    if root.parent == None: 
        Tree(Cell(root.midR + np.array([-root.L / 2,root.L / 2]),root.L / 2,parent=root),particles)
        Tree(Cell(root.midR + np.array([root.L / 2,root.L / 2]),root.L / 2,parent=root),particles)
        Tree(Cell(root.midR + np.array([root.L / 2,-root.L / 2]),root.L / 2,parent=root),particles)
        Tree(Cell(root.midR + np.array([-root.L / 2,-root.L / 2]),root.L / 2,parent=root),particles)

    # Init
    pcount = 0

    # Check if more than 1 particles inside square
    for indx, p in enumerate(particles):
        if (root.midR - root.L)[0] < p[0] < root.midR[0] and (root.midR + root.L)[1] > p[1] > root.midR[1]:
            pcount += 1
            rdd2.append(indx)
            rdd3.append(indx)
            rdd4.append(indx)
        elif root.midR[0] < p[0] < (root.midR + root.L)[0] and (root.midR + root.L)[1] > p[1] > root.midR[1]:
            pcount += 1
            rdd1.append(indx)
            rdd3.append(indx)
            rdd4.append(indx)
        elif root.midR[0] < p[0] < (root.midR + root.L)[0] and (root.midR - root.L)[1] > p[1] > root.midR[1]:
            pcount += 1
            rdd1.append(indx)
            rdd2.append(indx)
            rdd4.append(indx)
        elif (root.midR - root.L)[0] < p[0] < root.midR[0] and (root.midR - root.L)[1] > p[1] > root.midR[1]:
            pcount += 1
            rdd1.append(indx)
            rdd2.append(indx)
            rdd3.append(indx)

    # If theres more than one particle in a node, we can create new nodes!
    if pcount > 1:
        print(rdd1) # redundant particles in the new node of quadrant 1
        print(len(particles1)) # Status amount of particles left in quadrant 1

        # np.delete() does not work with empty lists

        if len(rdd1) != 0:
            particles1 = np.delete(particles1, rdd1,axis=0)

        if len(rdd2) != 0:
            particles2 = np.delete(particles2, rdd2,axis=0)

        if len(rdd3) != 0:
            particles3 = np.delete(particles3, rdd3,axis=0)

        if len(rdd4) != 0:
            particles4 = np.delete(particles4, rdd4,axis=0)

        print(len(particles1)) # check how much particles are left after removing redundant particles

        # CREATE THE NODES! and assign the correct particles to the nodes
        Tree(Cell(root.midR + np.array([-root.L / 2,root.L / 2]),root.L / 2,parent=root),particles1)
        Tree(Cell(root.midR + np.array([root.L / 2,root.L / 2]),root.L / 2,parent=root),particles2)
        Tree(Cell(root.midR + np.array([root.L / 2,-root.L / 2]),root.L / 2,parent=root),particles3)
        Tree(Cell(root.midR + np.array([-root.L / 2,-root.L / 2]),root.L / 2,parent=root),particles4)


if __name__ == "__main__":
    particles = [(20 * np.random.random() - 10, 20 * np.random.random() - 10) for i in range(100)]

    fig = figure(figsize=(10, 10))
    frame = fig.add_subplot(1,1,1)
    frame.scatter([particles[i][0] for i in range(100)],[particles[i][1] for i in range(100)], color='k')
    frame.set_xlim(-10, 10)
    frame.set_ylim(-10, 10)
    frame.set_xlabel(r"$x$", fontsize=16)
    frame.set_ylabel(r"$y$", fontsize=16)
    show()


    obj = []

    ROOT = Cell(np.array([0,0]),10,parent=None)
    Tree(ROOT, particles)

    time.sleep(1) # print fix
    print("\nTOTAL AMOUNT OF NODES: ",len(obj))
    
    # Test print all lengths of objects
    lengths = [o.L for o in obj]
    print("MINIMUM LENGTH IS: ",np.min(lengths))

