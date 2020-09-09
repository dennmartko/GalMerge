import numpy as np
from matplotlib.pyplot import figure, show
import time

#Prototype of a cell object
class Cell:
    def __init__(self, parent, midR, L):
        self.parent = parent #parent of the current cell 
        
        #Physical quantities
        self.M = None #total mass contained within the cell
        self.R_CM = None #location of the center of mass of the cell

        #Geometrical quantities
        self.midR = midR #coordinate location of the cell's center
        self.L = L #length of the cell's sides

def inCell(r, ival):
    if r[0] > ival[0,0] and r[0] < ival[0,1] and r[1] > ival[1,0] and r[1] < ival[1,1]:
        return True
    return False

def Quadrant(r):
    #this function returns values from 1 to 4 to represent the quadrant in which r is located
    if r[0] > 0 and r[1] > 0:
        return 1
    elif r[0] < 0 and r[1] > 0:
        return 2
    elif r[0] < 0 and r[1] < 0:
        return 3
    elif r[0] > 0 and r[1] < 0:
        return 4

def build_tree(r, nodes, leaves, midR, L, parent=None, indx=0, tmp_r=[]):
    #initialize cell as a node
    nodes.append(Cell(parent, midR, L))

    ival = np.array([midR - L/2, midR + L/2]).T #interval of the cell

    N = len(tmp_r) #number of particles already in the cell

    for i in range(indx, len(r)+1):

        #if N = 2 then the cell is full -> we need to split
        if N == 2:
            #new tmp_ri where i is a number from 1 to 4 defining the quadrant the particle is in
            tmp_r1 , tmp_r2, tmp_r3, tmp_r4 = ([], [], [], [])

            #divide the particles over the quadrants
            for p in tmp_r:
                Q = Quadrant(p-midR)
                if Q == 1:
                    tmp_r1.append(p)
                elif Q == 2:
                    tmp_r2.append(p)
                elif Q == 3:
                    tmp_r3.append(p)
                elif Q == 4:
                    tmp_r4.append(p)

            #recurse!
            build_tree(r, nodes, leaves, midR + np.array([L/4, L/4]), L/2, parent=nodes[-1], indx=i, tmp_r=tmp_r1)
            build_tree(r, nodes, leaves, midR + np.array([-L/4, L/4]), L/2, parent=nodes[-1], indx=i, tmp_r=tmp_r2)
            build_tree(r, nodes, leaves, midR + np.array([-L/4, -L/4]), L/2, parent=nodes[-1], indx=i, tmp_r=tmp_r3)
            build_tree(r, nodes, leaves, midR + np.array([L/4, -L/4]), L/2, parent=nodes[-1], indx=i, tmp_r=tmp_r4)
            break
        
        #if a particle is in the cell it is added to tmp_r and N is incremented with 1
        if i != len(r):
            if inCell(r[i], ival):
                N += 1
                tmp_r.append(r[i])
    
    #if we exited the for loop and N is still 1 we have found a leaf else if N = 0 no particles are in this cell so we delete it.
    if N == 1:
        leaves.append(nodes[-1]) #the final cell was reached which is a leaf
    elif N == 0:
        del nodes[-1] #no particles in this cell wherefore it is removed

if __name__ == "__main__":
    r = np.array([20*np.random.random(size=100000)-10, 20*np.random.random(size=100000)-10]).T
    
    nodes = []
    leaves = []
    L = 20.
    midR = np.array([0., 0.])

    #build the hierarchical tree of cells
    s = time.time()
    build_tree(r, nodes, leaves, midR, L)
    print(f"Duration {time.time()-s} seconds.")
    
    #print(leaves, len(leaves))

    fig = figure(figsize=(10, 10))
    frame = fig.add_subplot(1,1,1)
    frame.scatter(r[:,0], r[:,1], color='k', s=1)

    for leaf in leaves:
        frame.axhline(y=leaf.midR[1]-leaf.L/2, xmin=(leaf.midR[0]-leaf.L/2+L/2)/L, xmax=(leaf.midR[0]+leaf.L/2+L/2)/L)
        frame.axhline(y=leaf.midR[1]+leaf.L/2, xmin=(leaf.midR[0]-leaf.L/2+L/2)/L, xmax=(leaf.midR[0]+leaf.L/2+L/2)/L)
        frame.axvline(x=leaf.midR[0]-leaf.L/2, ymin=(leaf.midR[1]-leaf.L/2+L/2)/L, ymax=(leaf.midR[1]+leaf.L/2+L/2)/L)
        frame.axvline(x=leaf.midR[0]+leaf.L/2, ymin=(leaf.midR[1]-leaf.L/2+L/2)/L, ymax=(leaf.midR[1]+leaf.L/2+L/2)/L)

    frame.grid()
    frame.set_xlim(-10, 10)
    frame.set_ylim(-10, 10)
    frame.set_xlabel(r"$x$", fontsize=16)
    frame.set_ylabel(r"$y$", fontsize=16)
    show()