import numpy as np
from matplotlib.pyplot import figure, show

#Prototype of a cell object
class Cell:
    def __init__(self):
        self.parent = None #parent of the current cell 
        
        #Physical quantities
        self.M = None #total mass contained within the cell
        self.R_CM = None #location of the center of mass of the cell

        #Geometrical quantities
        self.midR = None #coordinate location of the cell's center
        self.L = None #length of the cell's sides

def build_tree(r, leaves, midR, L, indx, parent=None):
    if parent is None and indx == 0:
        indx += 1

    ival = np.array([midR - L/2, midR + L/2]).T
    
    """
        initialize cell here defining the following attributes:
        - .parent
        - .midR
        - .L
    """

    for i in range(indx, len(r)):
        if r[i,0] > ival[0,0] and r[i,0] < ival[0,1] and r[i,1] > ival[1,0] and r[i,1] < ival[1,1]:
            build_tree(r, leaves, midR + np.array([L/4, L/4]), indx+1) #parent= ADD parent object!
            build_tree(r, leaves, midR + np.array([L/4, -L/4]), indx+1) #parent= ADD parent object!
            build_tree(r, leaves, midR + np.array([-L/4, -L/4]), indx+1) #parent= ADD parent object!
            build_tree(r, leaves, midR + np.array([-L/4, L/4]), indx+1) #parent= ADD parent object!
    
    leaves.append() #append cell here!

if __name__ == "__main__":
    r = np.array([20*np.random.random(size=10)-10, 20*np.random.random(size=10)-10]).T
    leaves = []
    L = 20.
    midR = np.array([0., 0.])

    #build the hierarchical tree of cells
    build_tree(r, leaves, midR, L, 0)
    
    fig = figure(figsize=(10, 10))
    frame = fig.add_subplot(1,1,1)
    frame.scatter(r[:,0], r[:,1], color='k')
    frame.grid()
    frame.set_xlim(-10, 10)
    frame.set_ylim(-10, 10)
    frame.set_xlabel(r"$x$", fontsize=16)
    frame.set_ylabel(r"$y$", fontsize=16)
    show()