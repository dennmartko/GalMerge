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

def Tree(root, particles):
    obj.append(root)

    if root.parent == None: 
        Tree(Cell(root.midR + np.array([-root.L / 2,root.L / 2]),root.L / 2,parent=root),particles)
        Tree(Cell(root.midR + np.array([root.L / 2,root.L / 2]),root.L / 2,parent=root),particles)
        Tree(Cell(root.midR + np.array([root.L / 2,-root.L / 2]),root.L / 2,parent=root),particles)
        Tree(Cell(root.midR + np.array([-root.L / 2,-root.L / 2]),root.L / 2,parent=root),particles)

    # Init
    pcount = 0

    # Check if more than 1 particles inside square
    for p in particles:
        if (root.midR - root.L)[0] < p[0] < root.midR[0] and (root.midR + root.L)[1] > p[1] > root.midR[1]:
            pcount += 1
        elif root.midR[0] < p[0] < (root.midR + root.L)[0] and (root.midR + root.L)[1] > p[1] > root.midR[1]:
            pcount += 1
        elif root.midR[0] < p[0] < (root.midR + root.L)[0] and (root.midR - root.L)[1] > p[1] > root.midR[1]:
            pcount += 1
        elif (root.midR - root.L)[0] < p[0] < root.midR[0] and (root.midR - root.L)[1] > p[1] > root.midR[1]:
            pcount += 1


        if pcount > 1:
            Tree(Cell(root.midR + np.array([-root.L / 2,root.L / 2]),root.L / 2,parent=root),particles)
            Tree(Cell(root.midR + np.array([root.L / 2,root.L / 2]),root.L / 2,parent=root),particles)
            Tree(Cell(root.midR + np.array([root.L / 2,-root.L / 2]),root.L / 2,parent=root),particles)
            Tree(Cell(root.midR + np.array([-root.L / 2,-root.L / 2]),root.L / 2,parent=root),particles)
            break;






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
    #x, y = (20*np.random.random(size=10)-10, 20*np.random.random(size=10)-10)

    ROOT = Cell(np.array([0,0]),10,parent=None)
    Tree(ROOT, particles)

    print(len(obj))
    
    # Test print all lengths of objects
    for o in obj:
        print(o.L)
