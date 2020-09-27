from turtle import *
import numpy as np

#builds a quadtree at random
def build_tree(t, linelength=100):
    if linelength > 0:
        t.forward(linelength)
        for i in range(4):
            if i==0:
                t.right(45)
            else:
                t.left(30)
            if np.random.randint(low=0, high=1+1) == 1:
                build_tree(t, linelength-15)
        t.right(45) #turn back to center
        t.backward(linelength)

if __name__ == "__main__":
    #screen layout
    scr = Screen()
    scr.setup(height=.5, width=.25, startx=0)
    scr.title("Tree Plotter using turtle")

    #initialize turtle
    t = Turtle()
    t.color("black")
    t.pen(pen=dict(pensize=1.25, resizemode="auto"))
    t.speed(speed=0)
    t.setheading(90) #make turtle head upward
    t.up() #pull pen up
    t.backward(scr.screensize()[1]/2) #move backward half the screenheight
    t.down() #put pen down

    #start building the tree
    linelength = 100
    build_tree(t, linelength=linelength-15)

    exitonclick()