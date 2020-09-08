import numpy as np
from matplotlib.pyplot import figure, show

if __name__ == "__main__":
    x, y = (20*np.random.random(size=10)-10, 20*np.random.random(size=10)-10)
    
    fig = figure(figsize=(10, 10))
    frame = fig.add_subplot(1,1,1)
    frame.scatter(x, y, color='k')
    frame.grid()
    frame.set_xlim(-10, 10)
    frame.set_ylim(-10, 10)
    frame.set_xlabel(r"$x$", fontsize=16)
    frame.set_ylabel(r"$y$", fontsize=16)
    show()