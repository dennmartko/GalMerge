import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
#matplotlib.use('Agg')
import gc
from matplotlib.pyplot import figure, show, style
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tqdm import tqdm

style.use('dark_background')


def remove_axes(ax, hidelabels=True):
    #remove panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    #remove ticks and ticklabels
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_zaxis().set_ticks([])

    #remove spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    #remove grid
    ax.grid(False)

    #remove labels
    if hidelabels:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    

def plot_linear_cube(ax, midR, L, color=(.224, 1, .078 , 1)):
    x, y, z = midR - L / 2
    dx, dy, dz = [L] * 3

    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    kwargs = {'alpha': 1, 'color': color, 'linewidth' : 0.2}
    ax.plot3D(xx, yy, [z]*5, **kwargs)
    ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
    return ax

def AnimateOrbit(file,frames):
    
    def Frame(i):
        del ax.collections[:]
        del ax.texts[:]
        stars = ax.scatter(xdata[i][:,0].astype(float), xdata[i][:,1].astype(float), xdata[i][:,2].astype(float), s=0.4, color='white')
        ax.text (20 ,20 ,40 ,"t={:.2f}Gyr".format(t[i]))
        print(i)
        return (stars)


    with np.load(file,allow_pickle=True) as f:
        xdata = f["r"]

    xdata.astype(float)
    t = np.arange(0,0.005*frames,0.005)
    length = len(xdata[0][:,0])
    fig = figure(figsize =(10,10))
    ax = fig.add_subplot(111 , projection ='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.set_xlim((-30,30))
    ax.set_ylim((-30,30))
    ax.set_zlim((-30,30))
    ax.set_xlabel(r"$x$ [kpc]", fontsize=15, labelpad=30)
    ax.set_ylabel(r"$y$ [kpc]", fontsize=15, labelpad=30)
    ax.set_zlabel(r"$z$ [kpc]", fontsize=15, labelpad=30)

    ani = animation.FuncAnimation(fig, Frame, interval=200, frames=frames)
    outfile = os.path.dirname(os.path.abspath(__file__)) + "/animations/animationTEST.mp4"
    ani.save(outfile,fps = 30, writer='ffmpeg',dpi =300)

def AnimateCells(file, frames):
    def Frame(i):
        ax = fig.add_subplot(111 , projection ='3d')
        remove_axes(ax)
        lim = (-15, 15)
        ax.set(xlim=lim, ylim=lim, zlim=lim)
        #gc.collect()
        del ax.collections[:]
        del ax.texts[:]
        k = 0
        for cell in tqdm(Cdata[i]):
            if cell.L > 1 and cell.L < 2:
                k += 1
                if k%2 == 0:
                    ax = plot_linear_cube(ax, cell.midR, cell.L, color=(.224, 1, .078 , 1))
        ax.text (20 ,20 ,40 ,"t={:.2f}Gyr".format(t[i]))
        return (ax)

    with np.load(file, allow_pickle=True) as f:
        Cdata = f["cells"]
    
        fig = figure(figsize =(10,10))
    t = np.arange(0,0.005*frames,0.005)

    ani = animation.FuncAnimation(fig, Frame, interval=200, frames=frames)
    outfile = os.path.dirname(os.path.abspath(__file__)) + "/animations/animationCellsTEST.mp4"
    ani.save(outfile,fps = 30, writer='ffmpeg',dpi =300)


if __name__ == "__main__":
    #outfile = os.path.dirname(os.path.abspath(__file__)) + "/Data.npz"
    #AnimateOrbit(outfile, 100)

    outfile = os.path.dirname(os.path.abspath(__file__)) + "/Cells.npz"
    AnimateCells(outfile, 100)

    #with np.load(outfile,allow_pickle=True) as f:
    #    Cdata = f["cells"]

    #for frame in range(Cdata.shape[0]):
    #    fig = figure(figsize=(10,10))
    #    ax = fig.add_subplot(111, projection='3d')
    #    remove_axes(ax)
    #    limits = (-15, 15)
    #    ax.set(xlim=limits, ylim=limits, zlim=limits)
    #    for cell in tqdm(Cdata[frame]):
    #        if cell.daughters == [] and cell.L > .25 and cell.L < 2:
    #            plot_linear_cube(ax, cell.midR, cell.L, color=(.224, 1, .078 , 1))
    #    plt.show()