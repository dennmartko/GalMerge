import numpy as np
import os
import sys

#plotting imports
from matplotlib.pyplot import figure, style, show
import matplotlib.colors as colors
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from tqdm import tqdm
import gc

#debug imports
import matplotlib.pyplot as plt

def custom_cmap():
	#trigonometric white -> red colormap
	x = np.linspace(0, 1, 256)
	color_list = np.array([np.ones(x.size), 1-x, 1-x, np.ones(x.size)]).T
	cmp = colors.ListedColormap(color_list)
	return cmp

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

#source: https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
def axisEqual3D(ax):
	extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
	sz = extents[:,1] - extents[:,0]
	centers = np.mean(extents, axis=1)
	maxsize = max(abs(sz))
	r = maxsize/2
	ax.auto_scale_xyz(*np.column_stack((centers - r, centers + r)))


def AnimateOrbit(path, frames, filename="animation", fps=20, sleep=200, window=((-25, 65), (-25, 65), (-15, 30)), axes_off=False, c_mode='off', selection=None, debug=False, verbose=False):
	style.use('dark_background')

	#if the animations folder doesn't exist in path create it!
	if not os.path.isdir(path + "/animations"):
		os.mkdir(path + "/animations")

	#if we want to save a selected subset of frames and 'animations/frames' doesn't exist in path create it!
	if selection is not None and not os.path.isdir(path + "/animations/frames"):
		os.mkdir(path + "/animations/frames")
	
	def Frame(i):
		del ax.collections[:]
		gc.collect()
		if cdata is not None:
			stars = ax.scatter3D(*rdata[i].T, s=0.4, c=cmags[i], cmap=cmap, norm=norm)
		else:
			stars = ax.scatter3D(*rdata[i].T, s=0.4, color='white')
		txts[0].set_text(r"$t = {:.2f}$ Gyr".format(t[i]))
		txts[1].set_text(r"$dt = {}$".format(properties['dt']) + "\n" + r"$\theta = {}$".format(properties['θ']) + "\n" + "Number of Cells: {} \nBodies inside frame: {}".format(properties['NCinFrame'][i],properties['NPinFrame'][i]))
		
		if __name__ == "main" and debug:
			debugfile = "debug_log.txt"
			debugpath = os.path.dirname(os.path.abspath(__file__)) + '/' + "logs"
			debugmsg(os.path.join(debugpath, debugfile), message, write_mode='a', verbose=verbose, writer=None)
		elif verbose:
			print(f"frame {i}")

		#save the selected frames
		if selection is not None:
			if t[i] in selection:
				if c_mode != 'off':
					fig.savefig(path + "/animations/frames/" + filename + f"_{c_mode}_{t[i]}.pdf", dpi=fig.dpi)
				else:
					fig.savefig(path + "/animations/frames/" + filename + f"_{t[i]}.pdf", dpi=fig.dpi)

		return (stars, *txts)

	with np.load(path + "/Data.npz",allow_pickle=True) as f:
		rdata = f["r"]
		cdata = f[f"{c_mode}"] if c_mode != 'off' else None #load color data
		
	rdata = rdata.astype(float)
	if cdata is not None:
		cmap = custom_cmap()

		cdata = cdata.astype(float)

		#find normalization for colormap
		cmags = np.linalg.norm(cdata, axis=2)
		cmin = 0
		cmax = 2 * np.mean(cmags.flatten())
		norm = colors.Normalize(vmin=cmin, vmax=cmax)

	properties = np.load(path + "/Properties.npz",allow_pickle=True)

	t = np.arange(0, properties['dt']*frames, properties['dt'])

	fig = figure(figsize=(10,10))
	ax = fig.add_subplot(111 , projection ='3d')

	ax.set(xlim=window[0], ylim=window[1], zlim=window[2])
	axisEqual3D(ax)

	if axes_off:
		remove_axes(ax)
	else:
		ax.xaxis.pane.fill = False
		ax.yaxis.pane.fill = False
		ax.zaxis.pane.fill = False
	
		ax.set_xlabel(r"$x$ [kpc]", fontsize=15, labelpad=30)
		ax.set_ylabel(r"$y$ [kpc]", fontsize=15, labelpad=30)
		ax.set_zlabel(r"$z$ [kpc]", fontsize=15, labelpad=30)

	stars = ax.scatter3D(*rdata[0].T, s=0.4, color='white')

	txts = []
	txts += [ax.text2D(0.75, 1, r"$t = {:.2f}$ Gyr".format(t[0]), fontsize=16, transform=ax.transAxes)]
	txts += [ax.text2D(0, 1, r"$dt = {}$".format(properties['dt']) + "\n" + r"$\theta = {}$".format(properties['θ']) + "\n" + "Number of Cells: {} \nBodies inside frame: {}".format(properties['NCinFrame'][0],properties['NPinFrame'][0]),
				  ha="left", va="center", transform=ax.transAxes, bbox=dict(boxstyle="round", fc="none", ec="w", pad=0.5))]

	ani = animation.FuncAnimation(fig, Frame, interval=sleep, frames=frames)
	
	if c_mode != 'off':
		outfile = path + "/animations/" + filename + f"_{c_mode}.mp4"
	else:
		outfile = path + "/animations/" + filename + ".mp4"
	ani.save(outfile, fps = fps, writer='ffmpeg', dpi=fig.dpi)


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

def AnimateCells(path, frames, filename="animationCells", fps=30, sleep=200, window=((-25, 65), (-25, 65), (-15, 30)), dpi=300, selection=None, debug=False, verbose=False):
	style.use('dark_background')

	#if the animations folder doesn't exist in path create it!
	if not os.path.isdir(path + "/animations"):
		os.mkdir(path + "/animations")

	#if we want to save a selected subset of frames and 'animations/frames' doesn't exist in path create it!
	if selection is not None and not os.path.isdir(path + "/animations/frames"):
		os.mkdir(path + "/animations/frames")

	def Frame(i):
		del ax.collections[:]
		gc.collect()
		for k, cell in tqdm(enumerate(Cdata[i])):
			#if cell.L > 1 and cell.L < 2:
			#	if k % 4 == 0:
			plot_linear_cube(ax, cell.midR, cell.L, color=(.224, 1, .078 , 1))
		np.delete(Cdata, i, axis=0) #pop the list item
		txt.set_text(r"$t = {:.2f}$ Gyr".format(t[i]))

		if __name__ == "main" and debug:
			debugfile = "debug_log.txt"
			debugpath = os.path.dirname(os.path.abspath(__file__)) + '/' + "logs"
			debugmsg(os.path.join(debugpath, debugfile), message, write_mode='a', verbose=verbose, writer=tqdm.write)
		elif verbose:
			tqdm.write(f"frame {i}")

		#save the selected frames
		#if selection is not None:
		#	if t[i] in selection:
		#		fig.savefig(path + "/animations/frames/" + filename + f"_{t[i]}.pdf", dpi=fig.dpi)
		#		sys.exit()

		return txt,

	with np.load(path + "/Cells.npz", allow_pickle=True) as f:
		Cdata = f["cells"]
	
	properties = np.load(path + "/Properties.npz",allow_pickle=True)
	t = np.arange(0, properties["dt"] * frames, properties["dt"])
	del properties

	fig = figure(figsize=(10,10))
	ax = fig.add_subplot(111 , projection ='3d')
	
	ax.set(xlim=window[0], ylim=window[1], zlim=window[2])
	axisEqual3D(ax)

	remove_axes(ax)

	txt = ax.text2D(x=0.75, y=1, s=r"$t = {:.2f}$ Gyr".format(0), fontsize=16, transform=ax.transAxes)

	ani = animation.FuncAnimation(fig, Frame, interval=sleep, frames=frames)
	
	outfile = path + "/animations/" + filename + ".mp4"
	ani.save(outfile, fps = fps, writer='ffmpeg', dpi=dpi)


if __name__ == "__main__":
	paths = [os.path.dirname(os.path.abspath(__file__)) + "/testdata/run6/", os.path.dirname(os.path.abspath(__file__)) + "/testdata/run7/", os.path.dirname(os.path.abspath(__file__)) + "/testdata/run8/"]
	
	#make particle animations
	#selection = [0, 2, 4, 6, 8, 9.99] #selection of frames we want to store separately
	#c_modes = ["off", "v", "F"]
	#for path in paths:
	#	print(f"{path}")
	#	for c_mode in c_modes:
	#		print(f"{c_mode}")
	#		AnimateOrbit(path, 1000, verbose=True, c_mode=c_mode, selection=selection, axes_off=True)

	#make cell animations
	#selection = [0, 4.99] #selection of frames we want to store separately
	#for path in paths:
	#	print(f"{path}")
	#	AnimateCells(path, 100, selection=selection, verbose=True)

	#NORMA CODE!!!
	#make cell animations
	#paths = ["/net/virgo01/data/users/lourens/run6/", "/net/virgo01/data/users/lourens/run7/", "/net/virgo01/data/users/lourens/run8/"]
	#selection = [0, 4.99] #selection of frames we want to store separately
	#for path in paths:
	#	print(f"{path}")
	#	AnimateCells(path, 100, selection=selection, verbose=True)
	#	break