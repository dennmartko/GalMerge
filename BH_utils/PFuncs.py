from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.pyplot import figure, show

def CellPlotter(cells, particles):
	rectStyle = dict(fill=False, ec='lightgrey', lw=2, zorder=1)
	scatterStyle = dict(color='k', s=2, zorder=2)

	fig = figure(figsize=(10, 10))
	frame = fig.add_subplot(111)
	frame.set_xlim(-10, 10)
	frame.set_ylim(-10, 10)
	frame.scatter([p.r[0] for p in particles], [p.r[1] for p in particles], **scatterStyle)

	for o in cells:
		rect = patches.Rectangle((o.midR[0] - o.L / 2,o.midR[1] - o.L / 2), width=o.L, height=o.L, **rectStyle)
		frame.add_patch(rect)

	frame.set_xlabel(r"$x$", fontsize=16)
	frame.set_ylabel(r"$y$", fontsize=16)
	show()