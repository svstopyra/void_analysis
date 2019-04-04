# Python script to plot things in 3D
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# Plots a 3D scatter plot
def scatter(x,y,z):
	fig = pyplot.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(x,y,z)
	pyplot.show()
