import numpy as np
import scipy.spatial as sp
import networkx as nx
import os
import multiprocessing
from multiprocessing import Pool

# Pads the points from a snapshot periodically, adding duplicates of particles from one end of the box to the other.
def periodicPadding(snapshot,width,boxsize):
	points = snapshot['pos']
	indices = snapshot['iord']
	plowx = np.where((points[:,0] > 0.0) & (points[:,0] <= width))
	puppx = np.where((points[:,0] > boxsize - width) & (points[:,0] <= boxsize))
	points_new = np.vstack((points,points[plowx] + np.array([boxsize,0,0]),points[puppx] - np.array([boxsize,0,0])))
	indices = np.hstack((indices,indices[plowx[0]],indices[puppx[0]]))
	plowx = np.where((points_new[:,1] > 0.0) & (points_new[:,1] <= width))
	puppx = np.where((points_new[:,1] > boxsize - width) & (points_new[:,1] <= boxsize))
	points_new = np.vstack((points_new,points_new[plowx] + np.array([0,boxsize,0]),points_new[puppx] - np.array([0,boxsize,0])))
	indices = np.hstack((indices,indices[plowx[0]],indices[puppx[0]]))
	plowx = np.where((points_new[:,2] > 0.0) & (points_new[:,2] <= width))
	puppx = np.where((points_new[:,2] > boxsize - width) & (points_new[:,2] <= boxsize))
	points_new = np.vstack((points_new,points_new[plowx] + np.array([0,0,boxsize]),points_new[puppx] - np.array([0,0,boxsize])))
	indices = np.hstack((indices,indices[plowx[0]],indices[puppx[0]]))
	return [points_new,indices]

# Construct a voronoi diagram from a snapshot:
def voronoiOfSnap(snapshot,widthFactor=0.025):
	boxsize = snapshot.properties['boxsize'].ratio("Mpc a h**-1")
	# Voronoi code can't handle periodic BCs, so we need to extend the box with points from the boundaries padded around the box of the snapshot:
	width = widthFactor*boxsize
	[points_new,indices] = periodicPadding(snapshot,width,boxsize)
	# Construct voronoi diagram:
	vor = sp.Voronoi(points_new)
	# Get volumes and convex hulls:
	chList = []
	volumes = np.zeros(len(snapshot))
	for k in range(0,len(snapshot)):
		ch = sp.ConvexHull(vor.vertices[vor.regions[vor.point_region[k]]])
		chList.append(ch)
		volumes[k] = ch.volume
	# Get density, and density contrast:
	mass = np.zeros(len(snapshot))
	mass[:] = snapshot['mass'].in_units("Msol h**-1")[:]
	rhobar = np.sum(mass)/(boxsize**3)
	rho = mass/volumes
	delta = (rho - rhobar)/rhobar
	return [vor,chList,volumes,rho,rhobar,delta,points_new,indices]

def createSubVoronoi(points,subBoxSize,boxsize,offset,widthFactor = 0.1):
	overlap = widthFactor*subBoxSize
	subPoints = np.where((points[:,0] >= offset[0]) & (points[0] <= offset[0] + subBoxSize) & (points[:,1] >= offset[1]) & (points[1] <= offset[1] + subBoxSize) & (points[:,2] >= offset[2]) & (points[2] <= offset[2] + subBoxSize))
	bufferPoints = np.where((points[:,0] >= offset[0] - overlap) & (points[0] <= offset[0] + subBoxSize) & (points[:,1] >= offset[1]) & (points[1] <= offset[1] + subBoxSize) & (points[:,2] >= offset[2]) & (points[2] <= offset[2] + subBoxSize))

# Parallel voronoi:
def parallelVoronoiOfSnap(snap,widthFactor=0.1,divisions = 2):
	boxsize = snapshot.properties['boxsize'].ratio("Mpc a h**-1")
	width = widthFactor*boxsize
	[points_new,indices] = periodicPadding(snapshot,width,boxsize)
	subBoxWidth = boxsize/divisions

# Extract information about the connectedness of points with density below a given threshold:
def densityThresh(vor,indices,delta,deltaThresh,returnIntermediaries = False):
	# We want to identify adjacent voronoi regions, however, the padding with periodic particles means that there are duplicates. We want to identify which relationships between particles are actually duplicates of each other so that particles are mapped to the correct neighbours.
	ridges = np.zeros(vor.ridge_points.shape,dtype=int)
	ridges[:,:] = vor.ridge_points[:,:]
	# Use the indices list to map points back to their periodic duplicates:
	ridges[:,0] = indices[ridges[:,0]]
	ridges[:,1] = indices[ridges[:,1]]
	# Now we have the ridges between adjacent particles, we want to extract only particles with densities below the specified threshold:
	belowThresh = np.where((delta[ridges[:,0]] < deltaThresh) & (delta[ridges[:,1]] < deltaThresh))
	ridges_below = ridges[belowThresh]
	# Construct a graph of these particles, combining points which are adjacent voronoi regions:
	G = nx.Graph()
	G.add_nodes_from(indices[np.where(delta < deltaThresh)])
	G.add_edges_from(ridges_below)
	# Get a list of the connected regions:
	regionList = list(nx.connected_components(G))
	lengthList = np.zeros(len(regionList),dtype=int)
	for k in range(0,len(lengthList)):
		lengthList[k] = len(regionList[k])
	if returnIntermediaries:
		return [lengthList,regionList,G,ridges,ridges_below]
	else:
		return [lengthList,regionList]
	
# Use DTFE to get the density field
def dtfeDensity(snapname,grid=256,recompute=False):
	field = "density_a"
	if not recompute:
		recompute = not os.path.isfile(snapname + "_dtfe.a_den")
	if recompute:
		os.system("DTFE " + snapname + " " + snapname + "_dtfe" + " --grid " + str(grid) + " --periodic --field " + field)
	rho = np.reshape(np.fromfile(snapname + "_dtfe.a_den",dtype='float32'),(grid,grid,grid))
	rhobar = np.mean(rho)
	delta = (rho - rhobar)/rhobar
	return delta

def gridPositionsLinear(delta,boxsize):
	length = delta.size
	pos = np.zeros((delta.shape[0]*delta.shape[1]*delta.shape[2],3))
	xRange = np.linspace( boxsize/(2*delta.shape[0]),200.0 - boxsize/(2*delta.shape[0]),delta.shape[0])
	yRange = np.linspace( boxsize/(2*delta.shape[1]),200.0 - boxsize/(2*delta.shape[1]),delta.shape[1])
	zRange = np.linspace( boxsize/(2*delta.shape[2]),200.0 - boxsize/(2*delta.shape[2]),delta.shape[2])
	X, Y, Z = np.meshgrid(xRange,yRange,zRange)
	Xr = np.reshape(X,length)
	Yr = np.reshape(Y,length)
	Zr = np.reshape(Z,length)
	pos[:,0] = Xr
	pos[:,1] = Yr
	pos[:,2] = Zr
	deltaLin = np.reshape(delta,delta.shape[0]*delta.shape[1]*delta.shape[2])
	return [pos,deltaLin]

class TopHatFilter:
	def __init__(self,xs,delta,R):
		self.xs = xs
		self.delta = delta
		self.tree = sp.cKDTree(xs)
		self.R = R
	def __call__(self,x,treeSearch=False):
		result = np.zeros(len(x))
		for k in range(0,len(x)):
			points = self.tree.query_ball_point(x[k],self.R)
			result[k] = np.mean(self.delta[points])/(4*np.pi*self.R**3/3)
		return result
		








