#
from mayavi.mlab import points3d, text3d, plot3d, triangular_mesh
from mayavi import mlab
import pynbody
from void_analysis import context, snapedit, plot_utilities
import numpy as np
import imageio
import os
import gc
import matplotlib.pylab as plt
from matplotlib import cm
from . import cosmology
from scipy import integrate
import pandas
import seaborn as sns
from .plot_utilities import binCentres, binValues

# Plot the positions in a snapshot:
def subsnap_scatter(subsnap,color_spec=(1,1,1),scale=1.0,type='2dvertex'):
	r = subsnap['pos']
	points3d(r[:,0],r[:,1],r[:,2],mode=type,color=color_spec,scale_factor=scale)

# Plot a set of specified points (wrapper for points3d):
def point_scatter(r,color_spec=(1,1,1),scale=1.0,type='2dvertex'):
	if len(r.shape) == 2:
		points3d(r[:,0],r[:,1],r[:,2],mode=type,color=color_spec,scale_factor=scale)
	else:
		points3d(r[0],r[1],r[2],mode=type,color=color_spec,scale_factor=scale)

# Returns a subsnap contains the particles within halos only that satisfy the specified filter
#def halo_filter(halo_list,filt):
	

# Plot the surroundings of the specified halo	
def surroundings(halo,s,radius,color=(1,1,1),scale=1.0):
	centre = pynbody.analysis.halo.center_of_mass(halo)
	filt = pynbody.filt.Sphere(radius,centre)
	subsnap_scatter(s[filt],color_spec=color,scale_factor=scale)

# Plot numbered halos:
def plot_numbered_halos(h,to_plot,halo_centres,halo_colour=(0,0,1),text_scale=1000,text_colour=(1,1,1)):
	# h - halo catalogue
	# to_plot - indices (starting from zero - NOT the same as halo number)
	# halo_centres - positions of the centres of mass of the specified halos.
	for k in range(0,len(to_plot)):
		r = halo_centres[to_plot[k]]
		subsnap_scatter(h[to_plot[k]+1],color_spec=halo_colour)
		text3d(r[0],r[1],r[2],str(to_plot[k]+1),scale=text_scale,color=text_colour)

# Plot a list of clusters, together with their names:
def plot_named_clusters(cluster_pos,cluster_names,color_spec=(0,1,0),point_type='sphere',scale=1000,text_colour=(0,1,0),text_scale=1000):
	points3d(cluster_pos[:,0],cluster_pos[:,1],cluster_pos[:,2],color=color_spec,mode=point_type,scale_factor=scale)
	for k in range(0,len(cluster_names)):
		text3d(cluster_pos[k,0],cluster_pos[k,1],cluster_pos[k,2],cluster_names[k],color=text_colour,scale=text_scale)

def plot_numbered_voids(hr,to_plot,void_centres,bridge,void_colour=(1,0,0),text_scale=1000,text_colour=(1,0,0)):
	for k in range(0,len(to_plot)):
		r = void_centres[to_plot[k]]
		subsnap_scatter(bridge(hr[to_plot[k]+1]),color_spec=void_colour,type='sphere',scale=500)
		text3d(r[0],r[1],r[2],str(to_plot[k]+1),scale=text_scale,color=text_colour)

def recentre(centre):
	f = mlab.gcf()
	camera = f.scene.camera
	camera.focal_point = centre

def line_plot(line,color=(1,0,0),line_width=10,reset_zoom=False):
	plot3d(line[:,0],line[:,1],line[:,2],line_width=line_width,reset_zoom=reset_zoom,color=color,representation='wireframe')

def plotHistory(sn,sr,halo_list,halo,color_list,highlight_mode='point3d',scalefactor=1):
	childList = np.array(halo.properties['children']) - 1
	children = context.combineHalos(sn,halo_list,childList)
	extras = halo.setdiff(children)
	b = pynbody.bridge.Bridge(sn,sr)
	subsnap_scatter(b(extras),color_spec=(1,1,1))
	for k in range(0,len(childList)):
		subsnap_scatter(b(halo_list[childList[k]+1]),color_spec=color_list[np.mod(k,len(color_list))],scale=scalefactor,type=highlight_mode)

# Animate the evolution of the specified snapshots:
def animate(snaplist,plot_command,save_directory="./",size=None,scaling=1):
	filenames = []
	for k in snaplist:
		filenames.append(save_directory + "snapshot_" + "{:0>3d}".format(k) + ".png")
		plot_command(k)
		fig = mlab.gcf().scene
		if size is None:
			sceneSize = np.array(fig.get_size())*scaling
		else:
			sceneSize = np.array(size)*scaling
		mlab.savefig(save_directory + "snapshot_" + "{:0>3d}".format(k) + ".png",size=sceneSize)
		mlab.clf()
	# Construct gif:
	with imageio.get_writer(save_directory + "animation.gif",mode='I',duration=1) as writer:
		for filename in filenames:
			image = imageio.imread(filename)
			writer.append_data(image)

# Halves the RGB values of a specified color
def half_color(color):
	if(len(color) != 3):
		raise Exception("Colour must be a three element tuple.")
	return (color[0]/2,color[1]/2,color[2]/2)

# Construct a set of ncolors even spaced colours in rgb space.
def construct_color_list(n,ncolors):
	ncbrt = np.ceil(np.cbrt(ncolors))
	if(n > ncolors):
		raise Exception("Requested colour exceeds maximum specified number of colours. Specify more colours.")
	k = np.floor(n/(ncbrt**2))
	j = np.floor((n - k*ncbrt**2)/ncbrt)
	i = (n - k*ncbrt**2 - j*ncbrt)
	return (i/(ncbrt-1),j/(ncbrt-1),k/(ncbrt-1))

# Returns the specified number in scientific notation:
def scientificNotation(x,latex=False,s=3,powerRange = 0):
	log10x = np.log10(np.abs(x))
	z = np.floor(log10x).astype(int)
	y = 10.0**(log10x - z)
	if z > powerRange:
		resultString = "10^{" + "{0:0d}".format(z) + "}"
		if y != 1.0:
			resultString = ("{0:." + str(s) + "g}").format(y) + "\\times " + resultString
		if latex:
			resultString = "$" + resultString + "$"
	else:
		resultString = ("{0:." + str(s) + "g}").format(y*(10.0**z))
	if x < 0:
		resultString = "-" + resultString
	return resultString

# Plot binned halo densities as a function of redshift
def plot_halo_void_densities(z,rhoVnav,rhoVnsd,rhoVrav,rhoVrsd,bins):
	binNo = len(bins) - 1
	legendList = []
	# Want to format bins in scientific notation:
	
	for k in range(0,binNo):
		plt.semilogy(z,rhoVnav[:,k],color=construct_color_list(k+1,2*binNo))
		plt.fill_between(z,rhoVnav[:,k] - rhoVnsd[:,k],rhoVnav[:,k] + rhoVnsd[:,k],color=half_color(construct_color_list(k+1,2*binNo)))
		legendList.append('Halo Density, $' + scientificNotation(bins[k]) + '$ - $' + scientificNotation(bins[k+1]) + ' M_{sol}/h$')
	for k in range(0,binNo):
		plt.semilogy(z,rhoVrav[:,k],color=construct_color_list(binNo + k+1,2*binNo))
		plt.fill_between(z,rhoVrav[:,k] - rhoVrsd[:,k],rhoVrav[:,k] + rhoVrsd[:,k],color=half_color(construct_color_list(binNo + k+1,2*binNo)))
		legendList.append('Anti-halo Density, $' + scientificNotation(bins[k]) + '$ - $' + scientificNotation(bins[k+1]) + ' M_{sol}/h$')
	plt.xlabel('z')
	plt.ylabel('(Local Density)/(Background Density)')
	plt.legend(legendList)
	plt.show()

def computeHistogram(x,bins,z=1.0,density = True):
	noInBins = np.zeros(len(bins)-1,dtype=int)
	N = len(x)
	prob = np.zeros(len(bins)-1)
	sigma = np.zeros(len(bins)-1)
	inBins = []
	if N != 0:
		for k in range(0,len(bins)-1):
			inBins.append(np.where((x > bins[k]) & (x < bins[k+1]))[0])
			noInBins[k] = len(inBins[k])
			# Estimate of the probability density for this bin:
			p = len(inBins[k])/N
			if density:
				prob[k] = p/(bins[k+1] - bins[k])
			else:
				prob[k] = len(inBins[k])
			# Normal distribution approximation of the confidence interval on this density:
			if density:
				sigma[k] = z*np.sqrt(p*(1.0-p)/N)/(bins[k+1] - bins[k])
			else:
				sigma[k] = z*np.sqrt(p*(1.0-p)*N)
	return [prob,sigma,noInBins,inBins]

# Create bins for a list of values:
def createBins(values,nBins,log=False):
	if log:
		# Logarithmically spaced bins:
		return 10**np.linspace(np.log10(np.min(values)),np.log10(np.max(values)),nBins+1)
	else:
		# linearly spaced bins:
		return np.linspace(np.min(values),np.max(values),nBins+1)

# Plot a histogram, but include error bars for the confidence interval of the uncertainty
def histWithErrors(p,sigma,bins,ax = None,label="Bin probabilities"):
	x = (bins[1:len(bins)] + bins[0:(len(bins)-1)])/2
	width = bins[1:len(bins)] - bins[0:(len(bins)-1)]
	if ax is None:
		return plt.bar(x,p,width=width,yerr=sigma,alpha=0.5,label=label)
	else:
		return ax.bar(x,p,width=width,yerr=sigma,alpha=0.5,label=label)

# Histogram of halo densities
def haloHistogram(logrho,logrhoBins,masses,massBins,massBinList = None,massBinsToPlot = None,density=True,logMassBase = None,subplots=True,subplot_shape=None):
	# Plot all the mass bins unless otherwise specified:
	if massBinsToPlot is None:
		massBinsToPlot = range(0,len(massBins)-1)
	# Bin the masses of the halos if this has not already been supplied.
	if massBinList is None:
		[massBinList,noInBins] = binValues(masses,massBins)
	legendList = []
	if subplots:
		if subplot_shape is not None:
			# Check that the requested shape makes sense:
			if len(subplot_shape) != 2:
				raise Exception("Sub-plots must be arranged on a 2d grid.")
			if subplot_shape[0]*subplot_shape[1] < len(massBinsToPlot):
				raise Exception("Not enough room in requested sub-plot arrangement to fit all plots.")
			a = subplot_shape[0]
			b = subplot_shape[1]
		else:
			nearestSquareRoot = np.ceil(np.sqrt(len(massBinsToPlot))).astype(int)
			a = b = nearestSquareRoot
		fig, ax = plt.subplots(nrows=a,ncols=b)
		counter = 0
	for k in massBinsToPlot:
		[p,sigma,noInBins,inBins] = computeHistogram(logrho[massBinList[k]],logrhoBins)
		if subplots:
			# Plot axes on a square grid:
			i = np.floor(counter/b).astype(int)
			j = np.mod(counter,b).astype(int)
			histWithErrors(p,sigma,logrhoBins,ax[i,j])
			if logMassBase is None:
				ax[i,j].legend(['$' + scientificNotation(massBins[k]) + ' < M < ' + scientificNotation(massBins[k+1]) + ' M_{sol}/h$'])
			else:
				ax[i,j].legend(['$' + scientificNotation(logMassBase**massBins[k]) + '$ < M < $' + scientificNotation(logMassBase**massBins[k+1]) + ' M_{sol}/h$'])
			if i == a - 1:
				ax[i,j].set_xlabel('$log(\\langle\\rho\\rangle_V/\\bar{\\rho})$')
			if j == 0:
				ax[i,j].set_ylabel('Probability Density')
			counter = counter + 1			
		else:
			# Plot everything on one axis:
			histWithErrors(p,sigma,logrhoBins)
		if logMassBase is None:
			legendList.append('$' + scientificNotation(massBins[k]) + ' < M < ' + scientificNotation(massBins[k+1]) + ' M_{sol}/h$')
		else:
			# Exponentiate the masses if they were supplied in log space:
			legendList.append('$' + scientificNotation(logMassBase**massBins[k]) + ' < M < ' + scientificNotation(logMassBase**massBins[k+1]) + ' M_{sol}/h$')
	if not subplots:
		plt.xlabel('$log(\\langle\\rho\\rangle_V/\\bar{\\rho})$')
		plt.ylabel('Probability Density')
		plt.legend(legendList)
	plt.show()

# Plot Fraction of halos in a set of mass bins that are underdense:
def plotUnderdenseFraction(frac,sigma,logMassBins):
	# frac and sigma should be computed by halo_analysis.getExpansionFraction
	# Assuming given in log10 mass bins:
	massCentres = 10**((logMassBins[1:len(logMassBins)] + logMassBins[0:(len(logMassBins)-1)])/2)
	fig, ax = plt.subplots()
	ax.errorbar(massCentres,frac,yerr=sigma)
	ax.set_xscale('log')
	plt.xlabel('Mass bin$/M_{sol}/h$')
	plt.ylabel('Underdense fraction at z = 0')
	plt.show()

# Plot average density in each of the supplies mass bins:
def plotMassBinDensity(rhoV,binList,logMassBins):
	massCentres = 10**((logMassBins[1:len(logMassBins)] + logMassBins[0:(len(logMassBins)-1)])/2)
	fig, ax = plt.subplots()
	rhoVav = np.zeros(len(massCentres))
	rhoVsd = np.zeros(len(massCentres))
	for k in range(0,len(rhoVav)):
		if len(binList[k] != 0):
			rhoVav[k] = np.mean(rhoV[binList[k]])
			rhoVsd[k] = np.sqrt(np.var(rhoV[binList[k]])/len(binList[k]))
	ax.errorbar(massCentres,rhoVav,yerr=rhoVsd)
	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.xlabel('Mass bin$/M_{sol}/h$')
	plt.ylabel('$\\langle\\rho\\rangle/\\bar{\\rho}$')
	plt.show()

# Linear regression, together with various statistics
def linearRegression(x,y,full=False,errors=False):
	z = np.polyfit(x,y,deg=1)
	a = z[0]
	b = z[1]
	xbar = np.mean(x)
	ybar = np.mean(y)
	n = len(x)
	ssxx = np.sum((x - xbar)**2)
	ssyy = np.sum((np.log10(y) - ybar)**2)
	ssxy = np.sum((np.log10(y) - ybar)*(x - xbar))
	ssxy = np.sum((np.log10(y) - ybar)*(x - xbar))
	r2 = (ssxy**2)/(ssxx*ssyy)
	s = np.sqrt((ssyy - (ssxy**2)/ssxx)/(n-2))
	sea = s*np.sqrt((1/n) + (xbar**2)/ssxx)
	seb = s*np.sqrt(ssxx)
	if (full and errors):
		return [a,b,sea,seb,ssxx,ssyy,ssyy,r2,s]
	elif full:
		return [a,b,ssxx,ssyy,ssyy,r2,s]
	elif errors:
		return [a,b,sea,seb]
	else:
		return [a,b]

# Plot the convex hull around a set of points:
def plotConvexHull(snap,hull=None,color=(0,1,0),opacity=0.3,vertices=False):
	if hull is None:
		hull = halo_analysis.getConvexHull(snap)
	if vertices:
		point_scatter(snap['pos'][hull.vertices],color_spec=color,type='sphere')
	triangular_mesh(snap['pos'][:,0],snap['pos'][:,1],snap['pos'][:,2],hull.simplices,color=color,opacity=opacity,representation = 'wireframe')

def plotConvexHullFromPoints(pos,hull=None,color=(0,1,0),opacity=0.3,vertices=False):
	if hull is None:
		hull = spatial.ConvexHull(pos)
	triangular_mesh(pos[:,0],pos[:,1],pos[:,2],hull.simplices,color=color,opacity=opacity,representation = 'wireframe')

# Generate colours on the fly:
def linearColour(n,nmax,colourMap=cm.jet):
	return colourMap(np.int32(np.round((n/nmax)*256)))[0:3]


# Plot a halo relative to the centre of mass:
def plotPointsRelative(pos,boxsize,centre = None,weights = None,color_spec=(1,1,1),type='2dvertex',scale=1.0):
	if centre is None:
		if weights is None:
			weights = np.ones(len(pos))
		centre = context.computePeriodicCentreWeighted(pos,weights,boxsize)
	posAdjusted = snapedit.unwrap(snapedit.wrap(snapedit.unwrap(pos,boxsize)  - snapedit.unwrap(centre,boxsize),boxsize),boxsize)
	point_scatter(posAdjusted,color_spec=color_spec,type=type,scale=scale)
	
def plotHaloRelative(halo,centre = None,weights = None,color_spec=(1,1,1),type='2dvertex',scale=1.0):
	boxsize = halo.properties['boxsize'].ratio("Mpc a h**-1")
	plotPointsRelative(halo['pos'],boxsize,centre=centre,weights = weights,color_spec=color_spec,type=type,scale=scale)

def float_formatter(x,d=2):
	return str(np.around(x,decimals=d))
	
# Convert an array of floats into an array of strings:
def floatsToStrings(floatArray,precision=2):
	return [("%." + str(precision) + "f") % number for number in floatArray]

# Violin plots
def plotViolins(rho,radialBins,radiiFilter=None,ylim=1.4,ax = None,fontsize=14,fontname="serif",color=None,inner=None,linewidth=None,saturation=1.0,palette="colorblind"):
	radii = binCentres(radialBins)
	if radiiFilter is None:
		radiiFilter = np.arange(0,len(radii))
	if ax is None:
		fig, ax = plt.subplots()
	panData = pandas.DataFrame(data=rho[:,radiiFilter],columns=floatsToStrings(radii[radiiFilter]))
	sns.violinplot(data=panData,ax=ax,color=color,inner=inner,linewidth=linewidth,saturation=saturation,palette=palette)
	ax.set_xlabel('$R/R_{\\mathrm{eff}}$',fontsize=fontsize,fontfamily=fontname)
	ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=fontsize,fontfamily=fontname)
	ax.set_ylim([0,ylim])
	xlim = ax.get_xlim()
	ax.tick_params(axis='both',labelsize=fontsize)
	ax.axhline(y = 1.0,color='0.75',linestyle=':')

class LinearMapper:
	def __init__(self,inMin,inMax,outMin=0,outMax=1):
		self.inMin = inMin
		self.inMax = inMax
		self.outMin = outMin
		self.outMax = outMax
	def __call__(self,x,clip=False):
		return self.outMin + (self.outMax - self.outMin)*(x - self.inMin)/(self.inMax - self.inMin)
	def autoscale(self,A):
		self.inMin = np.min(A)
		self.inMax = np.max(A)
	def inverse(self,x):
		return self.inMin + (self.inMax - self.inMin)*(x - self.outMin)/(self.outMax - self.outMin)


class LogMapper:
	def __init__(self,inMin,inMax,outMin=0,outMax=1,logMin = 1.0):
		self.inMin = np.log(logMin + inMin)
		self.inMax = np.log(logMin + inMax)
		self.outMin = outMin
		self.outMax = outMax
		self.logMin = logMin
	def __call__(self,x,clip=False):
		return self.outMin + (self.outMax - self.outMin)*(np.log(self.logMin + x) - self.inMin)/(self.inMax - self.inMin)
	def autoscale(self,A):
		self.inMin = np.min(A)
		self.inMax = np.max(A)


	


