from mayavi.mlab import points3d, text3d, plot3d, triangular_mesh
from mayavi import mlab
import pynbody
from . import context
import numpy as np
import scipy.spatial as spatial


# Plot about a specific point:
def plotAboutPoint(snap,point,radius = 10,showCentre=True,color=(1,1,1)):
    filt = pynbody.filt.Sphere(radius,point)
    subsnap_scatter(snap[filt],color_spec=color)
    if showCentre:
        point_scatter(point,type='sphere',scale=1.0,color_spec=(1,0,0))
    mlab.show()

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


# Plot the surroundings of the specified halo	
def surroundings(halo,s,radius,color=(1,1,1),scale=1.0):
	centre = pynbody.analysis.halo.center_of_mass(halo)
	filt = pynbody.filt.Sphere(radius,centre)
	subsnap_scatter(s[filt],color_spec=color,scale_factor=scale)

# Plot numbered halos:
def plot_numbered_halos(h,to_plot,halo_centres,
	halo_colour=(0,0,1),text_scale=1000,text_colour=(1,1,1)):
	# h - halo catalogue
	# to_plot - indices (starting from zero - NOT the same as halo number)
	# halo_centres - positions of the centres of mass of the specified halos.
	for k in range(0,len(to_plot)):
		r = halo_centres[to_plot[k]]
		subsnap_scatter(h[to_plot[k]+1],color_spec=halo_colour)
		text3d(r[0],r[1],r[2],str(to_plot[k]+1),scale=text_scale,color=text_colour)


# Plot a list of clusters, together with their names:
def plot_named_clusters(cluster_pos,cluster_names,color_spec=(0,1,0),
	point_type='sphere',scale=1000,text_colour=(0,1,0),text_scale=1000):
	points3d(cluster_pos[:,0],cluster_pos[:,1],
		cluster_pos[:,2],color=color_spec,mode=point_type,scale_factor=scale)
	for k in range(0,len(cluster_names)):
		text3d(cluster_pos[k,0],cluster_pos[k,1],
			cluster_pos[k,2],cluster_names[k],color=text_colour,scale=text_scale)

def plot_numbered_voids(hr,to_plot,void_centres,bridge,
	void_colour=(1,0,0),text_scale=1000,text_colour=(1,0,0)):
	for k in range(0,len(to_plot)):
		r = void_centres[to_plot[k]]
		subsnap_scatter(
			bridge(hr[to_plot[k]+1]),color_spec=void_colour,type='sphere',scale=500)
		text3d(r[0],r[1],r[2],str(to_plot[k]+1),scale=text_scale,color=text_colour)

def recentre(centre):
	f = mlab.gcf()
	camera = f.scene.camera
	camera.focal_point = centre

def line_plot(line,color=(1,0,0),line_width=10,reset_zoom=False):
	plot3d(line[:,0],line[:,1],
		line[:,2],line_width=line_width,reset_zoom=reset_zoom,
		color=color,representation='wireframe')

def plotHistory(sn,sr,halo_list,halo,color_list,highlight_mode='point3d',scalefactor=1):
	childList = np.array(halo.properties['children']) - 1
	children = context.combineHalos(sn,halo_list,childList)
	extras = halo.setdiff(children)
	b = pynbody.bridge.Bridge(sn,sr)
	subsnap_scatter(b(extras),color_spec=(1,1,1))
	for k in range(0,len(childList)):
		subsnap_scatter(
		b(halo_list[childList[k]+1]),color_spec=color_list[np.mod(k,len(color_list))],
		scale=scalefactor,type=highlight_mode)

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
		mlab.savefig(save_directory + "snapshot_" + \
		"{:0>3d}".format(k) + ".png",size=sceneSize)
		mlab.clf()
	# Construct gif:
	with imageio.get_writer(save_directory + "animation.gif",mode='I',duration=1) as writer:
		for filename in filenames:
			image = imageio.imread(filename)
			writer.append_data(image)

# Plot a 3D interactive scatter of Abell clusters, 2M++ catalogue, and BORG halos:

def plotLocalUniverse(points2MPP,centres,masses,
		l_clusters,b_clusters,d_clusters,abell_d,abell_n,p_clusters,p_voids,p_abell,
		cluster_names,void_names,h=0.705,bgcolor=(0,0,0),rCut = 135,
		mUpper=1e16,mMid = 1e15,mLower = 5e14,upperScale=5,lowerScale=3,textScale=2,
		lowerColour=(1,0,1),upperColour=(0,0,1),clusterColour=(1,0,0),
		voidColour = (0,1,0),clusterTextScale=10,voidTextScale=10,
		abellColour=(1,1,0),abellScale=0.5,show=True,plotLimit=None,\
		mfig = None,returnFig=False,showVoids=True,showSuperclusters=True):
	if mfig is None:
		mlab.figure(bgcolor=bgcolor)
	point_scatter(points2MPP)
	# Add Halos:
	if rCut is not None:
		maskCondition = (np.sqrt(np.sum(centres**2,1)) <= rCut)
	else:
		maskCondition = np.ones(masses.shape,dtype=bool)
	plotCondition = np.where(maskCondition & (masses > mMid) & (masses < mUpper))[0]
	plotCondition2 = np.where(maskCondition & (masses > mLower) & (masses < mMid))[0]
	point_scatter(centres[plotCondition,:]/h,
		color_spec=upperColour,type='sphere',scale=upperScale)
	point_scatter(centres[plotCondition2,:]/h,
		color_spec=lowerColour,type='sphere',scale=lowerScale)
	if plotLimit is not None:
		lim1 = np.min([len(plotCondition),plotLimit])
		lim2 = np.min([len(plotCondition2),plotLimit])
	else:
		lim1 = len(plotCondition)
		lim2 = len(plotCondition2)
	for k in range(0,lim1):
		mlab.text3d(centres[plotCondition[k],0]/h,
			centres[plotCondition[k],1]/h,centres[plotCondition[k],2]/h,
			str(plotCondition[k]),color=upperColour,scale=upperScale)
	for k in range(0,lim2):
		mlab.text3d(centres[plotCondition2[k],0]/h,\
			centres[plotCondition2[k],1]/h,
			centres[plotCondition2[k],2]/h,
			str(plotCondition2[k]),color=lowerColour,scale=upperScale)
	# Supercluster expected locations:
	if showSuperclusters:
		point_scatter(p_clusters,type='sphere',color_spec=clusterColour,scale=1)
		for k in range(0,len(cluster_names)):
			mlab.text3d(p_clusters[k,0],p_clusters[k,1],
				p_clusters[k,2],cluster_names[k],color=clusterColour,
				scale=clusterTextScale)
	#Voids:
	if showVoids:
		point_scatter(p_voids,type='sphere',color_spec=voidColour,scale=1)
		for k in range(0,len(void_names)):
			mlab.text3d(p_voids[k,0],p_voids[k,1],p_voids[k,2],
				void_names[k],color=voidColour,scale=voidTextScale)
	# Abell clusters:
	# cluster expected locations:
	if rCut is not None:
		abellToPlot = np.where(abell_d < rCut)[0]
	else:
		abellToPlot = np.ones(abell_d.shape,dtype=int)
	point_scatter(p_abell[abellToPlot]/h,type='sphere',
		color_spec=abellColour,scale=abellScale)
	if type(abell_n[0]) == str:
		for k in range(0,len(p_abell[abellToPlot])):
			mlab.text3d(p_abell[abellToPlot][k,0]/h,p_abell[abellToPlot][k,1]/h,
				p_abell[abellToPlot][k,2]/h,abell_n[abellToPlot[k]],
				color=abellColour,scale=textScale)
	else:
		for k in range(0,len(p_abell[abellToPlot])):
			mlab.text3d(p_abell[abellToPlot][k,0]/h,p_abell[abellToPlot][k,1]/h,
				p_abell[abellToPlot][k,2]/h,str(abell_n[abellToPlot][k]),
				color=abellColour,scale=textScale)
	if show:
		mlab.show()
	if returnFig:
		return mfig


# Plot the convex hull around a set of points:
def plotConvexHull(snap,hull=None,color=(0,1,0),opacity=0.3,vertices=False):
	if hull is None:
		hull = halo_analysis.getConvexHull(snap)
	if vertices:
		point_scatter(snap['pos'][hull.vertices],color_spec=color,type='sphere')
	triangular_mesh(snap['pos'][:,0],snap['pos'][:,1],
		snap['pos'][:,2],hull.simplices,color=color,opacity=opacity,
		representation = 'wireframe')

def plotConvexHullFromPoints(pos,hull=None,color=(0,1,0),opacity=0.3,vertices=False):
	if hull is None:
		hull = spatial.ConvexHull(pos)
	triangular_mesh(pos[:,0],pos[:,1],
		pos[:,2],hull.simplices,color=color,
		opacity=opacity,representation = 'wireframe')
