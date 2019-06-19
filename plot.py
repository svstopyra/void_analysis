#
from mayavi.mlab import points3d, text3d, plot3d
from mayavi import mlab
import pynbody
from void_analysis import context
import numpy as np
import imageio
import os
import gc

# Plot the positions in a snapshot:
def subsnap_scatter(subsnap,color_spec=(1,1,1),scale=1.0,type='2dvertex'):
	r = subsnap['pos']
	points3d(r[:,0],r[:,1],r[:,2],mode=type,color=color_spec,scale_factor=scale)

# Plot a set of specified points (wrapper for points3d):
def point_scatter(r,color_spec=(1,1,1),scale=1.0,type='2dvertex'):
	points3d(r[:,0],r[:,1],r[:,2],mode=type,color=color_spec,scale_factor=scale)

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
	
	
