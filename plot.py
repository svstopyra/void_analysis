#
from mayavi.mlab import points3d
import pynbody

def subsnap_scatter(subsnap,color_spec=(1,1,1)):
	r = subsnap['pos']
	points3d(r[:,0],r[:,1],r[:,2],mode='2dvertex',color=color_spec)

# Returns a subsnap contains the particles within halos only that satisfy the specified filter
#def halo_filter(halo_list,filt):
	

# Plot the surroundings of the specified halo	
def surroundings(halo,s,radius,color=(1,1,1)):
	centre = pynbody.analysis.halo.center_of_mass(halo)
	filt = pynbody.filt.Sphere(radius,centre)
	subsnap_scatter(s[filt],color_spec=color)
