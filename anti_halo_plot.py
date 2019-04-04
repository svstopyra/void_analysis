# Python script to plot halos and the corresponding anti-halo
import pynbody
from mayavi.mlab import points3d

def scatter(halo,s1,s2):
	b = pynbody.bridge.Bridge(s1,s2)
	r1 = halo['pos']
	r2 = b(halo)['pos']
	points3d(r1[:,0],r1[:,1],r1[:,2],mode='2dvertex',color=(1,1,1))
	points3d(r2[:,0],r2[:,1],r2[:,2],mode='2dvertex',color=(1,0,0))
	
	
