# Plot the local structure around the Local group:
import pynbody
from void_analysis import plot, context
from mayavi.mlab import points3d, text3d
import pickle
import numpy as np
[r1,r2,r3,R1,R2,R3] = pickle.load(open("map_vectors_virgo_centaurus.p","rb"))
[cl_sgl_pos,cluster_names,void_sgl_pos,cl_pos,void_pos] = pickle.load(open("cluster_list_processed.p","rb"))
sn = pynbody.load("unreversed/cf2gvpecc1pt5elmo73_sig6distribsbvoldiv_RZA3Derrv2_256_500_ss8_zinit60_000")
hn = sn.halos()
[hn_centres,hn_masses] = pickle.load(open("unreversed/halo_additional_properties.p","rb"))
centre = np.array([250000,250000,250000])
radius = 50000
filt = pynbody.filt.Sphere(radius,centre)
# Plot background of N body simulation:
plot.subsnap_scatter(sn[filt])
# Highlight the halos in this region:
hn_in_sphere = context.halos_in_sphere(hn,radius,centre,hn_centres)
plot.plot_numbered_halos(hn,hn_in_sphere,hn_centres)
# Overlay the observed clusters rotated to match the N-body data:
plot.plot_named_clusters(cl_pos+centre,cluster_names)
# Add in the ray showing the direction to the local void:
points3d(void_pos[:,0]+centre[0],void_pos[:,1]+centre[1],void_pos[:,2]+centre[2],mode='sphere',color=(1,0,0),scale_factor=1000)

