# Functions for identifying the context of a specified halo:
import numpy as np
import pynbody

# Returns the nearest neighbour halos to the specified halo
def get_nearest_halos(centre,halo_list,coms='None',neighbours=1):
	# centre - position to find nearest neighbours to.
	# halo_list - halo catalogue
	# coms - centres of mass of the halos specified by halo_list (if 'None', this is generated)
	# neighbours - number of nearest neighbour halos to find.
	if coms == 'None':
		coms = np.zeros([len(halo_list),3])
		print("Computing halo centres of mass (this may take some time)...")
		for i in range(0,len(halo_list)):
			come[i,:] = pynbody.analysis.halo.center_of_mass(halo_list[i+1])

	# Construct KD tree for halo positions:	
	tree = spatial.cKDTree(coms)
	# Get nearest neighbours
	nearest = tree.query(centre,neighbours)
	return nearest[1]
	



	
