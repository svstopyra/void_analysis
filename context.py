# Functions for identifying the context of a specified halo:
import numpy as np
import pynbody
import pickle

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

# Compute some additional properties of the halo catalogue (and possible save them to a file)
def halo_centres_and_mass(h,save_file='none'):
	halo_centres = np.zeros([len(h),3])
	halo_masses = np.zeros(len(h))
	for k in range(0,len(h)):
		halo_centres[k,:] = pynbody.analysis.halo.center_of_mass(h[k+1])
		halo_masses[k] = np.sum(h[k+1]['mass'])
	if(save_file != 'none'):
		pickle.dump([halo_centres,halo_masses],open(save_file,"wb"))
	return [halo_centres,halo_masses]

# Gets halos in the reversed simulation (sr) and returns a list of their centres in the unreversed simulation.
def void_centres(sn,sr,hr):
	void_centre_list = np.zeros([len(hr),3])
	b = pynbody.bridge.Bridge(sn,sr)
	for k in range(0,len(hr)):
		void_centre_list[k,:] = pynbody.analysis.halo.center_of_mass(b(hr[k+1]))
	return void_centre_list
	
# Converts a list of reversed simulation halos to unreversed simulation voids:
def anti_halos_to_voids(ah_list,sn,sr):
	void_list = [];
	b = pynbody.bridge.Bridge(sn,sr)
	for k in ah_list:
		void_list.append(b(k))
	return void_list

def halo_filter(s,h,filt):
	# s - simulation snapshot
	# h - halo list (could also be a list of 
	halo_list = np.zeros(len(h))
	for k in range(0,len(h)):
		if len(h[k+1][filt] > 0):
			halo_list[k] = 1
	# Get halo indices (offset by 1 downwards from the halo number)
	indices = np.where(halo_list == 1)
	return indices[0]

# Get halos inside a specified sphere, using centre and mass:
def halos_in_sphere(h,radius,centre,halo_centres='none'):
	# Generate halo_centres and masses if these are not given:
	if(halo_centres == 'none'):
		[halo_centres,halo_masses] = halo_centres_and_mass(h)
	#Get distance from the centre specified:
	r = np.sqrt(np.sum((halo_centres - centre)**2,1))
	in_sphere = np.where(r < radius)
	collect = np.zeros(len(h))
	collect[in_sphere[0]] = 1
	# Check that these halos are actually in the sphere, because sometimes halos
	# can appear to have their centre of mass inside it, even though they are outside
	# because they lie on the periodic boundary (thus half their mass is on one side of the sphere and half on the other).
	filt = pynbody.filt.Sphere(radius,centre)
	for k in range(0,len(in_sphere[0])):
		if (len(h[in_sphere[0][k]+1][filt]) == 0):
			collect[in_sphere[0][k]] = 0
	in_sphere = np.where(collect == 1)
	return in_sphere[0]
def voids_in_sphere(hr,radius,centre,sn,sr,void_centre_list='none'):
	if(void_centre_list == 'none'):
		void_centre_list = void_centres(sn,sr,hr)
	r = np.sqrt(np.sum((void_centre_list - centre)**2,1))
	in_sphere = np.where(r < radius)
	collect = np.zeros(len(hr))
	collect[in_sphere[0]] = 1
	filt = pynbody.filt.Sphere(radius,centre)
	b = pynbody.bridge.Bridge(sn,sr)
	for k in range(0,len(in_sphere[0])):
		if (len(b(hr[in_sphere[0][k]+1])[filt]) == 0):
			collect[in_sphere[0][k]] = 0
	in_sphere = np.where(collect == 1)
	return in_sphere[0]

def anti_halo_filter(sn,sr,hr,filt):
	b = pynbody.bridge.Bridge(sn,sr)
	
# Compute the rotation matrix between two specified vectors.
# Returns the matrix that rotates the unit vector of a onto
# the unit vector of b:
def rotation_between(a,b):
	if(len(a) != 3 or len(b) != 3):
		raise NameError('Invalid 3 vectors')
	na = a/np.sqrt(np.sum(a**2))
	nb = b/np.sqrt(np.sum(b**2))
	nanb = np.dot(na,nb)
	v = np.cross(na,nb)
	v2 = np.sum(v**2)
	vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
	return np.identity(3) + vx + np.dot(vx,vx)*(1.0 - nanb)/v2

# Returns the part of vector a that is orthogonal to b:
def orthogonal_part(a,b):
	if(len(a) != 3 or len(b) != 3):
		raise NameError('Invalid 3 vectors')
	return a - np.dot(a,b)*b/(np.sum(b**2))

# Attempt to fit three clusters in one data set to match three clusters in another
def cluster_fit(cluster,r1,r2,r3,R1,R2,R3):
	# cluster - list of vectors that need to be transformed. Should be 3 columns, arbitrary number of rows.
	# to_map - 3-vector with indices of the vectors that we want to map onto R1,R2,R3	
	# Map everything so that R1 and r1 respectively lie at the origins of the sets of points:
	R2p = R2 - R1
	R3p = R3 - R1
	clusterp = cluster - r1
	r2p = r2 - r1
	r3p = r3 - r1	
	# Now rotate the whole system of vectors so that r2 lies as close as possible to R2, ie,
	# so that their unit vectors point along the same direction:
	M = rotation_between(r2p,R2p)
	clusterp = np.dot(clusterp,M.T)
	r3p = np.dot(r3p,M.T)
	# Next, rotate about the axis between R2 and R1 so that r3 best matches R3. That means
	# finding the rotation that maps the part of r3 orthogonal to R2 - R1 onto the part of R3 orthogonal to R2 - R1:
	M = rotation_between(orthogonal_part(r3p,R2p),orthogonal_part(R3p,R2p))
	clusterp = np.dot(clusterp,M.T)
	# Finally, add back in R1:
	return clusterp + R1

# Converts equitorial co-ordinate data into 3d positions in kpc/h:
def position3d(equitorial):
	# equitorial - N x 3 matrix, with first column the right ascension, second column the declension, and 3rd column the observed redshift.
	c = 299792458
	kpc = 3.086e+19
	# Hubble rate, H0/h:
	H0 = 3.240440699935191e-18
	# Distance in kpc, using Hubble's law:
	pos = np.zeros(equitorial.shape)
	if(equitorial.ndim > 1):
		D = c*equitorial[:,2]/(H0*kpc)
		# Angular co-ordinates:
		phi = equitorial[:,0]*np.pi/180.0
		theta = (90.0 - equitorial[:,1])*np.pi/180.0
		pos[:,0] = D*np.sin(theta)*np.cos(phi)
		pos[:,1] = D*np.sin(theta)*np.sin(phi)
		pos[:,2] = D*np.cos(theta)
	else:
		D = c*equitorial[2]/(H0*kpc)
		# Angular co-ordinates:
		phi = equitorial[0]*np.pi/180.0
		theta = (90.0 - equitorial[1])*np.pi/180.0
		pos[0] = D*np.sin(theta)*np.cos(phi)
		pos[1] = D*np.sin(theta)*np.sin(phi)
		pos[2] = D*np.cos(theta)
	return pos

# Converts galactic co-ordinates into equitorial co-ordinates:
def galactic_to_equitorial(galactic):
	# galactic - N x 2 array with galactic longitude in column 1, galactic latitude in column 2
	# Ascension and Declination of the North Galactic Pole in radians):
	aNGP = 192.85948*np.pi/180.0
	dNGP = 27.12825*np.pi/180.0
	# And the north celestial pole in galactic longitude:
	lNCP = 122.93192*np.pi/180.0
	# Galactic co-ordinates in radians:
	if galactic.ndim > 1:
		l = galactic[:,0]*np.pi/180.0
		b = galactic[:,1]*np.pi/180.0
	else:
		l = galactic[0]*np.pi/180.0
		b = galactic[1]*np.pi/180.0
	# Intermediates:
	sinb = np.sin(b)
	cosb = np.cos(b)
	sinlp = np.sin(lNCP - l)
	coslp = np.cos(lNCP - l)
	sindNGP = np.sin(dNGP)
	cosdNGP = np.cos(dNGP)
	# Sin of the declination angle:
	sind = sindNGP*sinb + cosdNGP*cosb*coslp
	cosd = np.sqrt(1.0 - sind**2)
	# Have to check the sin of the RHS here to get the correct conversion to right ascension:
	sinap = cosb*sinlp/cosd
	cosap = (sinb*cosdNGP - cosb*sindNGP*coslp)/cosd
	negrange = np.where(sinap < 0)
	alphap = (np.arccos(cosap))
	# Get the correct angles for anything that is actually between pi and 2pi radians:
	if galactic.ndim > 1:
		alphap[negrange] = (2.0*np.pi - alphap[negrange])
	elif sinap < 0.0:
		alphap = 2.0*np.pi - alphap
		
	# Add on the right ascension of the NGP:
	alpha = np.mod(alphap + aNGP,2*np.pi)
	# Right ascension is easy, because asin always gives something between -pi/2 and pi/2:
	delta = np.arcsin(sind)
	
	# Final result (in degrees):
	equitorial = np.zeros(galactic.shape)
	if galactic.ndim > 1:
		equitorial[:,0] = alpha*(180.0/np.pi)
		equitorial[:,1] = delta*(180.0/np.pi)
	else:
		equitorial[0] = alpha*(180.0/np.pi)
		equitorial[1] = delta*(180.0/np.pi)
	return equitorial

# Inverse of the galactic_to_equitorial function:
def equitorial_to_galactic(equitorial):
	# Ascension and Declination of the North Galactic Pole in radians):
	aNGP = 192.85948*np.pi/180.0
	dNGP = 27.12825*np.pi/180.0
	# And the north celestial pole in galactic longitude:
	lNCP = 122.93192*np.pi/180.0
	# Equitorial co-ordinates in radians:
	if equitorial.ndim > 1:
		a = equitorial[:,0]*np.pi/180.0
		d = equitorial[:,1]*np.pi/180.0
	else:
		a = equitorial[0]*np.pi/180.0
		d = equitorial[1]*np.pi/180.0
	# Intermediates:
	sindNGP = np.sin(dNGP)
	cosdNGP = np.cos(dNGP)
	sind = np.sin(d)
	cosd = np.cos(d)
	sinap = np.sin(a - aNGP)
	cosap = np.cos(a - aNGP)
	# Sin and cos of galactic co-ordinate parameters:
	sinb = sindNGP*sind + cosdNGP*cosd*cosap
	sinlp = cosd*sinap
	coslp = cosdNGP*sind - sindNGP*cosd*cosap
	negrange = np.where(sinlp < 0)
	lp = (np.arccos(coslp))
	if equitorial.ndim > 1:
		lp[negrange] = (2.0*np.pi - lp[negrange])
	elif sinlp < 0.0:
		lp = 2.0*np.pi - lp
	# Get the longitude:
	l = np.mod(lNCP - lp,2.0*np.pi)
	# Latitude (already correct range):
	b = np.arcsin(sinb)
	# Final result (in degrees):	
	galactic = np.zeros(equitorial.shape)
	if galactic.ndim > 1:
		galactic[:,0] = l*(180.0/np.pi)
		galactic[:,1] = b*(180.0/np.pi)
	else:
		galactic[0] = l*(180.0/np.pi)
		galactic[1] = b*(180.0/np.pi)
	return galactic


# Matrix for converting between supergalactic and galactic co-ordinates:
def sgl_gal_matrix(lx,lz,bx,bz):
	coslx = np.cos(lx)
	coslz = np.cos(lz)
	cosbx = np.cos(bx)
	cosbz = np.cos(bz)
	sinlx = np.sin(lx)
	sinlz = np.sin(lz)
	sinbx = np.sin(bx)
	sinbz = np.sin(bz)
	return np.array([[coslx*cosbx,sinlz*cosbz*sinbx-sinbz*sinlx*cosbx,coslz*cosbz],[sinlx*cosbx,sinbz*coslx*cosbx - coslz*cosbz*sinbx,sinlz*cosbz],[sinbx,cosbz*cosbx*np.sin(lx - lz),sinbz]])

# Converts a 3d position vector in galactic co-ordinates into a 3d position vector in supergalactic co-ordinates.
def galactic_to_supergalactic(galactic_pos):
	M = sgl_gal_matrix(137.37*np.pi/180.0,47.37*np.pi/180.0,0.0,bz = 6.32*np.pi/180.0)
	Mi = np.linalg.inv(M)
	return (Mi.dot(galactic_pos.T)).T

# Inverse - convert supergalactic to galactic co-ordinates:
def supergalactic_to_galactic(sgl_pos):
	M = sgl_gal_matrix(137.37*np.pi/180.0,47.37*np.pi/180.0,0.0,bz = 6.32*np.pi/180.0)
	return (M.dot(sgl_pos.T)).T
	
	
	
# Converts observed helio-centric redshifts into redshifts corrected using the velocity of the local group.
def local_group_z_correction(z_helio,b,l):
	c = 299792458
	cz = c*z_helio/1000 # in km/2
	vcorr = cz - 79.0*np.cos(l*np.pi/180)*np.cos(b*np.pi/180) + 296.0*np.sin(l*np.pi/180)*np.cos(b*np.pi/180) - 36.0*np.sin(b*np.pi/180)
	return vcorr*1000/c

# Quickly convert row vector to column vectors:
def row2col(row):
	return np.reshape(row,(len(row),1))

	
