# Functions for identifying the context of a specified halo:
import numpy as np
import pynbody
import pickle
import astropy

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
	units = h[1]['pos'].units
	boxsize = h[1].properties['boxsize'].ratio(h[1]['pos'].units)
	for k in range(0,len(h)):
		halo_centres[k,:] = periodicCentre(h[k+1],boxsize,units=units)
		halo_masses[k] = np.sum(h[k+1]['mass'])
	if(save_file != 'none'):
		pickle.dump([halo_centres,halo_masses],open(save_file,"wb"))
	return [halo_centres,halo_masses]

# Gets halos in the reversed simulation (sr) and returns a list of their centres in the unreversed simulation.
def void_centres(sn,sr,hr):
	void_centre_list = np.zeros([len(hr),3])
	b = pynbody.bridge.Bridge(sn,sr)
	units = sn['pos'].units
	boxsize = sn.properties['boxsize'].ratio(sn['pos'].units)
	for k in range(0,len(hr)):
		void_centre_list[k,:] = periodicCentre(b(hr[k+1]),boxsize,units=units)
	return void_centre_list

def computePeriodicCentreWeighted(positions,weight,periodicity):
	if np.isscalar(periodicity):
		period = (periodicity,periodicity,periodicity)
	else:
		period = periodicity
	if(len(period) != 3):
		raise Exception("Periodicity must be a length 3 vector or a scalar.")
	# Map everything into angles so that we can properly account for how close particles are:
	theta = np.zeros((len(positions),3))
	theta[:,0] = (positions[:,0])*2.0*np.pi/period[0]
	theta[:,1] = (positions[:,1])*2.0*np.pi/period[1]
	theta[:,2] = (positions[:,2])*2.0*np.pi/period[2]
	M = np.sum(weight)
	xi = np.cos(theta)
	zeta = np.sin(theta)
	# Angular averages:
	xibar = np.sum(weight[:,None]*xi,0)/M
	zetabar = np.sum(weight[:,None]*zeta,0)/M
	# Back to theta:
	thetabar = np.arctan2(-zetabar,-xibar) + np.pi
	return (period*thetabar/(2.0*np.pi))

# Get the centre of a system of points each with some weight (eg mass, volume), taking into account periodicity:
def periodicCentreWeighted(snap,weight,periodicity,units = pynbody.units.Unit("Mpc a h**-1")):
	# Map everything into angles so that we can properly account for how close particles are:
	positions = snap['pos'].in_units(units)
	centre = computePeriodicCentreWeighted(positions,weight,periodicity)*units
	return centre

# Centre of mass, taking into account periodicity:
def periodicCentre(snap,periodicity,units = pynbody.units.Unit("Mpc a h**-1")):
	return periodicCentreWeighted(snap,snap['mass'],periodicity,units = units)
	
	
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
def halos_in_sphere(h,radius,centre,halo_centres=None):
	# Generate halo_centres and masses if these are not given:
	if(halo_centres == None):
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
def voids_in_sphere(hr,radius,centre,sn,sr,void_centre_list=None):
	if(void_centre_list == None):
		void_centre_list = void_centres(sn,sr,hr)
	r = np.sqrt(np.sum((void_centre_list - centre)**2,1))
	in_sphere = np.where(r < radius)
	collect = np.zeros(len(hr))
	collect[in_sphere[0]] = 1
	filt = pynbody.filt.Sphere(radius,centre)
	b = pynbody.bridge.Bridge(sn,sr)
	for k in range(0,len(in_sphere[0])):
		if (len(b(hr[in_sphere[0][k]+1])[filt]) < len(hr[in_sphere[0][k]+1])/2):
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

# Converts equatorial co-ordinate data into 3d positions in kpc/h:
def position3d(equatorial,distType="redshift",distUnit = 3.086e+19,H0 = 3.240440699935191e-18,c = 299792458):
	# equatorial - N x 3 matrix, with first column the right ascension, second column the declension, and 3rd column the observed redshift.
	# Distance in kpc, using Hubble's law:
	pos = np.zeros(equatorial.shape)
	if(equatorial.ndim > 1):
		if distType == "redshift":
			D = c*equatorial[:,2]/(H0*distUnit)
		elif distType == "velocity":
			D = equatorial[:,2]/(H0*distUnit)
		else:
			D = equatorial[:,2]
		# Angular co-ordinates:
		phi = equatorial[:,0]*np.pi/180.0
		theta = (90.0 - equatorial[:,1])*np.pi/180.0
		pos[:,0] = D*np.sin(theta)*np.cos(phi)
		pos[:,1] = D*np.sin(theta)*np.sin(phi)
		pos[:,2] = D*np.cos(theta)
	else:
		if distType == "redshift":
			D = c*equatorial[:,2]/(H0*distUnit)
		elif distType == "velocity":
			D = equatorial[:,2]/(H0*distUnit)
		else:
			D = equatorial[:,2]
		# Angular co-ordinates:
		phi = equatorial[0]*np.pi/180.0
		theta = (90.0 - equatorial[1])*np.pi/180.0
		pos[0] = D*np.sin(theta)*np.cos(phi)
		pos[1] = D*np.sin(theta)*np.sin(phi)
		pos[2] = D*np.cos(theta)
	return pos

# Converts galactic co-ordinates into equatorial co-ordinates:
def galactic_to_equatorial(galactic):
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
	cosd = np.sqrt(1.0 - sind**2) # No sign, as declination is always between -pi/2 and pi/2
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
	equatorial = np.zeros(galactic.shape)
	if galactic.ndim > 1:
		equatorial[:,0] = alpha*(180.0/np.pi)
		equatorial[:,1] = delta*(180.0/np.pi)
	else:
		equatorial[0] = alpha*(180.0/np.pi)
		equatorial[1] = delta*(180.0/np.pi)
	return equatorial

# Inverse of the galactic_to_equatorial function:
def equatorial_to_galactic(equatorial):
	# Ascension and Declination of the North Galactic Pole in radians):
	aNGP = 192.85948*np.pi/180.0
	dNGP = 27.12825*np.pi/180.0
	# And the north celestial pole in galactic longitude:
	lNCP = 122.93192*np.pi/180.0
	# Equitorial co-ordinates in radians:
	if equatorial.ndim > 1:
		a = equatorial[:,0]*np.pi/180.0
		d = equatorial[:,1]*np.pi/180.0
	else:
		a = equatorial[0]*np.pi/180.0
		d = equatorial[1]*np.pi/180.0
	# Intermediates:
	sindNGP = np.sin(dNGP)
	cosdNGP = np.cos(dNGP)
	sind = np.sin(d)
	cosd = np.cos(d)
	sinap = np.sin(a - aNGP)
	cosap = np.cos(a - aNGP)
	# Sin and cos of galactic co-ordinate parameters:
	sinb = sindNGP*sind + cosdNGP*cosd*cosap
	# Galactic Latitude (already correct range):
	b = np.arcsin(sinb)
	cosb = np.cos(b)
	# Galactic Longitude (have to account for sign of sin(lp)):
	sinlp = cosd*sinap/cosb
	coslp = (cosdNGP*sind - sindNGP*cosd*cosap)/cosb
	negrange = np.where(sinlp < 0)
	lp = (np.arccos(coslp))
	if equatorial.ndim > 1:
		lp[negrange] = (2.0*np.pi - lp[negrange])
	elif sinlp < 0.0:
		lp = 2.0*np.pi - lp
	# Get the longitude in the right range:
	l = np.mod(lNCP - lp,2.0*np.pi)
	# Final result (in degrees):	
	galactic = np.zeros(equatorial.shape)
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
# Angular co-ordinate conversion, galactic to supergalactic:
def gal2SG(galactic):
	if galactic.ndim > 1:	
		cosl = np.cos(galactic[:,0]*np.pi/180.0).reshape((len(galactic[:,0]),1))
		sinl = np.sin(galactic[:,0]*np.pi/180.0).reshape((len(galactic[:,0]),1))
		cosb = np.cos(galactic[:,1]*np.pi/180.0).reshape((len(galactic[:,0]),1))
		sinb = np.sin(galactic[:,1]*np.pi/180.0).reshape((len(galactic[:,0]),1))
	else:
		cosl = np.cos(galactic[0]*np.pi/180.0)
		sinl = np.sin(galactic[0]*np.pi/180.0)
		cosb = np.cos(galactic[1]*np.pi/180.0)
		sinb = np.sin(galactic[1]*np.pi/180.0)
	# Unit vector in galactic co-ordinates:	
	Xgal = np.hstack((cosb*cosl,cosb*sinl,sinb))
	# Rotate to unit vector in sgl co-ordinates:
	Xsgl = galactic_to_supergalactic(Xgal)
	# Obtain sin SGB:
	sinSGB = Xsgl[:,2] if (galactic.ndim > 1) else Xsgl[2]
	SGB = np.arcsin(sinSGB) # Already correct range: (-pi/2,pi/2)
	cosSGB = np.cos(SGB)
	cosSGL = Xsgl[:,0]/cosSGB if (galactic.ndim > 1) else Xsgl[0]/cosSGB
	sinSGL = Xsgl[:,1]/cosSGB if (galactic.ndim > 1) else Xsgl[1]/cosSGB
	SGL = np.arccos(cosSGL)
	if(galactic.ndim > 1):
		neg = np.where(sinSGL < 0.0)
		SGL[neg] = 2.0*np.pi - SGL[neg]
	elif sinSGL < 0.0:
		SGL = 2.0*np.pi - SGL		
	return np.hstack((SGL.reshape((len(SGL),1)),SGB.reshape((len(SGB),1))))*(180.0/np.pi)

# Angular co-ordinate conversion, galactic to supergalactic:
def sg2gal(sgl):
	if sgl.ndim > 1:	
		cosSGL = np.cos(sgl[:,0]*np.pi/180.0).reshape((len(sgl[:,0]),1))
		sinSGL = np.sin(sgl[:,0]*np.pi/180.0).reshape((len(sgl[:,0]),1))
		cosSGB = np.cos(sgl[:,1]*np.pi/180.0).reshape((len(sgl[:,0]),1))
		sinSGB = np.sin(sgl[:,1]*np.pi/180.0).reshape((len(sgl[:,0]),1))
	else:
		cosSGL = np.cos(sgl[0]*np.pi/180.0)
		sinSGL = np.sin(sgl[0]*np.pi/180.0)
		cosSGB = np.cos(sgl[1]*np.pi/180.0)
		sinSGB = np.sin(sgl[1]*np.pi/180.0)
	# Unit vector in galactic co-ordinates:	
	Xsgl = np.hstack((cosSGB*cosSGL,cosSGB*sinSGL,sinSGB))
	# Rotate to unit vector in sgl co-ordinates:
	Xgal = supergalactic_to_galactic(Xsgl)
	# Obtain sin b:
	sinb = Xgal[:,2] if (sgl.ndim > 1) else Xgal[2]
	b = np.arcsin(sinb) # Already correct range: (-pi/2,pi/2)
	cosb = np.cos(b)
	cosl = Xgal[:,0]/cosb if (sgl.ndim > 1) else Xgal[0]/cosb
	sinl = Xgal[:,1]/cosb if (sgl.ndim > 1) else Xgal[1]/cosb
	l = np.arccos(cosl)
	if(sgl.ndim > 1):
		neg = np.where(sinl < 0.0)
		l[neg] = 2.0*np.pi - l[neg]
	elif sinl < 0.0:
		l = 2.0*np.pi - l		
	return np.hstack((l.reshape((len(l),1)),b.reshape((len(b),1))))*(180.0/np.pi)	
	
# Converts observed helio-centric redshifts into redshifts corrected using the velocity of the local group.
def local_group_z_correction(z_helio,b,l):
	c = 299792458
	cz = c*z_helio/1000 # in km/2
	vcorr = cz - 79.0*np.cos(l*np.pi/180)*np.cos(b*np.pi/180) + 296.0*np.sin(l*np.pi/180)*np.cos(b*np.pi/180) - 36.0*np.sin(b*np.pi/180)
	return vcorr*1000/c

# Converts supergalactic angular positions into supergalactic positions:
def supergalactic_ang_to_pos(ang):
	# Assume Nx3 matrix, with first column radius, second column sgl, third column sgb
	# Convert to polar co-ordinates in radians:
	r = ang[:,0]
	theta = (90.0 - ang[:,2])*np.pi/180.0
	phi = ang[:,1]*np.pi/180.0
	SGX = r*np.sin(theta)*np.cos(phi)
	SGY = r*np.sin(theta)*np.sin(phi)
	SGZ = r*np.cos(theta)
	return np.hstack([SGX.reshape((len(SGX),1)),SGY.reshape((len(SGY),1)),SGZ.reshape((len(SGZ),1))])

# Quickly convert row vector to column vectors:
def row2col(row):
	return np.reshape(row,(len(row),1))

# Return the mean distance of halo h from the specified centre
def mean_distance(h,centre=(0,0,0)):
	return np.mean(np.sqrt(np.sum((h['pos'] - centre)**2,1)))

# Returns the distance of the list of positions, pos, from centre
def distance(pos,centre = (0,0,0)):
	if(pos.ndim == 1):
		return np.sqrt(np.sum((pos - centre)**2))
	else:
		return np.sqrt(np.sum((pos - centre)**2,1))

# Return the distances of each halo from the specified centre:
def halo_distances(hlist,centre = (0,0,0)):
	dist = np.zeros(len(hlist))
	for k in range(0,len(hlist)):
		dist[k] = mean_distance(hlist[k+1],centre)
	return dist

def void_distances(hr,b,centre = (0,0,0)):
	dist = np.zeros(len(hr))
	for k in range(0,len(hr)):
		dist[k] = mean_distance(b(hr[k+1]),centre)
	return dist

# Creates a list of points from the union of many halos:
def snapunion_positions(halolist,to_use):
	totals = np.zeros(len(to_use),dtype='int')
	for k in range(0,len(to_use)):
		# In most cases, we are using a halo catalogue, which is offset by one
		# from the indices:
		totals[k] = len(halolist[to_use[k] + 1])
	cumsum = np.cumsum(totals)
	total = np.sum(totals)
	# Pre-allocate array:
	pos = np.zeros((total,3))
	# Populate:
	pos[0:cumsum[0],:] = halolist[to_use[0] + 1]['pos']
	for k in range(1,len(to_use)):
		pos[cumsum[k-1]:cumsum[k],:] = halolist[to_use[k] + 1]['pos']
	return pos

# Return a sphere of particles extracted from s, at the specified location:
def select_sphere(s,radius,distance,direction,offset=(0,0,0)):
	dir_norm = direction/np.sqrt(np.sum(direction**2))
	filt = pynbody.filt.Sphere(radius,distance*dir_norm + offset)
	return s[filt]

# Return the halos that the selection has particles from (offset by downwards)
def get_containing_halos(snap,halos):
	particle_count = np.zeros(len(halos),dtype='int')
	for k in range(0,len(halos)):
		particle_count[k] = len(halos[k+1].intersect(snap))
	
	containing_halos = np.where(particle_count > 0)
	particles = particle_count[containing_halos]
	sortOrder = np.argsort(-particles)
	return [containing_halos[0][sortOrder],particles[sortOrder]]

# Combine specified halos into a single subsnap:
def combineHalos(snap,halos,to_include = None):
	if to_include is None:
		to_include = range(0,len(halos))
	lengths = np.zeros(len(to_include),dtype=int)
	ntotal = 0
	for k in range(0,len(to_include)):
		lengths[k] = len(halos[to_include[k]+1])
	ntotal = np.sum(lengths)
	ind = np.zeros(ntotal,dtype=int)
	counter = 0
	for k in range(0,len(to_include)):
		ind[counter:(counter + lengths[k])] = halos[to_include[k]+1]['iord']
		counter = counter + lengths[k]
	ind = np.unique(ind)
	return snap[ind]

# Returns true if the pair of specified halos is a viable local group candidate. Assumes kpc
def localGroupTest(n1,n2,halo_centres,halo_masses,centre,testScale=1000):
	com = (halo_centres[n1-1]*halo_masses[n1-1] + halo_centres[n2-1]*halo_masses[n2-1])/(halo_masses[n2-1] + halo_masses[n1-1])
	total_mass = halo_masses[n2-1] + halo_masses[n1-1]
	mSmall = np.min(np.array([halo_masses[n1-1],halo_masses[n2-1]]))
	mLarge = np.max(np.array([halo_masses[n1-1],halo_masses[n2-1]]))
	dist = distance(halo_centres[n1-1] - halo_centres[n2-1])
	mLargeNear = 0
	nearestList = []
	nLargest = 0
	boxDistance = distance(com,centre=centre)
	for k in range(0,len(halo_masses)):
		if ((k+1 == n1) | (k+1 == n2)):
			continue
		if (distance(hn_centres[k] - com) < 2.5*testScale):
			nearestList.append(k)
			if (halo_masses[k] > mLargeNear):
				mLargeNear = halo_masses[k]
				nLargest = k + 1
	# 5 tests:
	test1 = (total_mass < 500)
	test2 = (mSmall > 50)
	test3 = ((dist > 0.3*testScale) & (dist < 1.5*testScale))
	test4 = (mLargeNear  < mSmall)
	test5 = boxDistance < 5*testScale
	if(test1):
		print("Total mass ok for local group: " + str(total_mass*1e10) + " M_sol/h.")
	else:
		print("TEST FAILED: Total mass too large for local group halos: " + str(total_mass*1e10) + " M_sol/h. > " + str(500*1e10) + "M_sol/h.")
	if(test2):
		print("Smallest halo mass ok for local group: " + str(mSmall*1e10) + " M_sol/h.")
	else:
		print("TEST FAILED: Smallest halo too large for local group:" + str(mSmall*1e10) + " M_sol/h.")
	if(test3):
		print("Separation ok for local group: " + str(dist/testScale) + " Mpc/h.")
	else:
		print("TEST FAILED: Halo separation not in valid range for local group: " + str(dist/testScale) + " Mpc/h")
	if(test4):
		print("No large halos within 2.5 Mpc/h.")
	else:
		print("TEST FAILED: halo number " + str(nLargest) +  " with mass " + str(mLargeNear*1e10) + " M_sol/h exists at distance " + str(distance(hn_centres[nLargest-1] - com)/testScale) + " from centre of mass of halo pair.")
	if(test5):
		print("Halo pair centre is within 5Mpc of the box centres.")
	else:
		print("TEST FAILED: Halo pair centre is " + str(boxDistance/testScale) + " Mpc/h from box centre.")
	return [test1 & test2 & test3 & test4 & test5, nearestList]
	
# Estimate volume of a set of intersecting spheres by the monte carlo method:
def spheresMonteCarlo(centres,radii,boundBox,tol=1e-3,count_max=100,nRand=None,nConv = 3):
	boundVolume = boundBox[0]*boundBox[1]*boundBox[2]
	counter = 0
	diffRatio = 1
	totalRands = 0
	totalIncluded = 0
	fracGuess = 0.5
	inARow = 0
	# Use a binomial distribution confidence interval to estimate how many randoms we need:
	if nRand is None:
		nRand = np.ceil(fracGuess*(1 - fracGuess)/(tol**2)).astype(int)
	
	while (counter < count_max):
		rand = np.random.rand(nRand,3)
		for j in range(0,3):
			rand[:,j] = rand[:,j]*boundBox[j]
		included = np.zeros(nRand,dtype=int)
		for k in range(0,len(radii)):
			inSphere = np.where(np.sqrt(np.sum((rand - centres[k,:])**2,1)) <\
				radii[k])[0]
			included[inSphere] = 1
		noIncluded = np.sum(included)
		totalIncluded = totalIncluded + noIncluded
		totalRands = totalRands + nRand
		frac = totalIncluded/totalRands
		diffRatio = np.abs(fracGuess - frac)/frac
		fracGuess = frac
		sigma = np.sqrt(fracGuess*(1 - fracGuess)/totalIncluded)
		print("fracGuess = " + str(frac))
		print(counter)
		if sigma < tol:
			break
		else:
			nRand = np.ceil(fracGuess*(1 - fracGuess)/(tol**2)).astype(int) - nRand
			if nRand < 0:
				break
		#print(diffRatio)
		#print(frac)
		counter = counter + 1
	if (counter >= count_max):
		print("Warning, failed to converge after " + str(count_max) + " iterations.")
	volume = fracGuess*boundVolume
	return volume

# Wrappers for astropy to convert between different sky-coordinates:
def mapEquatorialSnapshotToGalactic(snap,snapCoord=None):
	h = snap.properties['h']
	snap['pos'].convert_units("Mpc a h**-1")
	if snapCoord is None:
		R = np.sqrt(np.sum(snap['pos']**2,1))
		ra = np.arctan2(snap['pos'][:,1],snap['pos'][:,0])
		dec = np.arcsin(snap['pos'][:,2]/R)
		snapCoord = astropy.coordinates.SkyCoord(ra=ra*astropy.units.rad,
			dec=dec*astropy.units.rad,distance=R*astropy.units.Mpc/h)
	snap['pos'][:,0] = snapCoord.galactic.cartesian.x.value*h
	snap['pos'][:,1] = snapCoord.galactic.cartesian.y.value*h
	snap['pos'][:,2] = snapCoord.galactic.cartesian.z.value*h

def mapGalacticSnapshotToEquatorial(snap,snapCoord=None):
	h = snap.properties['h']
	snap['pos'].convert_units("Mpc a h**-1")
	if snapCoord is None:
		R = np.sqrt(np.sum(snap['pos']**2,1))
		l = np.arctan2(snap['pos'][:,1],snap['pos'][:,0])
		b = np.arcsin(snap['pos'][:,2]/R)
		snapCoord = astropy.coordinates.SkyCoord(l=l*astropy.units.rad,
			b=b*astropy.units.rad,distance=R*astropy.units.Mpc/h)
	snap['pos'][:,0] = snapCoord.icrs.cartesian.x.value*h
	snap['pos'][:,1] = snapCoord.icrs.cartesian.y.value*h
	snap['pos'][:,2] = snapCoord.icrs.cartesian.z.value*h

def mapEquatorialToGalactic(points,h = 0.705):
	R = np.sqrt(np.sum(points**2,1))
	ra = np.arctan2(points[:,1],points[:,0])
	dec = np.arcsin(points[:,2]/R)
	snapCoord = astropy.coordinates.SkyCoord(ra=ra*astropy.units.rad,
		dec=dec*astropy.units.rad,distance=R*astropy.units.Mpc/h)
	pointsGalactic = np.zeros(points.shape)
	pointsGalactic[:,0] = snapCoord.galactic.cartesian.x.value*h
	pointsGalactic[:,1] = snapCoord.galactic.cartesian.y.value*h
	pointsGalactic[:,2] = snapCoord.galactic.cartesian.z.value*h
	return pointsGalactic

def mapGalacticToEquatorial(points,h = 0.705):
	R = np.sqrt(np.sum(points**2,1))
	l = np.arctan2(points[:,1],points[:,0])
	b = np.arcsin(points[:,2]/R)
	snapCoord = astropy.coordinates.SkyCoord(l=l*astropy.units.rad,
		b=b*astropy.units.rad,distance=R*astropy.units.Mpc/h)
	pointsEquatorial = np.zeros(points.shape)
	pointsEquatorial[:,0] = snapCoord.icrs.cartesian.x.value*h
	pointsEquatorial[:,1] = snapCoord.icrs.cartesian.y.value*h
	pointsEquatorial[:,2] = snapCoord.icrs.cartesian.z.value*h
	return pointsEquatorial

# Convert Cartesian co-ordinates to RA and DEC in the equatorial system, then generate an 
# astropy SkyCoord object to handle them.
def equatorialXYZToSkyCoord(points,h = 0.705):
	if len(points.shape) == 1:
		R = np.sqrt(np.sum(points**2))
		ra = np.arctan2(points[1],points[0])
		dec = np.arcsin(points[2]/R)
	else:
		R = np.sqrt(np.sum(points**2,1))
		ra = np.arctan2(points[:,1],points[:,0])
		dec = np.arcsin(points[:,2]/R)
	skycoord = astropy.coordinates.SkyCoord(ra=ra*astropy.units.rad,
		dec=dec*astropy.units.rad,distance=R*astropy.units.Mpc/h)
	return skycoord


	
	











