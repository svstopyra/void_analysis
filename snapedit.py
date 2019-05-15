# Python code to edit snapshots
import numpy as np
import pynbody

# Wrap positions so that they lie in the periodic domain:
def wrap(pos,boxsize):
	wrap_up = np.where(pos < 0.0)
	wrap_down = np.where(pos > boxsize)
	wrap = pos[:]*1.0
	wrap[wrap_up] = wrap[wrap_up] + boxsize
	wrap[wrap_down] = wrap[wrap_down] - boxsize
	return wrap

# Unwrap positions so that all points lie within (-boxsize/2,boxsize/2]:
def unwrap(pos,boxsize):
	unwrap = wrap(pos,boxsize)
	large = np.where(unwrap > boxsize/2.0)
	unwrap[large] = unwrap[large] - boxsize
	return unwrap


# Converts positions into the nearest index point on a uniform grid.
# pos - array containing the positions we want to assign to grid points (can be a simarray)
# boxsize - size of the box in the same units as the positions are given
# N - number of cells on one side of the box (generally there are N^3 points)
# lfl_corner - lower front left hand corner of the box (assumed [0,0,0] by default)
def index(pos,boxsize,N,lfl_corner=[0,0,0]):
	ind = np.zeros(pos.shape,dtype='int')
	#ind[:] = (np.floor((pos.in_units("Mpc a h**-1") - lfl_corner)*N/boxsize)).astype(int)
	# Grid centroids are actually offset by half a cell from the lower front left corner, so
	# we have to subtract off this part before rounding to the closest grid.
	ind[:] = (np.round((pos.in_units("Mpc a h**-1") - lfl_corner)*N/boxsize - 0.5)).astype(int)
	return ind

def grid_offset(pos,boxsize,N,lfl=[0,0,0]):
	ind = index(pos,boxsize,N,lfl) # Grid indices for each point (assuming uniform initial grid)
	grid = (ind*boxsize/N) + (boxsize/(2*N)) # positions of centroids for each cell.
	return pos - grid

# Gets the zeldovich approximation factor between position and velocity offsets:
def zeldovich_factor(s,factor=1):
	return factor*100.0*np.sqrt(s.properties['omegaM0']/(s.properties['a']**3) + s.properties['omegaL0'])*np.sqrt(s.properties['a'])

# Assuming zeldovich ICs, try to infer the underlying grid:
def getGridFromZeldovich(s,factor):
	# Unit conversion factor:
	un = pynbody.units.Unit("Mpc**-1 km a**-1/2 s**-1 h" )
	# Zeldovich ratio between position offsets and velocities:
	fac = zeldovich_factor(s,factor)*un
	# Boxsize:
	boxsize = s.properties['boxsize'].ratio("Mpc a h**-1")
	return wrap((s['pos'].in_units("Mpc a h**-1") - (s['vel']/fac)),boxsize)

# Test whether a set of initial conditions uses the zeldovich approximation
def zeldovich_test(s,N,factor=1,tol=1e-4,lfl=[0,0,0],dist_unit="Mpc"):
	
	# Get positions of particles subtracting off what the offsets would be if we were using the zeldovich approximation:
	grid = getGridFromZeldovich(s,factor)
	# Get the closest indices to these points (if the initial conditions are zeldovich, these will be extremely close to the grid points, so this should give something close to grid):
	diff = grid_offset(grid,boxsize,N,lfl)
	diff_dist = np.sqrt(np.sum(diff**2,1))
	diff_ratio = diff_dist*N/(s.properties['boxsize'].in_units("Mpc a h**-1"))
	mean = np.mean(diff_ratio)
	stddev = np.sqrt(np.var(diff_ratio))
	print("Average difference from expected grid point as a fraction of cell size: " + str(mean))
	print("Standard deviation: " + str(stddev))
	# Get number of violating:
	violaters = np.where(np.abs(diff_ratio) > tol)
	if (np.abs(mean) < tol):
		print("ICs appear to be Zeldovich within tolerance of " + str(tol))
		if(len(violaters[0]) > 0):
			print(str(len(violaters[0])) + " particles not sufficiently close to grid points after inversion. Maximum violation is " + str(np.max(diff_ratio[violaters[0]])))
		return True
	else:
		print("ICs do not appear to be Zeldovich.")
		if(len(violaters[0]) > 0):
			print(str(len(violaters[0])) + " particles not sufficiently close to grid points after inversion. Maximum violation is " + str(np.max(diff_ratio[violaters[0]])))
		return False

# Computes the logarithmic derivative of the linear structure growth function, used to get
# peculiar velocities in the Zeldovich approximation:
def f1(Om,Ol,z):
	Omz = Om*(1.0+z)**3/(Om*(1.0+z)**3 + Ol)
	Olz = Ol/(Om*(1+z)**3 + Ol)
	return Omz**0.6 + Olz*(1.0 + Omz/2.0)/70.0

# Indexing co-ordinates on an NxN grid - convert linear to co-ordinate:
def lin2coord(n,N):
	i = np.floor(n/(N**2))
	j = np.floor((n - i*N**2)/N)
	k = n - i*N**2 - j*N
	return np.array([i,j,k],dtype='int')

# Inverse of lin2coord - convert co-ordinates to linear indices:
def coord2lin(coord,N):
	return coord[0]*N**2 + coord[1]*N + coord[2]

# Construct a list of indices that we expect for an NxNxN grid using lin(i,j,k) = N^2*i + N*j + k as the linear index
def gridList(N):
	return gridListPermutation(N)

# Inverted list of indices, where we use lin(i,j,k) = N^2*k + N*j + i as the linear index
def gridListInverted(N):
	return gridListPermutation(N,perm=(2,1,0))

# Arbitrary permutation of i,j,k indices in the definition of the linear index.
def gridListPermutation(N,perm=(0,1,2)):
	if all(np.sort(perm) != np.array([0,1,2])):
		raise Exception("Must supply a permutation of 0,1,2")
	ind = np.zeros((N**3,3),dtype='int')
	ind[:,perm[0]] = np.repeat(range(0,N),N**2)
	ind[:,perm[1]] = np.tile(np.repeat(range(0,N),N),N)
	ind[:,perm[2]] = np.tile(range(0,N),N**2)	
	return ind
	
# Attempt to figure out what type of particle arrangement we have:
def gridTest(s,grid,lfl_corner=[0,0,0]):
	boxsize = s.properties['boxsize'].ratio("Mpc a h**-1")
	n3 = len(s)
	n = np.cbrt(n3)
	N = np.round(n).astype(int)
	if(n - N != 0.0):
		# Suspect grid is not cubic, so probably a zoom simulation
		print("Non-cubic grid. Exact centroids cannot be computed.")
		return -1
	else:
		ind = index(grid,boxsize,N,lfl_corner=lfl_corner)
		# Check format:
		third = np.where(ind[1] == 1)
		second = np.where(ind[N] == 1)
		first = np.where(ind[N**2] == 1)
		if((len(first[0]) != 1) or (len(second[0]) != 1) or (len(third[0]) != 1) ):
			# Nonstandard order:
			print("Warning - grid appears cubic, but non-standard particle order detected. Exact centroids cannot be computed.")
			return -1
		else:
			return (first[0][0],second[0][0],third[0][0])

# Return a grid pointing to the centres of the cells in a given snapshot
def centroid(s,N,units=None,perm=(0,1,2)):
	if(units == None):
		# Inherit the units from the previous snapshot
		units = s['pos'].units
	grid = pynbody.array.SimArray(gridListPermutation(N,perm=perm),"")		
	return (grid*s.properties['boxsize']/N + s.properties['boxsize']/(2*N)).in_units(units)



# Takes a set of initial conditions, reverses them, and then output a new initial conditions file.
def reverseICs(s,filename=None,units=None,fmt=None,inverted=False,factor=1,perm=None):
	if(units == None):
		# Inherit the units from the previous snapshot
		units = s['pos'].units
	if(fmt == None):
		# Inherit snap format from s:
		fmt = type(s)
	centres = getGridFromZeldovich(s,factor)
	if(perm == None):
		# Try to figure out what sort of particle order the file is using, unless this has been specified using perm	
		perm = gridTest(s,centres)
	if(perm != -1):
		# Identified as a specific cubic grid
		N = np.round(np.cbrt(len(s))).astype(int)
		centres = centroid(s,N,perm=perm)
	else:
		# If perm == -1, then we probably have a zoom-grid or other non-cubic grid.
		# We will have to approximate the correct grid centroids assuming the
		# Zeldovich approximation holds.
		# getGridFromZeldovich gives us something in Mpc, so convert back to the original units to ensure consistency:
		centres = centres.in_units(units)
	
	snew = pynbody.snapshot.new(len(s))
	snew['pos'] = wrap(2.0*centres - s['pos'],s.properties['boxsize'])
	snew['vel'] = -s['vel']
	# Other properties should be the same:
	snew.properties = s.properties
	snew['mass'] = s['mass']
	snew['iord'] = s['iord']
	if(filename != None):
		# Only output the snapshot if a filename is specified.
		snew.write(filename=filename,fmt=fmt)
	return snew



	
	
	



	
