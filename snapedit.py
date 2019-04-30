# Python code to edit snapshots
import numpy as np
import pynbody

# Converts positions into the nearest index point on a uniform grid.
# pos - array containing the positions we want to assign to grid points (can be a simarray)
# boxsize - size of the box in the same units as the positions are given
# N - number of cells on one side of the box (generally there are N^3 points)
# lfl_corner - lower front left hand corner of the box (assumed [0,0,0] by default)
def index(pos,boxsize,N,lfl_corner=[0,0,0]):
	ind = np.zeros(pos.shape,dtype='int')
	ind[:] = (np.floor((pos.in_units("Mpc a h**-1") - lfl_corner)*N/boxsize)).astype(int)
	return ind

def grid_offset(pos,boxsize,N,lfl=[0,0,0]):
	ind = index(pos,boxsize,N,lfl) # Grid indices for each point (assuming uniform initial grid)
	grid = (ind*boxsize/N) + (boxsize/(2*N)) # positions of centroids for each cell.
	return pos - grid

# Gets the zeldovich approximation factor between position and velocity offsets:
def zeldovich_factor(s,factor=1):
	return factor*100.0*np.sqrt(s.properties['omegaM0']/(s.properties['a']**3) + s.properties['omegaL0'])*np.sqrt(s.properties['a'])

# Test whether a set of initial conditions uses the zeldovich approximation
def zeldovich_test(s,N,factor=1,tol=1e-4,lfl=[0,0,0],dist_unit="Mpc"):
	# Unit conversion factor:
	un = pynbody.units.Unit("Mpc**-1 km a**-1/2 s**-1 h" )
	# Zeldovich ratio between position offsets and velocities:
	fac = zeldovich_factor(s,factor)*un
	# Boxsize:
	boxsize = s.properties['boxsize'].ratio("Mpc a h**-1")
	
	# Get positions of particles subtracting off what the offsets would be if we were using the zeldovich approximation:
	grid = (s['pos'].in_units("Mpc a h**-1") - (s['vel']/fac))
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


	
