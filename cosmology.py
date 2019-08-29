# Functions related to cosmological parameters
import numpy as np
import scipy.integrate as integrate
import hmf, astropy
from .import plot

# Universal fitting function, for Tinker Mass Function:
def fsigma(sigma,A,a,b,c):
	return A*((sigma/b)**(-a) + 1)*np.exp(-c/(sigma**2))

def interpolateTransfer(k,ki,Tki):
	kmax = np.max(ki)
	if np.isscalar(k):
		if k <= kmax:
			trans = np.interp(k,ki,Tki)
		else:
			keq = 0.01 # Arbitrary scale
			kratio = k/kmax
			trans = (np.log(k/keq)/np.log(kmax/keq))/(kratio**2)
	else:
		trans = np.zeros(len(k))
		inRange = np.where(k <= kmax)
		trans[inRange] = np.interp(k[inRange],ki,Tki)
		belowRange = np.where(k < 0)
		if len(k[belowRange]) > 0:
			raise Exception("Cannot compute power spectrum at negative k.")
		aboveRange = np.where(k > kmax)
		if len(k[aboveRange]) > 0:
			keq = 0.01 # Arbitrary scale
			kratio = k[aboveRange]/kmax
			trans[aboveRange] = (np.log(k[aboveRange]/keq)/np.log(kmax/keq))/(kratio**2)
	return trans/Tki[0]

# Power spectrum interpolation function:
# k - query values
# Pki - power spectrum value computed at ki
# ki - values at which P(k) is known.
# z - redshift at which to compute the power spectrum. Defaults to z = 0
def Pkinterp(k,Tki,ki,ns,amp):
	trans = interpolateTransfer(k,ki,Tki)
	# Currently, we use a user supplied amplitude. However, it can be calculated from:
	# amp = (sigma8*az/(sigma80*a0))**2
	# where sigma8 is the linear variance in a sphere of radius 8Mpc, computed at the input redshift, az is the scale factor at this redshift, sigma80 is the linear variance factor at z = 0, and a0 is the scale factor at z = 0.
	# Fourier transforms may also use:
	# amp = N**3/Vbox where N is the side length of the box, and Vbox its volume.	
	# Compute power spectrum:
	return amp*(k**ns)*(trans**2)

# Fourier transform of the spherical top hat window function:
def What(k,R):
	x = k*R
	eps = np.finfo(float).eps
	if np.isscalar(x):
		if np.abs(x) >= eps:
			res = 3.0*(np.sin(x) - x*np.cos(x))/(x**3)
		else:
			res = 1.0 - x**2/10.0
	else:
		normal = np.where(np.abs(x) >= eps)
		small = np.where(np.abs(x) <  eps)
		res = np.zeros(len(x))
		# Main result:
		res[normal] = 	3.0*(np.sin(x[normal]) - x[normal]*np.cos(x[normal]))/((x[normal])**3)
		# Approximation for small x:
		res[small] = 1.0 - x[small]**2/10.0
	return res

# Derivative with respect to kR:
def Whatp(k,R):
	x = k*R
	eps = np.finfo(float).eps
	if np.isscalar(x):
		if np.abs(x) >= eps:
			res = 3.0*(x**2*np.sin(x) - 3*np.sin(x) + 3*np.cos(x))/((x)**4)
		else:
			res = - x/5.0
	else:
		normal = np.where(np.abs(x) >= eps)
		small = np.where(np.abs(x) <  eps)
		res = np.zeros(len(x))
		# Main result for normal x:
		res[normal] = 3.0*(x[normal]**2*np.sin(x[normal]) - 3*np.sin(x[normal]) + 3*np.cos(x[normal]))/((x[normal])**4)
		# Approximation valid for small x, avoiding problems there:
		res[small] = - x[small]/5.0
	return res

# Estimate linear variance at redshift 0:
def computeSigma80(radius,ns,Tki,ki):
	kmin = ki[0]
	kmax = np.min([np.max(ki),200.0/radius])
	amp = (9.0/(2.0*np.pi**2))
	# Effectively perform a numerical integral:
	s = integrate.quad(lambda k : (k**(ns + 2))*What(k,radius)**2*(interpolateTransfer(k,ki,Tki)**2),kmin,kmax)[0]
	# Otherwise:
	#k = np.linspace(kmin,kmax,intervals+1)
	#trans = interpolateTransfer(k,ki,Tki)
	#Deltak = (kmax - kmin)/intervals
	#var = (k**(ns + 2))*What(k,radius)/(interpolateTransfer(k,ki,Tki)**2)
	#s = np.sum(var)
	#return np.sqrt(s*amp*Deltak)
	return np.sqrt(s*amp)

# Amplitude needed for power spectrum:
def computePkAmplitude(sigma8,z,ki,Tki,ns,sigma80 = None):
	if sigma80 is None:
		sigma80 = computeSigma80(8.0,ns,Tki,ki)
	return ((sigma8/sigma80)/(1.0 + z))**2

# Integrand for sigma integral:
def sigmaIntegrand(k,R,ki,Tki,ns,amp):
	return (k**2)*Pkinterp(k,Tki,ki,ns,amp)*What(k,R)**2

# Integrate to obtain sigma:
def computeSigma(z,M,rhoB,Tki,ki,ns,amp):
	# Scale at which to average:
	R = np.cbrt(3.0*M/(4.0*np.pi*rhoB))
	if np.isscalar(M):
		res = integrate.quad(sigmaIntegrand,0,+np.inf,args=(R,ki,Tki,ns,amp))[0]
	else:
		res = np.zeros(len(M))
		for k in range(0,len(M)):
			res[k] = integrate.quad(sigmaIntegrand,0,+np.inf,args=(R[k],ki,Tki,ns,amp))[0]
	return res

# Integrand for sigma integral:
def sigmapIntegrand(k,R,Rp,ki,Tki,ns,amp):
	return (k**3)*Pkinterp(k,Tki,ki,ns,amp)*2.0*What(k,R)*Whatp(k,R)*Rp

# Integrate to get d sigma/ dM:
def computeDSigmaDM(z,M,rhoB,Tki,ki,ns,amp):
	# Scale at which to average:
	R = np.cbrt(3.0*M/(4.0*np.pi*rhoB))
	Rp = R/(3.0*M)
	if np.isscalar(M):
		res = integrate.quad(sigmapIntegrand,0,+np.inf,args=(R,Rp,ki,Tki,ns,amp))[0]
	else:
		res = np.zeros(len(M))
		for k in range(0,len(M)):
			res[k] = integrate.quad(sigmapIntegrand,0,+np.inf,args=(R[k],Rp[k],ki,Tki,ns,amp))[0]
	return res

# Function to get the halo mass function derivative, dn/dM
def TMF(M,A,a,b,c,z,rhoB,Tki,ki,ns,sigma8):
	amp = computePkAmplitude(sigma8,z,ki,Tki,ns)
	sigma = computeSigma(z,M,rhoB,Tki,ki,ns,amp)
	sigmap = computeDSigmaDM(z,M,rhoB,Tki,ki,ns,amp)
	return fsigma(sigma,A,a,b,c)*(rhoB/M)*(-sigmap/sigma)

# Function to create a mass function using the hmf package, assuming a flat cosmology.
# h - dimensionless Hubble parameter
# Tcmb0 - Temperature of the CMB, in Kelvin
# Om0 - matter density fraction
# Ob0 - baryon density fraction
# Mmin - log10 of minimum mass, in units of Msol
# Mmax - log10 of maximum halo mass, in units of Msol
# dlog10m - step between Mmin and Mmax, in log10 space.
def TMF_from_hmf(Mmin,Mmax,dlog10m=0.01,h = 0.677,Tcmb0 = 2.725,Om0=0.307,Ob0 = 0.0486,returnObjects = False,sigma8 = 0.8159):
	cosmo = astropy.cosmology.FlatLambdaCDM(H0 = 100*h,Om0 = Om0, Tcmb0 = Tcmb0, Ob0 = Ob0)
	tmf = hmf.hmf.MassFunction(Mmin=np.log10(Mmin),Mmax=np.log10(Mmax),hmf_model=hmf.fitting_functions.Tinker08,cosmo_model=cosmo,sigma_8 = sigma8,delta_wrt='crit')
	if returnObjects:
		return [tmf.dndm,tmf.m,tmf,cosmo]
	else:
		return [tmf.dndm,tmf.m]

# Function to convert the number density per unit mass into a number density in mass bins:
def dndm_to_n(m,dndm,massBins):
	n = np.zeros(len(massBins)-1)
	for k in range(0,len(massBins)-1):
		n[k] = integrate.quad(lambda x: np.interp(np.log10(x),np.log10(m),dndm),massBins[k],massBins[k+1])[0]
	return n

# Function to convert redshift and square degrees for a survey volume into a physical volume:
def vol(zlow,zhigh,sa,Omegam,Omegak = 0,Omegar = 0):
	c = 3e5 # Speed of light in km/s
	H0 = 100 # Hubble rate divided by h
	#return (sa*(np.pi/180.0)**2)*((c*z/100)**3)/3
	angular = (sa*(np.pi/180.0)**2)/(4.0*np.pi)
	prefactor = (c/H0)**3
	return prefactor*angular*integrate.quad(lambda x: dADim(x,Omegam,Omegak,Omegar)/(E(x,Omegam,Omegak,Omegar)*(1 + x)),zlow,zhigh)[0]


# Function to get the linear side length of a cube equal to a given survey volume, in GPc:
def Lvol(z,sa):
	return np.cbrt(vol(z,sa))/1000.0

#Cosmological growth:
def E(z,Omegam,Omegak,Omegar):
	OmegaL = 1 - Omegam - Omegak - Omegar
	return np.sqrt(Omegar*(1+z)**4 + Omegam*(1+z)**3 + Omegak*(1 + z)**2 + OmegaL)

#Angular diameter distance (without the c/H0 prefactor)
def dADim(z,Omegam,Omegak=0,Omegar=0):
	return integrate.quad(lambda x: 1/E(z,Omegam,Omegak,Omegar),0,z)[0]


		
		
