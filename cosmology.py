# Functions related to cosmological parameters
import numpy as np
import scipy.integrate as integrate
import hmf, astropy
#from .import plot
from scipy import interpolate
import camb

# Linear growth factor as a function of z:
def fLinear(z,Om,Ol):
	return ((Om*(1 + z)**3)/(Om*(1+z)**3 + (1 - Om - Ol)*(1 + z)**2 + Ol))**(4/7)

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
def TMF_from_hmf(Mmin,Mmax,dlog10m=0.01,h = 0.677,Tcmb0 = 2.725,Om0=0.307,Ob0 = 0.0486,returnObjects = False,sigma8 = 0.8159,delta_wrt='SOCritical',Delta=500,z=0):
	cosmo = astropy.cosmology.FlatLambdaCDM(H0 = 100*h,Om0 = Om0, Tcmb0 = Tcmb0, Ob0 = Ob0)
	tmf = hmf.hmf.MassFunction(Mmin=np.log10(Mmin),Mmax=np.log10(Mmax),hmf_model=hmf.fitting_functions.Tinker08,cosmo_model=cosmo,sigma_8 = sigma8,mdef_model=delta_wrt,z=z)
	if returnObjects:
		return [tmf.dndm,tmf.m,tmf,cosmo]
	else:
		return [tmf.dndm,tmf.m]

def PSMF(Mmin,Mmax,dlog10m=0.01,h = 0.677,Tcmb0 = 2.725,Om0=0.307,Ob0 = 0.0486,returnObjects = False,sigma8 = 0.8159,delta_c=None):
	cosmo = astropy.cosmology.FlatLambdaCDM(H0 = 100*h,Om0 = Om0, Tcmb0 = Tcmb0, Ob0 = Ob0)
	tmf = hmf.hmf.MassFunction(Mmin=np.log10(Mmin),Mmax=np.log10(Mmax),hmf_model=hmf.fitting_functions.PS,cosmo_model=cosmo,sigma_8 = sigma8,delta_wrt='crit',delta_c=delta_c)
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
	angular = (sa*(np.pi/180.0)**2)
	prefactor = (c/H0)**3
	return prefactor*angular*integrate.quad(lambda z: drCo(z,Omegam,Omegak,Omegar)**2/(E(z,Omegam,Omegak,Omegar)),zlow,zhigh)[0]


# Function to get the linear side length of a cube equal to a given survey volume, in GPc:
def Lvol(z,sa):
	return np.cbrt(vol(z,sa))/1000.0

#Cosmological growth:
def E(z,Omegam,Omegak,Omegar):
	OmegaL = 1 - Omegam - Omegak - Omegar
	return np.sqrt(Omegar*(1+z)**4 + Omegam*(1+z)**3 + Omegak*(1 + z)**2 + OmegaL)

#Comoving distance (without the c/H0 prefactor)
def drCo(z,Omegam,Omegak=0,Omegar=0):
	return integrate.quad(lambda zp: 1/E(zp,Omegam,Omegak,Omegar),0,z)[0]


def fTinker(sigma,A,a,b,c):
	return A*( (sigma/b)**(-a) + 1 )*np.exp(-c/(sigma**2))
# Tinker, self implemented:
def tmf(M,ki,Pki,rhom,Omegam,deltaType='crit',Delta=200,z=0,A = None,a = None,b = None,c = None):
	# TMF is calibrated using the mean matter density, not the critical density. If the Delta we supply is actually relative to the critical density, we have to adjust it:
	# rhom and Omegam are assumed to be the values TODAY (z = 0)
	if deltaType == 'crit':
		Omegamz = Omegam*(1 + z)**3/(Omegam*(1 + z)**3 + (1 - Omegam))# Assuming matter and DM only
		rhocrit = rhom*(1 + z)**3/(Omegamz)
		Delta = (rhocrit/rhom)*Delta

	# Determine parameters:
	DeltaList = np.array([200,300,400,600,800,1200,1600,2400,3200])
	if A is None:
		AList = np.array([0.186,0.200,0.212,0.218,0.248,0.255,0.260,0.260,0.260])
		select = np.where(Delta == DeltaList)
		if len(select[0]) == 0:
			interpolator = interpolate.interp1d(DeltaList,AList)
			A = interpolator(Delta)
		else:
			A = AList[select[0]][0]

	if a is None:
		aList = np.array([1.47,1.52,1.56,1.61,1.87,2.13,2.30,2.53,2.66])
		select = np.where(Delta == DeltaList)
		if len(select[0]) == 0:
			interpolator = interpolate.interp1d(DeltaList,aList)
			a = interpolator(Delta)
		else:
			a = aList[select[0]][0]

	if b is None:
		bList = np.array([2.57,2.25,2.05,1.87,1.59,1.51,1.46,1.44,1.41])
		select = np.where(Delta == DeltaList)
		if len(select[0]) == 0:
			interpolator = interpolate.interp1d(DeltaList,bList)
			b = interpolator(Delta)
		else:
			b = bList[select[0]][0]

	if c is None:
		cList = np.array([1.19,1.27,1.34,1.45,1.58,1.80,1.97,2.24,2.44])
		select = np.where(Delta == DeltaList)
		if len(select[0]) == 0:
			interpolator = interpolate.interp1d(DeltaList,cList)
			c = interpolator(Delta)
		else:
			c = cList[select[0]][0]
	
	gammaf = 4*np.pi/3
	R = windowMtoR(M,rhom,gammaf=gammaf)
	sigma = sigmaRspatialTH(R,ki,Pki)
	#sigma = sigmaTinker(R,Pk,rhom,kmin,kmax)
	#jacobian = jacobian_tinker(sigma,M,rhom,Pk,kmin,kmax)
	jacobian = np.abs(jacobian_spatialTH(sigma,M,rhom,ki,Pki))
	return fTinker(sigma,A,a,b,c)*(rhom/(M**2))*jacobian

# Linear growth factor	
def linearGrowthD(z,Omegam,Omegak = 0.0):
	a = 1.0/(1.0 + z)
	Omegal = 1.0 - Omegam - Omegak
	Hsq = Omegam/(a**3) + Omegal + Omegak/(a**2)
	return 2.5*a*(Omegam/(a**3))/( Hsq * ( (Omegam/(Hsq*a**3))**(4.0/7.0) - Omegal/Hsq + (1.0 + 0.5*Omegam/(Hsq*a**3))*(1 + Omegal/(70.0*Hsq)) ) )

def linearGrowthf(z,Omegam,Omegak = 0.0):
	a = 1.0/(1.0 + z)
	Omegal = 1.0 - Omegam - Omegak
	return ((Omegam/(a**3))/(Omegam/(a**3) + Omegak/(a**2) + Omegal))**(4.0/7.0)

def windowRtoM(R,rhobar,gammaf = 6*np.pi**2):
	# Default is a top hat in frequency space.
	return gammaf*rhobar*R**3

def windowMtoR(M,rhobar,gammaf = 6*np.pi**2):
	# Default is a top hat in frequency space.
	return np.cbrt(M/(gammaf*rhobar))

def W_spatialTH(kR):
	return np.where(kR > 1.4e-6,3.0*(np.sin(kR) - kR*np.cos(kR))/(kR)**3,1.0)

def W_spatialTHp(kR):
	return np.where(kR > 1e-3,(3.0*((kR)**2 - 3.0)*np.sin(kR) + 9.0*kR*np.cos(kR))/((kR)**4),0.0)

# Compute sigma with the frequency space top hat window function:
def sigma2Mtophat(M,Pk,rhobar):
	kmax = np.max(ki)
	kc = 1/windowMtoR(M,rhobar,gammaf=6*np.pi**2)
	if np.isscalar(kc):
		kmin = np.max([np.min(ki),kc])
		sigma2 = integrate.quad(lambda k: Pk(k)*k**2/(2*np.pi**2),kmin,kmax)[0]
	else:
		sigma2 = np.zeros(len(kc))
		for l in range(0,len(kc)):
			kmin = np.max([np.min(ki),kc[l]])
			sigma2[l] = integrate.quad(lambda k: Pk(k)*k**2/(2*np.pi**2),kmin,kmax)[0]
	return sigma2

def sigmaTinker(R,Pk,rhom,kmin,kmax):
	if np.isscalar(R):
		return integrate.quad(lambda k: Pk(k)*W_spatialTH(k*R)*k**2,kmin,kmax)[0]
	else:
		sigma = np.zeros(len(R))
		for l in range(0,len(R)):
			sigma[l] = integrate.quad(lambda k: Pk(k)*W_spatialTH(k*R[l])*k**2,kmin,kmax)[0]
		return sigma

def jacobian_tinker(sigma,M,rhobar,Pk,kmin,kmax):
	R = windowMtoR(M,rhobar,gammaf = 4.0*np.pi/3.0)
	if np.isscalar(R):
		return -(R/(3.0*sigma))*integrate.quad(lambda k: Pk(k)*W_spatialTHp(k*R)*k**3,kmin,kmax)
	else:
		result = np.zeros(len(R))
		for l in range(0,len(R)):
			result[l] = -(R[l]/(3.0*sigma[l]))*integrate.quad(lambda k: Pk(k)*W_spatialTHp(k*R[l])*k**3,kmin,kmax)[0]
		return result

# Compute sigma of R in the spatial top hat window:
def sigmaRspatialTH(R,ki,Pki,order=0):
	rk = np.outer(R,ki)
	dlnk = np.log(ki[1] / ki[0])
	rest = Pki * ki ** (3 + order * 2)
	integ = rest*W_spatialTH(rk)**2
	return np.sqrt(integrate.simps(integ,dx=dlnk,axis=-1)/(2*np.pi**2))

# SVdW model:
def f_SVdW(sigma,deltav,deltac):
	# Approximation suggested in arxiv 1304.6087:
	D = np.abs(deltav)/(deltac + np.abs(deltav))
	x = D*sigma/np.abs(deltav)
	if np.isscalar(x):
		if x <= 0.276:
			return np.sqrt(2.0/np.pi)*(deltav/sigma)*np.exp(-deltav**2/(2*sigma**2))
		else:
			j = np.array(range(1,5))
			summand = 2*j*np.pi*x**2*np.sin(j*np.pi*D)*exp(-(j*np.pi*x)**2/2)
			return np.sum(summand)
	else:
		if x <= 0.276:
			return np.sqrt(2.0/np.pi)*(deltav/sigma)*np.exp(-deltav**2/(2*sigma**2))
		else:
			j = np.array(range(1,5))
			summand = np.outer(2*j*np.pi*np.sin(j*np.pi*D),x**2)*np.exp(-np.outer(j*np.pi,x)**2/2.0)
			return np.sum(summand,0)

# Compute dln(sigma^-1)/dlnM (sharp k-space filter):
def jacobian_SVdW(sigma,M,rhobar,Pk):
	kc = np.cbrt(6*np.pi**2*rhobar/M)
	return np.abs(-rhobar*Pk(kc)/(2*sigma**2*M))

# Compute dln(sigma^-1)/dlnM (real top hat filter):
def jacobian_spatialTH(sigma,M,rhobar,ki,Pki,order=0):
	R = windowMtoR(M,rhobar,gammaf = 4.0*np.pi/3.0)
	rk = np.outer(R,ki)
	dlnk = np.log(ki[1] / ki[0])
	W = W_spatialTH(rk)
	Wp = W_spatialTHp(rk)
	return -(R/(6.0*np.pi**2*sigma**2))*integrate.simps(Pki*W*Wp*ki**(4 + 2*order),dx=dlnk,axis=-1)

# Matter density in SI units:
def rhoSI(Om):
	G = 6.67e-11
	pc = 3.0857e16
	Mpc = 1e6*pc
	H0si = 100*1e3/Mpc
	return  Om*(3*H0si**2/(8*np.pi*G))

# Matter density in (Msol/h)/(Mpc/h)^3
def rhoCos(Om):
	Mpc = 1e6*3.0857e16
	Msol = 1.989e30
	return rhoSI(Om)*Mpc**3/Msol


# Compute power spectrum (Currently very indirect - could do this directly with CAMB):
def powerSpectrum(h = 0.67,Om0=0.315568,Ob0 = 0.059235,sigma8=0.830,z = 0,kmin = 1e-4,kmax=2.0,npoints=200,Ok = 0.0,mnu=0.06,ns=0.96,As=2e-9,r=0,tau=0.06,nonLinear=True):
	pars = camb.CAMBparams()
	pars.set_cosmology(H0 = 100.0*h,ombh2=Ob0*h**2,omch2=Om0*h**2,mnu=mnu,omk=Ok,tau=tau)
	pars.InitPower.set_params(As=As, ns=ns, r=r)
	# Set redshift:
	if np.isscalar(z):
		redshifts =  [z]
	else:
		redshifts = z
	pars.set_matter_power(redshifts=redshifts, kmax=kmax)
	if nonLinear:
		pars.NonLinear = camb.model.NonLinear_both
	else:
		pars.NonLinear = camb.model.NonLinear_none
	results = camb.get_results(pars)
	kh, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = npoints)
	return [kh, pk]

# Estimate correlation function from power spectrum:
def pkToxi(x,ki,pki):
	xi = np.zeros(x.shape)
	lnk = np.log(ki)
	kix = np.outer(ki,x)
	sinckx = np.sinc(kix)
	for k in range(0,len(xi)):
		xi[k] = integrate.simps(pki*ki**3*sinckx[:,k],x=lnk)
	return xi

# Linear velocity predicted for a given cumulative spherical density contrast
def vLinear(r,Delta,Om,Ol,z=0):
	f = fLinear(z,Om,Ol)
	a = 1/(1 + z)
	H0 = 100
	H = H0*np.sqrt(Om*(1 + z)**3 + Ol)
	return -(1/3)*f*a*H*r*Delta

# Compute delta:
def deltaCumulative(pairCounts,volLists,nbar):
	return (np.cumsum(pairCounts,axis=1)/np.cumsum(volLists,axis=1))/nbar - 1
