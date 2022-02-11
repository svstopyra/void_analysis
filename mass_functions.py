import camb
import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from void_analysis import cosmology


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

def powerSpectrumFromCamb(z=0,h=0.674,Om0=0.315,Ob0 = 0.0486,sigma8 = 0.811,\
		kmin = 1e-4,kmax = 2,Ok=0,mnu = 0.06,ns=0.96,As=2e-9,\
		r = 0,tau=0.06,npoints=200,neff=3.044,Tcmb0 = 2.725,w0=-1):
	# Setup CAMB for transfer functions:
	cambParams = camb.CAMBparams(DoLensing=False,Want_CMB=False,Want_CMB_lensing=False,\
		WantCls=False,WantDerivedParameters=False)
	cambParams.Transfer.high_precision = False
	cambParams.Transfer.k_per_logint = 0
	cambParams.Transfer.kmax = kmax
	cambParams.set_cosmology(H0 = 100*h,ombh2 = Ob0*h**2,omch2=(Om0 - Ob0)*h**2,\
		mnu=mnu,neutrino_hierarchy="degenerate",omk=Ok,nnu=neff,\
		standard_neutrino_neff=neff,TCMB=Tcmb0)
	cambParams.WantTransfer=True
	cambParams.set_dark_energy(w = w0)
	# Get transfer functions:
	camb_transfers = camb.get_transfer_functions(cambParams)
	T = camb_transfers.get_matter_transfer_data().transfer_data
	T = np.log(T[[0, 6], :, 0])
	lnk = np.linspace(np.log(kmin),np.log(kmax),npoints)
	if lnk[0] < T[0, 0]:
		# 
		start = 0
		for i in range(len(T[0,:]) - 1):
			if abs((T[1,:][i + 1] - T[1,:][i]) / \
					(T[0,:][i + 1] - T[0,:][i])) < 0.0001:
				start = i
				break

		lnT = T[1,:][start:-1]
		lnkout = T[0,:][start:-1]
		lnkout[0] = lnk[0]
	else:
		lnkout = T[0,:]
		lnT = T[1,:]
	lnT -= lnT[0]
	lnt = spline(lnkout,lnT,k=1)(lnk)
	# Construct power spectrum:
	kh = np.exp(lnk)
	pkh = kh**ns*np.exp(lnt)**2
	return [kh,pkh]


# Top hat function, Fourier space:
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

def Whatp(k,R):
	x = k*R
	eps = np.finfo(float).eps
	if np.isscalar(x):
		if np.abs(x) >= eps:
			res = 3.0*(x**2*np.sin(x) - 3*np.sin(x) + 3*x*np.cos(x))/((x)**4)
		else:
			res = - x/5.0
	else:
		normal = np.where(np.abs(x) >= eps)
		small = np.where(np.abs(x) <  eps)
		res = np.zeros(len(x))
		# Main result for normal x:
		res[normal] = 3.0*(x[normal]**2*np.sin(x[normal]) - 3*np.sin(x[normal]) + 3*x*np.cos(x[normal]))/((x[normal])**4)
		# Approximation valid for small x, avoiding problems there:
		res[small] = - x[small]/5.0
	return res


# sigma(R):
def sigmaR(R,kh,pkh):
	if np.isscalar(R):
		integrand = pkh*What(kh,R)**2*kh**2/(2*np.pi**2)
		res = integrate.simpson(integrand,kh)
	else:
		res = np.zeros(len(R))
		for k in range(0,len(R)):
			integrand = pkh*What(kh,R[k])**2*kh**3/(2*np.pi**2) 
			res[k] = integrate.simpson(integrand,np.log(kh))# Integrate in logk
	return np.sqrt(res)

# sigma(M,z): 
def sigmaM(M,kh,pkh,Om0):
	# Compute power spectrum using CAMB:
	
	#[kh,pkh] = powerSpectrumFromCamb(h=h,Om0=Om0,Ob0=Ob0,sigma8=sigma8,z=z,\
	#	kmin=kmin,kmax=kmax,Ok=Ok,mnu=mnu,ns=ns,As=As,\
	#	r=r,tau=tau,nonLinear=nonLinear,npoints=npoints)
	# Scale, R:
	rho = Om0*2.7754e11 # Mean density in h^2Msol/Mpc^3
	R = np.cbrt(3*M/(4*np.pi*rho)) # M expected in Msol/h, R in Mpc/h
	# Compute variance at this R:
	sigmaM = sigmaR(R,kh,pkh)
	return sigmaM
	
def computeDSigma2DM(M,kh,pkh,Om0):
	# Scale at which to average:
	rho = Om0*2.7754e11
	R = np.cbrt(3.0*M/(4.0*np.pi*rho))
	Rp = R/(3.0*M)
	if np.isscalar(M):
		sigmapIntegrand = (kh**4)*pkh*2.0*What(kh,R)*Whatp(kh,R)*Rp/(2*np.pi**2)
		res = integrate.simpson(sigmapIntegrand,np.log(kh))
	else:
		res = np.zeros(len(M))
		for k in range(0,len(M)):
			sigmapIntegrand = (kh**4)*pkh*2.0*What(kh,R[k])*Whatp(kh,R[k])*Rp[k]\
				/(2*np.pi**2)
			res[k] = integrate.simpson(sigmapIntegrand,np.log(kh))
	return res

def Omz(z,Om0,Ok=0,Or=0):
	E2 = Om0*(1+z)**3 + Or*(1 + z)**4 + Ok*(1 + z)**2 + (1 - Om0 - Or -Ok)
	return Om0*(1 + z)**3/E2

# Critical density as a function of z, in units h^2Msol/Mpc^3
def rhoCrit(z,Om0,Ok=0,Or=0):
	E2 = Om0*(1+z)**3 + Or*(1 + z)**4 + Ok*(1 + z)**2 + (1 - Om0 - Or -Ok)
	return 2.7754e11*E2

# Mean density:
def rhoMean(z,Om0,Ok=0,Or=0):
	E2 = Om0*(1+z)**3 + Or*(1 + z)**4 + Ok*(1 + z)**2 + (1 - Om0 - Or -Ok)
	return 2.7754e11*E2*Omz(z,Om0,Ok=Ok,Or=Or)

# Growth factor (Carrol1992 approx):
def dPlus(z,Om0):
	a = 1/(1+z)
	om = Om0*(1+z)**3
	denom = (1-Om0) + om
	OmegaM = om/denom
	OmegaL = (1-Om0)/denom
	coeff = 5.0*OmegaM/(2*(1+z))
	term1 = OmegaM**(4.0/7.0)
	term2 = (1.0 + 0.5*OmegaM)*(1.0 + OmegaL/70.0)
	return coeff/(term1 - OmegaL + term2)

def growthFactor(z,Om0):
	return dPlus(z,Om0)/dPlus(0,Om0)

def toFOFMassLukic(M,z,massDef,Delta,Om0,h,Ok=0):
	# Convert to M200c if not already:
	if massDef == "FOF":
		Mfof = M
	else:
		conc = cosmology.cMBhattacharya(M,z=z,Om=Om0,\
			Ol =1 - Om0,h=h,convertType=massDef,relaxed=False,Ok=Ok)
		if massDef != "critical":
			M200c = cosmology.convertCriticalMass(M,Delta,D2=200,\
				z = z,Om=Om0,Ol=1 - Om0,Ok=Ok,h = h,\
				type1=massDef,type2='critical',relaxed=False,\
				cErrFrac = 0.33,returnError=False)
		else:
			M200c = M
		# Lukic et al. (2009) mass relationship:
		a1 = -0.1374
		a2 = 1.0900
		a3 = 0.9714
		Mfof = M200c*(a1/conc**2 + a2/conc + a3)
	return Mfof

def toFOFMassMore(M,z,massDef,Delta,Om0,h,Ok=0,deltaMore = 80.62):
	# Convert to M200c if not already:
	if massDef == "FOF":
		Mfof = M
	else:
		conc = cosmology.cMBhattacharya(M,z=z,Om=Om0,\
			Ol =1 - Om0,h=h,convertType=massDef,relaxed=False,Ok=Ok)
		Mfof = cosmology.convertCriticalMass(M,Delta,D2=deltaMore,\
			z = z,Om=Om0,Ol=1 - Om0,Ok=Ok,h = h,\
			type1=massDef,type2='critical',relaxed=False,\
			cErrFrac = 0.33,returnError=False)
	return Mfof

def toFOFMass(M,z,massDef,Delta,Om0,h,Ok=0,method="More"):
	methodsList = {"Lukic":toFOFMassLukic,"More":toFOFMassMore}
	if method in methodsList:
		return methodsList[method](M,z,massDef,Delta,Om0,h,Ok=Ok)
	else:
		raise Exception("Unrecognised method.")

def getTinkerParams(halo_density_mean,z=0):
	delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
	defaults = {
			# -- A
			"A_200": 1.858659e-01,"A_300": 1.995973e-01,
			"A_400": 2.115659e-01,"A_600": 2.184113e-01,
			"A_800": 2.480968e-01,"A_1200": 2.546053e-01,
			"A_1600": 2.600000e-01,"A_2400": 2.600000e-01,
			"A_3200": 2.600000e-01,
			# -- a
			"a_200": 1.466904,"a_300": 1.521782,"a_400": 1.559186,
			"a_600": 1.614585,"a_800": 1.869936,"a_1200": 2.128056,
			"a_1600": 2.301275,"a_2400": 2.529241,"a_3200": 2.661983,
			# --- b
			"b_200": 2.571104,"b_300": 2.254217,"b_400": 2.048674,
			"b_600": 1.869559,"b_800": 1.588649,"b_1200": 1.507134,
			"b_1600": 1.464374,"b_2400": 1.436827,"b_3200": 1.405210,
			# --- c
			"c_200": 1.193958,"c_300": 1.270316,"c_400": 1.335191,
			"c_600": 1.446266,"c_800": 1.581345,"c_1200": 1.795050,
			"c_1600": 1.965613,"c_2400": 2.237466,"c_3200": 2.439729,
			# -- others
			"A_exp": 0.14,"a_exp": 0.06}
	# Interpolate if given mean not in this set:
	if halo_density_mean not in delta_virs:
		A_array = np.array([defaults["A_%s" % d] \
			for d in delta_virs])
		a_array = np.array([defaults["a_%s" % d] \
			for d in delta_virs])
		b_array = np.array([defaults["b_%s" % d] \
			for d in delta_virs])
		c_array = np.array([defaults["c_%s" % d] \
			for d in delta_virs])
		A_func = spline(delta_virs, A_array)
		a_func = spline(delta_virs, a_array)
		b_func = spline(delta_virs, b_array)
		c_func = spline(delta_virs, c_array)
		A_0 = A_func(halo_density_mean)
		a_0 = a_func(halo_density_mean)
		b_0 = b_func(halo_density_mean)
		c_0 = c_func(halo_density_mean)
	else:
		A_0 = defaults["A_%s" % (int(halo_density_mean))]
		a_0 = defaults["A_%s" % (int(halo_density_mean))]
		b_0 = defaults["A_%s" % (int(halo_density_mean))]
		c_0 = defaults["A_%s" % (int(halo_density_mean))]
	# Redshift evolution:
	A = A_0 * (1 + z)**(-defaults["A_exp"])
	a = a_0 * (1 + z)**(-defaults["a_exp"])
	alpha = 10**(-((0.75/np.log10(halo_density_mean/75.0))**(1.2)))
	b = b_0*(1 + z)**(-alpha)
	c = c_0
	return [A, a, b, c]

# Number density as a function of M:
def dndm(M,fittingFunction,Delta=200,massDef = 'critical',\
		ffParams=None,z=0,h=0.674,Om0=0.315,Ob0 = 0.0486,sigma8 = 0.811,\
		kmin = 1e-4,kmax = 2,Ok=0,mnu = 0.06,ns=0.9649,As=2.099e-9,r = 0,tau=0.0544,
		nonLinear=False,npoints=200,neff=3.044,Tcmb0 = 2.725,w0=-1,delta_c=1.686,\
		kh = None,pkh=None):
	#[kh,pkh] = powerSpectrum(h=h,Om0=Om0,Ob0=Ob0,sigma8=sigma8,z=z,kmin=kmin,kmax=kmax,\
	#	Ok=Ok,mnu=mnu,ns=ns,As=As,r=r,tau=tau,nonLinear=nonLinear,npoints=npoints)
	if (kh is None) or (pkh is None):
		[kh,pkh] = powerSpectrumFromCamb(h=h,Om0=Om0,Ob0=Ob0,sigma8=sigma8,z=z,\
			kmin=kmin,kmax=kmax,Ok=Ok,mnu=mnu,ns=ns,
			As=As,r=r,tau=tau,npoints=npoints)
	# Normalise power spectrum:
	sigma8un = sigmaR(8,kh,pkh)
	pkhNorm = growthFactor(z,Om0)**2*sigma8**2*pkh/sigma8un**2
	if fittingFunction != "HR4":
		sigma = sigmaM(M,kh,pkhNorm,Om0)
		sigma2p = computeDSigma2DM(M,kh,pkhNorm,Om0)
	if massDef == 'critical':
		halo_density = Delta*rhoCrit(z,Om0,Ok=Ok)
	elif massDef == 'mean':
		halo_density = Delta*rhoMean(z,Om0,Ok=Ok)
	elif massDef == "FOF":
		halo_density = Delta*rhoMean(z,Om0,Ok=Ok) # Not really used in any case.
	else:
		raise Exception("Unrecognised massDef")
	halo_density_mean = halo_density/rhoMean(z,Om0,Ok=Ok)
	if fittingFunction == "Tinker":
		if ffParams is not None:
			A = ffParams[0]
			a = ffParams[1]
			b = ffParams[2]
			c = ffParams[3]
		else:
			# Default parameters for different mean_overdensities:
			[A, a, b, c] = getTinkerParams(halo_density_mean,z=z)
		fsigma = A*((sigma/b)**(-a) + 1)*np.exp(-c/sigma**2)
	elif fittingFunction == "HR4":
		# First convert to FOF mass, using Lukic et al. (2009), if not already there:
		#https://ui.adsabs.harvard.edu/abs/2009ApJ...692..217L/abstract
		# Recompute sigma:
		Mfof = toFOFMass(M,z,massDef,Delta,Om0,h,Ok=Ok)
		sigma = sigmaM(Mfof,kh,pkhNorm,Om0)
		sigma2p = computeDSigma2DM(Mfof,kh,pkhNorm,Om0)
		# HR4 fitting function:
		A = 0.333
		p = 0.807
		q = 0.788
		r = 1.795
		chiB = np.sqrt(q) * delta_c/sigma
		chiSz = 0.09*np.tanh(0.9*z) + 0.01
		phiz = np.exp(-z/10) + 0.025
		chiHR = chiB - chiSz
		fsigma = A*phiz*np.sqrt( 2.0 / np.pi )*(chiHR**r)*\
			(1.0 + chiHR**(-2.0*p))*np.exp( - chiHR**2 / 2.0 )
	else:
		raise Exception("Unrecognised fitting function")
	rhoM = rhoMean(z,Om0,Ok=Ok)
	# Jacobian factor: dln(sigma^-1)/dM = -0.5 dln(sigma^2)/dM/(sigma^2):
	jacobian= -0.5*sigma2p/sigma**2
	return fsigma*(rhoM/M)*jacobian

		
	
















