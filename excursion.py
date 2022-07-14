# Code for computing excursion set void number densities, and plotting them.
from . import cosmology
from void_analysis.plot import binValues, binCentres
import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate as integrate

# Converting between mass and radius in the excursion set model:
def windowRtoM(R,rhobar,gammaf = 6*np.pi**2):
	# Default is a top hat in frequency space.
	return gammaf*rhobar*R**3

def windowMtoR(M,rhobar,gammaf = 6*np.pi**2):
	# Default is a top hat in frequency space.
	return np.cbrt(M/(gammaf*rhobar))

# Fourier space form of real-space top hat function:
def W_spatialTH(k,R):
	return 3.0*(np.sin(k*R) - k*R*np.cos(k*R))/(k*R)**3

# Compute sigma with the frequency space top hat window function:
def sigma2Mtophat(M,ki,Pki,rhobar,interpMode='linear'):
	kc = 1/windowMtoR(M,rhobar,gammaf=6*np.pi**2)
	Pk = interp1d(ki,Pki,kind=interpMode)
	integrals = np.zeros(len(kc))
	for k in range(0,len(kc)):
		kinew = np.logspace(np.log10(ki[0]),np.min([np.log10(ki[-1]),np.log10(kc[k])]),len(ki))
		pknew = Pk(kinew)
		dlnk = np.log10(kinew[1]/kinew[0])
		integrand = pknew*(kinew**3)/(2*np.pi**2)
		integrals[k] = integrate.simps(integrand,dx=dlnk)
	return integrals

# Compute sigma of R in the spatial top hat window:
def sigma2RspatialTH(R,Pk,kmin,kmax):
	if np.isscalar(R):
		return integrate.quad(lambda k: Pk(k)*W_spatialTH(k,R)**2*k**2/(2*np.pi**2),kmin,kmax)[0]
	else:
		sigma2 = np.zeros(len(R))
		for l in range(0,len(R)):
			sigma2[l] = integrate.quad(lambda k: Pk(k)*W_spatialTH(k,R[l])**2*k**2/(2*np.pi**2),kmin,kmax)[0]
		return sigma2

# SVdW model:
def f_SVdW(sigma,deltav,deltac,nmax=4):
	# Approximation suggested in arxiv 1304.6087:
	D = np.abs(deltav)/(deltac + np.abs(deltav))
	x = D*sigma/np.abs(deltav)
	if np.isscalar(x):
		if x <= 0.276:
			return np.sqrt(2.0/np.pi)*(np.abs(deltav)/sigma)*np.exp(-deltav**2/(2*sigma**2))
		else:
			j = np.array(range(1,5))
			summand = 2*j*np.pi*x**2*np.sin(j*np.pi*D)*exp(-(j*np.pi*x)**2/2)
			return np.sum(summand)
	else:
		result = np.zeros(x.shape)
		smallRegion = np.where(x <= 0.276)
		if len(smallRegion[0]) > 0:
			result[smallRegion] = np.sqrt(2.0/np.pi)*(np.abs(deltav)/sigma[smallRegion])*np.exp(-deltav**2/(2*sigma[smallRegion]**2))
		largeRegion = np.where((x > 0.276))
		if len(largeRegion[0]) > 0:
			j = np.array(range(1,nmax+1))
			summand = np.outer(2*j*np.pi*np.sin(j*np.pi*D),x[largeRegion]**2)*np.exp(-np.outer(j*np.pi,x[largeRegion])**2/2.0)
			result[largeRegion] = np.sum(summand,0)
		return result

# Jacobian factor appearing in SVdW number density
def jacobian_SVdW(sigma,M,rhobar,ki,Pki,interpMode='linear'):
	Pk = interp1d(ki,Pki,kind=interpMode)
	kc = np.cbrt(6*np.pi**2*rhobar/M)
	return np.abs(-rhobar*Pk(kc)/(2*sigma**2*M))

# Number density of voids per unit mass, SVdW model:
def dn_SVdWdlnm(M,ki,Pki,rhom,deltav=-2.7,deltac = 1.686,nmax=4):
	sigma2 = sigma2Mtophat(M,ki,Pki,rhom)
	sigma = np.sqrt(sigma2)
	return (rhom/M)*f_SVdW(sigma,deltav,deltac,nmax=nmax)*np.abs(jacobian_SVdW(sigma,M,rhom,ki,Pki))

# Halo number density, EPS model:
def dn_dlnm_EPS(M,ki,Pki,rhom,deltac = 1.686):
	sigma2 = sigma2Mtophat(M,ki,Pki,rhom)
	sigma = np.sqrt(sigma2)
	return (rhom/M)*np.sqrt(2.0/np.pi)*(deltac/sigma)*np.exp(-deltac**2/(2*sigma**2))*np.abs(jacobian_SVdW(sigma,M,rhom,ki,Pki))


# Get void density in given mass bins, SVdW model:
def voidDensityInBins(massBins,rhom,ki,Pki,deltav = -2.7,deltac = 1.686,intervals=21,nPoints=200,interpMode='linear'):
	massBinCentres = (massBins[1:] + massBins[0:-1])/2
	voidDensity = np.zeros(len(massBinCentres))
	massPoints = np.logspace(np.log10(massBins[0]),np.log10(massBins[-1]),nPoints)
	SVdW_interp = interp1d(massPoints,dn_SVdWdlnm(massPoints,ki,Pki,rhom,deltav=deltav,deltac=deltac),kind=interpMode)
	for k in range(0,len(voidDensity)):
		mRange = np.logspace(np.log10(massBinsLow[k]),np.log10(massBinsLow[k+1]),intervals)
		dlnM = np.log10(mRange[1]/mRange[0])
		voidDensity[k] = integrate.simps(SVdW_interp(mRange),dx=dlnM,axis=-1)
	return voidDensity

# Get halo density in mass bins, EPS model.
def haloDensityInBins(massBins,rhom,ki,Pki,deltac = 1.686,intervals=21,nPoints=200,interpMode='linear'):
	massBinCentres = (massBins[1:] + massBins[0:-1])/2
	haloDensity = np.zeros(len(massBinCentres))
	massPoints = np.logspace(np.log10(massBins[0]),np.log10(massBins[-1]),nPoints)
	epsInterp = interp1d(massPoints,dn_dlnm_EPS(massPoints,ki,Pki,rhom,deltac=deltac),kind=interpMode)
	for k in range(0,len(haloDensity)):
		mRange = np.logspace(np.log10(massBinsLow[k]),np.log10(massBinsLow[k+1]),intervals)
		dlnM = np.log10(mRange[1]/mRange[0])
		haloDensity[k] = integrate.simps(epsInterp(mRange),dx=dlnM,axis=-1)
	return haloDensity

# Output fraction of crushed halos in a given bin:
def crushedRatio(halo_masses,halo_densities,massBins,delta_crush,volSim):
	if np.isscalar(delta_crush):
		# Uniform threshold applied to all:
		nonCrushed = np.where(halo_densities < delta_crush)[0]
	else:
		# Using a mass-varying crushing threshold. This is somewhat more complicated
		condition = np.zeros(halo_densities.shape,dtype=bool)
		if len(delta_crush) != len(massBins) - 1:
			raise Exception('Specified crushing threshold does not match bin list.')
		for k in range(0,len(massBins) - 1):
			condition = condition  | ( (halo_masses*1e10 > massBins[k]) & (halo_masses*1e10 <= massBins[k+1]) & (halo_densities < delta_crush[k]))
		nonCrushed = np.where(condition)		
	[voids_binList,voids_noInBins] = plot.binValues(halo_masses[nonCrushed]*1e10,massBins)
	[halos_binList,halos_noInBins] = plot.binValues(halo_masses*1e10,massBins)
	nzDenom = np.where(halos_noInBins != 0.0)
	ratio = voids_noInBins[nzDenom]/halos_noInBins[nzDenom]
	sigmaRatio = np.sqrt(voids_noInBins[nzDenom]*(halos_noInBins[nzDenom] - voids_noInBins[nzDenom])/halos_noInBins[nzDenom])/halos_noInBins[nzDenom]
	return [ratio,sigmaRatio,nzDenom]

# Expected virialisation density
def deltaVir(Om):
	x = Om - 1.0
	return (18*np.pi**2 + 82*x - 39*x**2)/Om


# Plotting number densities:
def plotMassBinNoDensity(ax,masses,massBins,volume,fmt='x',label='Number density',colour=None):
	# Bin masses:
	[binListData,noInBinsData] = binValues(masses,massBins)
	# Number of samples:
	N = np.sum(noInBinsData)
	# Number density and error estimate:
	noDensity = noInBinsData/volume
	sigmaNoDensity = np.sqrt(noInBinsData*(N - noInBinsData)/N)/volume
	massBinCentres = binCentres(massBins)
	ax.errorbar(massBinCentres,noDensity,yerr=sigmaNoDensity,fmt=fmt,label=label,color=colour)



# Plot tmf:
def plotBinsTMF(ax,massBins,h=0.6726,Om0= 0.315568,sigma8 = 0.830,Ob0 = 0.01,delta_wrt='crit',Delta=500,z=0,tmf=None,tmf_dndm=None,tmf_cosmo=None,tmf_masses=None,fmt='-'):
	if tmf is None:
		[tmf_dndm,tmf_masses,tmf,tmf_cosmo] = cosmology.TMF_from_hmf(massBins[0],massBins[-1],h=h,Om0=Om0,returnObjects=True,Ob0=Ob0,sigma8 = sigma8,delta_wrt=delta_wrt,Delta=Delta,z=z)
	tmfInBinsReal = cosmology.dndm_to_n(tmf_masses,tmf_dndm,massBins)
	mReal = np.logspace(np.log10(massBins[0]),np.log10(massBins[-1]),200)
	massBinCentres = binCentres(massBins)
	if delta_wrt == 'crit':
		Msuffix = 'c'
	elif delta_wrt == 'mean':
		Msuffix = 'm'
	else:
		raise Exception("Invalid delta_wrt")
	ax.plot(massBinCentres,tmfInBinsReal,fmt,label='Tinker Mass Function prediction ($\Omega_m = ' + str(Om0) + ', \sigma_8 = ' + str(sigma8) + '$, z = ' + str(z)+ ', $M_{' + str(Delta) + Msuffix + '}$' + ')')

def plotRadiusBinsTMF(ax,rBins,RtoMFunction = None,h=0.6726,Om0= 0.315568,sigma8 = 0.830,Ob0 = 0.01,delta_wrt='crit',Delta=500,z=0,tmf=None,tmf_dndm=None,tmf_cosmo=None,tmf_masses=None,fmt='-'):
	if RtoMFunction is None:
		rhom = cosmology.rhoCos(Om0)
		a = np.log10(3/(4*np.pi*rhom))/3.0
		b = 1.0/3.0
		RtoMFunction = lambda x: (x/(10**a))**(1/b)
	massBins = RtoMFunction(rBins)
	if tmf is None:
		[tmf_dndm,tmf_masses,tmf,tmf_cosmo] = cosmology.TMF_from_hmf(massBins[0],massBins[-1],h=h,Om0=Om0,returnObjects=True,Ob0=Ob0,sigma8 = sigma8,delta_wrt=delta_wrt,Delta=Delta,z=z)
	tmfInBinsReal = cosmology.dndm_to_n(tmf_masses,tmf_dndm,massBins)
	mReal = np.logspace(np.log10(massBins[0]),np.log10(massBins[-1]),200)
	rBinCentres = binCentres(rBins)
	if delta_wrt == 'crit':
		Msuffix = 'c'
	elif delta_wrt == 'mean':
		Msuffix = 'm'
	else:
		raise Exception("Invalid delta_wrt")
	ax.plot(rBinCentres,tmfInBinsReal,fmt,label='Tinker Mass Function prediction ($\Omega_m = ' + str(Om0) + ', \sigma_8 = ' + str(sigma8) + '$, z = ' + str(z)+ ', $M_{' + str(Delta) + Msuffix + '}$' + ')')

def voidSVdWDensityInBins(massBins,rhom,ki,Pki,deltav = -2.7,deltac = 1.686,intervals=21,nPoints=200,interpMode='linear'):
	massBinCentres = (massBins[1:] + massBins[0:-1])/2
	voidDensity = np.zeros(len(massBinCentres))
	massPoints = np.logspace(np.log10(massBins[0]),np.log10(massBins[-1]),nPoints)
	SVdW_interp = interp1d(massPoints,dn_SVdWdlnm(massPoints,ki,Pki,rhom,deltav=deltav,deltac=deltac),kind=interpMode)
	for k in range(0,len(voidDensity)):
		mRange = np.logspace(np.log10(massBins[k]),np.log10(massBins[k+1]),intervals)
		dlnM = np.log10(mRange[1]/mRange[0])
		voidDensity[k] = integrate.simps(SVdW_interp(mRange),dx=dlnM,axis=-1)
	return voidDensity

def haloEPSDensityInBins(massBins,rhom,ki,Pki,deltac = 1.686,intervals=21,nPoints=200,interpMode='linear'):
	massBinCentres = (massBins[1:] + massBins[0:-1])/2
	haloDensity = np.zeros(len(massBinCentres))
	massPoints = np.logspace(np.log10(massBins[0]),np.log10(massBins[-1]),nPoints)
	epsInterp = interp1d(massPoints,dn_dlnm_EPS(massPoints,ki,Pki,rhom,deltac=deltac),kind=interpMode)
	for k in range(0,len(haloDensity)):
		mRange = np.logspace(np.log10(massBins[k]),np.log10(massBins[k+1]),intervals)
		dlnM = np.log10(mRange[1]/mRange[0])
		haloDensity[k] = integrate.simps(epsInterp(mRange),dx=dlnM,axis=-1)
	return haloDensity

# Plot SVdW model for void density:
def plotBinsSVdWDensity(ax,massBins,deltac=1.686,deltav = -2.7,h=0.6726,Om0= 0.315568,sigma8 = 0.830,Ob0 = 0.059235,ns=0.965,z=0,fmt='--',interpolationPoints=200,ki=None,Pki=None,intervals=21,interpMode='linear'):
	massBinCentres = binCentres(massBins)
	# Generate a power spectrum if none was provided
	if any([ki is None,Pki is None]):
		[ki,Pki] = cosmology.powerSpectrum(Om0 = Om0,sigma8=sigma8,Ob0 = Ob0,z=z)
	rhom = cosmology.rhoCos(Om0) # Matter density in units of Msol*h^2/Mpc^3
	voidDensitySVdW = voidSVdWDensityInBins(massBins,rhom,ki,Pki,deltav=deltav,deltac=deltac,intervals=intervals,nPoints=interpolationPoints,interpMode=interpMode)
	ax.plot(massBinCentres,voidDensitySVdW,fmt,label='Void number density, SVdW model ($\delta_v = ' + str(deltav) +', \delta_c = ' + str(deltac) + '$)')
	
# Plot EPS model for halo density
def plotBinsEPSDensity(ax,massBins,deltac=1.686,h=0.6726,Om0= 0.315568,sigma8 = 0.830,Ob0 = 0.059235,ns=0.965,z=0,fmt='--',interpolationPoints=200,ki=None,Pki=None,intervals=21,interpMode='linear'):
	massBinCentres = binCentres(massBins)
	# Generate a power spectrum if none was provided
	if any([ki is None,Pki is None]):
		[ki,Pki] = cosmology.powerSpectrum(Om0 = Om0,sigma8=sigma8,Ob0 = Ob0,z=z)
	rhom = cosmology.rhoCos(Om0) # Matter density in units of Msol*h^2/Mpc^3
	halDensityEPS = haloEPSDensityInBins(massBins,rhom,ki,Pki,deltac=deltac,intervals=intervals,nPoints=interpolationPoints,interpMode=interpMode)
	ax.plot(massBinCentres,halDensityEPS,fmt,label='Halo number density, EPS model ($\delta_c = ' + str(deltac) + '$)')

# Output fraction of crushed halos in a given bin. Assumes halo masses are in units of 10^10 Msol
def crushedRatio(halo_masses,halo_densities,massBins,delta_crush,volSim):
	if np.isscalar(delta_crush):
		# Uniform threshold applied to all:
		nonCrushed = np.where(halo_densities < delta_crush)[0]
	else:
		# Using a mass-varying crushing threshold. This is somewhat more complicated
		condition = np.zeros(halo_densities.shape,dtype=bool)
		if len(delta_crush) != len(massBins) - 1:
			raise Exception('Specified crushing threshold does not match bin list.')
		for k in range(0,len(massBins) - 1):
			condition = condition  | ( (halo_masses*1e10 > massBins[k]) & (halo_masses*1e10 <= massBins[k+1]) & (halo_densities < delta_crush[k]))
		nonCrushed = np.where(condition)		
	[voids_binList,voids_noInBins] = binValues(halo_masses[nonCrushed]*1e10,massBins)
	[halos_binList,halos_noInBins] = binValues(halo_masses*1e10,massBins)
	nzDenom = np.where(halos_noInBins != 0.0)
	ratio = voids_noInBins[nzDenom]/halos_noInBins[nzDenom]
	sigmaRatio = np.sqrt(voids_noInBins[nzDenom]*(halos_noInBins[nzDenom] - voids_noInBins[nzDenom])/halos_noInBins[nzDenom])/halos_noInBins[nzDenom]
	return [ratio,sigmaRatio,nzDenom]

# Prediction for crused fraction in the SVdW model
def crushedRatioSVdW(massBins,deltav,deltac,rhom,ki,Pki):
	return voidSVdWDensityInBins(massBins,rhom,ki,Pki,deltav=deltav,deltac=deltac)/haloEPSDensityInBins(massBins,rhom,ki,Pki,deltac=deltac)

# Plot crushed fraction:
def plotCrushedVoidFraction(ax,massBins,halo_masses,halo_densities,delta_crush,volume,fmt='x'):
	[ratio,sigmaRatio,nzDenom] = crushedRatio(halo_masses,halo_densities,delta_crush,volume)
	massBinCentres = binCentres(massBins)
	ax.errorbar(massBinCentres[nzDenom],ratio,yerr=sigmaRatio,fmt=fmt,label='Uncrushed void fraction, $\delta_{\mathrm{crush}}$ = ' + str(delta_crush) + '$')

# Plot crushed void fraction in SVdW model:
def plotCrushedVoidFractionSVdW(ax,massBins,deltav,deltac,rhom,ki,Pki,fmt='--'):
	massBinCentres = binCentres(massBins)
	ax.plot(massBinCentres,crushedRatioSVdW(massBins,deltav,deltac,rhom,ki,Pki),fmt,label='SVdW uncrushed void fraction, $\delta_v = ' + str(deltav) + ', \delta_c = ' + str(deltac) + '$')

def plotVoidNumberDensity(voidCat,ax=None,returnObjects=False,radii=None,radiusBins=np.linspace(0.0,100.0,100),fmt='x',labels=True,logscale=True,legend=True,ylim=[1e-17,1e-4]):
	# Get radii:
	if radii is None:
		radii = np.zeros(len(voidCat.voids))
		for k in range(0,len(radii)):
			radii[k] = voidCat.voids[k].radius
	# Number density of voids:
	vbox = voidCat.boxLen[0]*voidCat.boxLen[1]*voidCat.boxLen[2]
	[binListData,noInBinsData] = plot.binValues(radii,radiusBins)
	N = np.sum(noInBinsData)
	noDensity = noInBinsData/vbox
	sigmaNoDensity = np.sqrt(noInBinsData*(N - noInBinsData)/N)/vbox
	# Construct plot:
	radiusBinCentres = (radiusBins[1:] + radiusBins[0:-1])/2
	ax.errorbar(radiusBinCentres,noDensity,yerr=sigmaNoDensity,label='ZOBOV Voids',fmt=fmt)



