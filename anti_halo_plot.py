# Python script to plot halos and the corresponding anti-halo
import pynbody
from mayavi.mlab import points3d
import numpy as np
from . import excursion, antihalos, stacking, plot
from .plot import float_formatter
import multiprocessing as mp
thread_count = mp.cpu_count()
import matplotlib.pylab as plt

def scatter(halo,s1,s2):
	b = pynbody.bridge.Bridge(s1,s2)
	r1 = halo['pos']
	r2 = b(halo)['pos']
	points3d(r1[:,0],r1[:,1],r1[:,2],mode='2dvertex',color=(1,1,1))
	points3d(r2[:,0],r2[:,1],r2[:,2],mode='2dvertex',color=(1,0,0))
	

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
	
# Plot the number density in mass bins of a void catalog
def plotVoidCatalog(ax,massBins,cat,a,b,label="Void Catalog",radii=None,masses=None,vol=None,fmt='x',colour=None):
	if radii is None:
		radii = zi.getVoidProperty(cat,"radius")
	if masses is None:
		masses = RtoM(radii,a,b)
	if vol is None:
		vol = zi.getCatalogVolume(cat)
	excursion.plotMassBinNoDensity(ax,masses,massBins,vol,label=label,fmt=fmt,colour=colour)

# Plot the number density in radius bins of a void catalog
def plotVoidCatalogByRadius(ax,rBins,cat,a,b,label="Void Catalog",radii=None,vol=None,fmt='x'):
	if radii is None:
		radii = zi.getVoidProperty(cat,"radius")
	if vol is None:
		vol = zi.getCatalogVolume(cat)
	excursion.plotMassBinNoDensity(ax,radii,rBins,vol,label=label,fmt=fmt)

# Compare densities of different void catalogues:
def compareVoidNumberDensities(massBins,catalogList,labelList,a,b,colourList = None,fmtList = None,configuration = "mass",fontsize=20,ticListR = [10,20,30,40,50,60,70,80,90],ylim=[1e-8,1e-3],legendLoc='upper right'):
	fig, ax = plt.subplots()
	ax.set_xlabel('Mass bin centre ($M_{sol}/h$)',fontsize=fontsize)
	ax.set_ylabel('Void or Cluster number density ($h^3\mathrm{Mpc}^{-3}$)',fontsize=fontsize)
	ax2 = ax.twiny()
	ax2.set_xlabel('$R_{\\mathrm{eff}} (\\mathrm{Mpc}h^{-1})$',fontsize=fontsize)
	if len(catalogList) != len(labelList):
		raise Exception("Catalog and label lists must be equal.")
	for k in range(0,len(catalogList)):
		if colourList is not None:
			if k < len(colourList):
				colour = colourList[k]
			else:
				colour = None
		else:
			colour = None
		if fmtList is not None:
			if k < len(fmtList):
				fmt = fmtList[k]
			else:
				fmt = None
		else:
			fmt = None
		plotVoidCatalog(ax,massBins,catalogList[k],label=labelList[k],fmt=fmt,colour=colour)
	ax.set_yscale('log')
	font = font_manager.FontProperties(family='serif',size=fontsize)
	if configuration == "mass":
		ax.set_xscale('log')
		ax2.set_xscale('log')
		ax2.set_xticks(ax.get_xticks())
		ax2.set_xbound(ax.get_xbound())
		ticListM = antihalos.RtoM(ticListR,a,b)
		ticLabelsR = []
		for k in range(0,len(ticListR)):
			ticLabelsR.append(float_formatter(ticListR[k]))
		ax2.set_xticks(ticListM)
		ax2.set_xticklabels(ticLabelsR)
		ax.legend(loc=legendLoc,prop=font)
		ax.set_ylim(ylim)
	elif configuration == "radius":
		ax.set_xticks(ax2.get_xticks())
		ax.set_xbound(ax2.get_xbound())
		ticListR = ax2.get_xticks()
		ticListM = antihalos.RtoM(ticListR,a,b)
		ticLabelsM = []
		for k in range(0,len(ticListM)):
			ticLabelsM.append(scientificNotation(ticListM[k]))
		ax.set_xticks(ticListR)
		ax.set_xticklabels(ticLabelsM)
		ax2.legend(loc=legendLoc,prop=font)
		ax.set_ylim(ylim)
	# Font control
	ax.tick_params(axis='both',labelsize=fontsize)
	ax2.tick_params(axis='both',labelsize=fontsize)

def compareVoidsAndAntiHalosInRange(antiHaloRadii,cat,rMin,rMax):
	range_ah = np.where((antiHaloRadii >= rMin) & (antiHaloRadii <= rMax))
	range_voids = np.where((cat.radius >= rMin) & (cat.radius <= rMax))
	for k in range(0,len(range_ah[0])):
		plot.subsnap_scatter(bridge(hr[range1020_ah[0][k] + 1]),color_spec=(1,0,0))
	
	for k in range(0,len(range_voids[0])):
		plot.subsnap_scatter(snap[cat.void2Parts(range_voids[0][k])],color_spec=(0,0,1))

#Histogram to show the distribution of voids and halos x-positions
def voidHaloPositionsHistogram(centresList,labelList,boxsize = 200,nBins = 100,constraintList = None,component = 0):
	fig, ax = plt.subplots()
	for k in range(0,len(voidCatList)):
		if constraintList is not None:
			constraint = contraintList[k]
			if constraint is None:
				constraint = np.where(centresList[k][:,0] ==centresList[k][:,0])[0]
		else:
			constraint = np.where(centresList[k][:,0] ==centresList[k][:,0])[0]
		if np.isscalar(nBins):
			bins = np.linspace(0,boxsize,nBins)
		else:
			bins = np.linspace(0,boxsize,nBins[k])
		[pVoid,sigmaVoid,noInBinsVoid,inBinsVoid]  = plot.computeHistogram(centresList[k].voidCentres[constraint,component])
		plot.histWithErrors(pVoid,sigmaVoid,bins,ax=ax,label=labelList[k])
	ax.set_xlabel("$x$ position of void centre ($\mathrm{Mpc} h^{-1}$)")
	ax.set_ylabel("Probability density")
	ax.legend()
	
# Plot a scatter of c values, with optional fitted bounding lines.
def plotCScatter(Cs,rFilter=None,fontsize = 15,Cbins = np.linspace(0,1,101),legendLoc='lower right',returnAx = False):
	fig, ax = plt.subplots()
	if rFilter is None:
		rFilter = slice(len(Cs[:,0]))
	ax.scatter(Cs[rFilter,0],Cs[rFilter,1],marker='.',label='C')
	ax.set_xlabel('C[0] (Anti-halo overlap fraction)',fontsize=fontsize)
	ax.set_ylabel('C[1] (ZOBOV void overlap fraction)',fontsize=fontsize)
	if Cbins is not None:
		[binListC0,inBinsC0] = plot.binValues(Cs[:,0],Cbins)
		cBinCentres = plot.binCentres(Cbins)
		cBinMin = np.zeros(len(cBinCentres))
		cBinMax = np.zeros(len(cBinCentres))
		for k in range(0,len(cBinCentres)):
			if len(binListC0[k]) > 0:
				cBinMin[k] = np.min(Cs[binListC0[k],1])
				cBinMax[k] = np.max(Cs[binListC0[k],1])
		thresh = np.where(cBinMin > 0)
		fitLine1 = np.polyfit(cBinCentres[thresh],cBinMin[thresh],1)
		maxThresh = 0.4
		threshList = np.where((cBinCentres < maxThresh) & (cBinMax > 0))
		fitLine2 = np.polyfit(cBinCentres[threshList],cBinMax[threshList],1)
		ax.plot(cBinCentres,fitLine1[0]*cBinCentres + fitLine1[1],label='Lower-bound fit, $C[1] = ' + str(np.around(fitLine1[1],decimals=2)) + ' + ' + str(np.around(fitLine1[0],decimals=2)) + '\\times C[0]$',color='r')
		ax.plot(cBinCentres,fitLine2[0]*cBinCentres + fitLine2[1],label='Upper-bound fit, $C[1] = ' + str(np.around(fitLine2[1],decimals=2)) + ' + ' + str(np.around(fitLine2[0],decimals=2)) + '\\times C[0]$',color='g')
		ax.plot(cBinCentres,0.5*cBinCentres,'--',label='$C[1] = ' + str(0.5) + '\\times C[0]$',color='r')
		ax.plot(cBinCentres,2*cBinCentres,'--',label='$C[1] = ' + str(2) + '\\times C[0]$',color='g')
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	ax.legend(loc=legendLoc)
	ax.tick_params(axis='both',labelsize=fontsize)
	if returnAx:
		return ax

# Plot the fraction of voids and anti-halos that match each other
def plotMatchedFractions(rBins,antiHaloRadii,zobovRadii,Cs,fractions=[0.5,0.7,0.9],fontsize=15):
	[binListR,noInBinsR] = plot.binValues(antiHaloRadii,rBins)
	rBinEdgeCentre = plot.binCentres(rBins)
	noCompleteList = np.zeros((len(noInBinsR),len(fractions)),dtype=np.int32)
	for k in range(0,len(fractions)):
		for l in range(0,len(noInBinsR)):
			noCompleteList[l,k] = len(np.where((Cs[binListR[l],0] > fractions[k]) & (Cs[binListR[l],1] > fractions[k]))[0])
	completenessFrac = noCompleteList/noInBinsR[:,None]
	fractionError = np.sqrt(completenessFrac*(1.0 - completenessFrac)/noInBinsR[:,None])
	fig, ax = plt.subplots()
	for k in range(0,len(fractions)):
		ax.errorbar(rBinEdgeCentre,completenessFrac[:,k],yerr=fractionError[:,k],label='$C > ' + str(fractions[k]) + '$')
	ax.set_xlabel("Anti-halo radius [$\mathrm{Mpc}/h$]",fontsize=fontsize)
	ax.set_ylabel("Fraction with matching voids",fontsize=fontsize)
	ax.tick_params(axis='both',labelsize=fontsize)
	ax.legend()

# Plot power spectra for voids and anti-halos, computed by GenPK
def plotGenPKSpectra(psAHs,psVoids,psCross,psMatter,returnAx = False,fontzise = 15,boxsize=200):
	fig, ax = plt.subplots()
	ax.loglog(psAHs[:,0]*(2*np.pi/boxsize),psAHs[:,1]/psAHs[0,1],label="Anti-halos")
	ax.loglog(psVoids[:,0]*(2*np.pi/boxsize),psVoids[:,1]/psVoids[0,1],label="ZOBOV voids")
	ax.loglog(psCross[:,0]*(2*np.pi/boxsize),psCross[:,1]/psCross[0,1],label="Cross Correlation")
	ax.loglog(psMatter[:,0]*(2*np.pi/boxsize),psMatter[:,1]/psMatter[0,1],label="Matter")
	ax.set_xlabel("k [$h/\mathrm{Mpc}$]",fontsize = fontzise)
	ax.set_ylabel("P(k)/P(0)",fontsize = fontzise)
	ax.legend()
	ax.tick_params(axis='both',labelsize=fontzise)
	if returnAx:
		return ax

# Plot cross correlations:
def plotCrossCorrelations(rBins,xiAM,xiVM,xiAV,ax=None,rMin=0,rMax = np.inf):
	rBinCentres = plot.binCentres(rBins)
	if ax is None:
		fig, ax = plt.subplots()
	ax.plot(rBinCentres,xiAM,label="Antihalo-Matter cross correlation")
	ax.plot(rBinCentres,xiVM,label="ZOBOV void-Matter cross correlation")
	ax.plot(rBinCentres,xiAV,label="ZOBOV void-Antihalo cross correlation")
	ax.set_xlabel('$r [\mathrm{Mpc}/h]$')
	ax.set_ylabel('$\\xi(r)$',fontsize = 15)
	ax.legend()
	ax.tick_params(axis='both',labelsize=15)
	ax.set_ylim([-1,5])
	if rMin > 0:
		if np.isfinite(rMax):
			ax.set_title('$R_{\mathrm{eff}} = $' + scientificFormat(rMin) + '-' + scientificFormat(rMax) + '$\,\mathrm{Mpc}/h ($' + scientificFormat(RtoM(rMin,a,b)) + '-' + scientificFormat(RtoM(rMax,a,b)) + '$M_{\mathrm{sol}}/h)$')
		else:
			ax.set_title('$R_{\mathrm{eff}} > $' + scientificFormat(rMin) + '$\,\mathrm{Mpc}/h ($' + scientificFormat(RtoM(rMin,a,b)) + '$M_{\mathrm{sol}}/h)$')
	else:
		if np.isfinite(rMax):
			ax.set_title('$R_{\mathrm{eff}} < $' + scientificFormat(rMax) + '$\,\mathrm{Mpc}/h ($' + scientificFormat(RtoM(rMax,a,b)) + '$M_{\mathrm{sol}}/h)$')
		else:
			ax.set_title('All voids/anti-halos')

# Plot lambda distribution:
def plotLambdaDistribution(lambdaAH,lambdaZV,filterAH = None,filterZV = None,fontsize = 14,title=None,lambdaBins = np.linspace(-30,30,101),returnAx = False,returnData=False):
	if filterAH is None:
		filterAH = slice(len(lambdaAH))
	if filterZV is None:
		filterZV = slice(len(lambdaZV))
	[pLambdaAH,sigmaLambdaAH,noInBinsLambdaAH,inBinsLambdaAH] = plot.computeHistogram(lambdaAH[filterAH],lambdaBins)
	[pLambdaZV,sigmaLambdaZV,noInBinsLambdaZV,inBinsLambdaZV] = plot.computeHistogram(lambdaZV[filterZV],lambdaBins)
	fig, ax = plt.subplots()
	plot.histWithErrors(pLambdaAH,sigmaLambdaAH,lambdaBins,ax=ax,label="Anti-Halos $\\lambda_v$")
	plot.histWithErrors(pLambdaZV,sigmaLambdaZV,lambdaBins,ax=ax,label="ZOBOV Voids $\\lambda_v$")
	ax.set_xlabel("$\\lambda_v$",fontsize=fontsize)
	ax.set_ylabel("Probability density",fontsize=fontsize)
	ax.legend(prop={"size":fontsize})
	ax.set_title(title,fontsize=fontsize)
	ax.tick_params(axis='both',labelsize=fontsize)
	if returnData:
		dataStructAH = [pLambdaAH,sigmaLambdaAH,noInBinsLambdaAH,inBinsLambdaAH]
		dataStructZV = [pLambdaZV,sigmaLambdaZV,noInBinsLambdaZV,inBinsLambdaZV]
	if returnAx:
		if returnData:
			return [ax,dataStructAH,dataStructZV]
		else:
			return ax
	else:
		if returnData:
			return [dataStructAH,dataStructZV]

from void_analysis.plot import computeHistogram
def plotDeltaBarDistribution(deltaBarAH,deltaBarZV,filterAH = None,filterZV = None,fontsize = 14,title = None,deltaBarBins = np.linspace(-1,2,101),returnAx = False,returnData = False,mode = "density"):
	if filterAH is None:
		filterAH = slice(len(deltaBarAH))
	if filterZV is None:
		filterZV = slice(len(deltaBarZV))
	[pdeltaBarAH,sigmaDeltaAH,noInBinsDeltaAH,inBinsDeltaAH] = computeHistogram(deltaBarAH[filterAH],deltaBarBins,density = (mode == "density"))
	[pdeltaBarZV,sigmaDeltaZV,noInBinsDeltaZV,inBinsDeltaZV] = computeHistogram(deltaBarZV[filterZV],deltaBarBins,density = (mode == "density"))
	fig, ax = plt.subplots()
	if mode == "density":
		plot.histWithErrors(pdeltaBarAH,sigmaDeltaAH,deltaBarBins,ax=ax,label="Anti-Halos $\\bar{\\delta}_v$")
		plot.histWithErrors(pdeltaBarZV,sigmaDeltaZV,deltaBarBins,ax=ax,label="ZOBOV Voids $\\bar{\\delta}_v$")
		ax.set_ylabel("Probability density",fontsize=fontsize)
		if returnData:
			dataStructAH = [pdeltaBarAH,sigmaDeltaAH,noInBinsDeltaAH,inBinsDeltaAH]
			dataStructZV = [pdeltaBarZV,sigmaDeltaZV,noInBinsDeltaZV,inBinsDeltaZV]
	elif mode == "absolute":
		NAH = len(deltaBarAH[filterAH])
		NZV = len(deltaBarZV[filterZV])
		nDeltaBarAH = NAH*pdeltaBarAH
		nDeltaBarZV = NZV*pdeltaBarZV
		sigmanDeltaAH = NAH*sigmaDeltaAH
		sigmanDeltaZV = NZV*sigmaDeltaZV
		plot.histWithErrors(nDeltaBarAH,sigmanDeltaAH,deltaBarBins,ax=ax,label="Anti-Halos")
		plot.histWithErrors(nDeltaBarZV,sigmanDeltaZV,deltaBarBins,ax=ax,label="ZOBOV Voids")
		ax.set_ylabel("Number of voids",fontsize=fontsize)
		if returnData:
			dataStructAH = [nDeltaBarAH,sigmanDeltaAH,noInBinsDeltaAH,inBinsDeltaAH]
			dataStructZV = [nDeltaBarZV,sigmanDeltaZV,noInBinsDeltaZV,inBinsDeltaZV]
	else:
		raise Exception("Invalid plot mode.")
	ax.set_xlabel("$\\delta$",fontsize=fontsize)
	ax.legend(prop={"size":fontsize})
	ax.set_title(title,fontsize=fontsize)
	ax.tick_params(axis='both',labelsize=fontsize)
	if returnAx:
		if returnData:
			return [ax,dataStructAH,dataStructZV]
		else:
			return ax
	else:
		if returnData:
			return [dataStructAH,dataStructZV]
	
# Scatter plot for void/anti-halos masses and radii:
def scatterMassRadius(antiHaloMasses,antiHaloRadii,zobovMasses,zobovRadii,a,b):
	fig, ax = plt.subplots()
	ax.scatter(zobovMasses,zobovRadii,marker='.',label='ZOBOV voids')
	ax.scatter(antiHaloMasses,antiHaloRadii,marker='x',label='Anti-halos')
	ax.plot(antiHaloMasses,antihalos.MtoR(antiHaloMasses,a,b),'--',label='Mass-to-Radius map.')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('Mass, $[M_{\\mathrm{sol}}/h]$')
	ax.set_ylabel('Radius, $[\\mathrm{MPc}/h]$')
	ax.legend()

from void_analysis.plot import scientificNotation
def plotStacks(rBins,nBarsAH,nBarsZV,sigmaBarsAH,sigmaBarsZV,sizeBins,binType,nbar,ax=None,colorList = ['r','g','b'],fontsize=14,plotAH=True,plotZV=True,yUpper = 1.3,binLabel="",labelAH = 'Anti-halos ',labelZV = 'ZOBOV Voids ',title = "Stacked void profiles",powerRange = 3):
	if ax is None:
		fig, ax = plt.subplots()
	rBinStackCentres = plot.binCentres(rBins)
	#nbar = len(snap)/(snap.properties['boxsize'].ratio("Mpc a h**-1")**3)
	for k in range(0,len(sizeBins)-1):
		if binType == "radius":
			rangeLabel = str(sizeBins[k]) + "$ < R_{\mathrm{eff}} < $" + str(sizeBins[k+1])
		elif (binType == "mass") or (binType == "RtoM"):
			rangeLabel = '$' +  scientificNotation(sizeBins[k],powerRange = powerRange) + " < M/M_{\\mathrm{sol}} < " + scientificNotation(sizeBins[k+1],powerRange = powerRange) + '$'
		else:
			rangeLabel = '$' + scientificNotation(sizeBins[k],powerRange = powerRange) + " < " + binLabel + " < " + scientificNotation(sizeBins[k+1],powerRange = powerRange) + '$'
		if k < len(colorList):
			color = colorList[k]
		else:
			color = None
		if plotAH:
			ax.errorbar(rBinStackCentres,nBarsAH[k]/nbar,yerr=sigmaBarsAH[k]/nbar,label=labelAH + rangeLabel,color=color,fmt='-')
		if plotZV:
			ax.errorbar(rBinStackCentres,nBarsZV[k]/nbar,yerr=sigmaBarsZV[k]/nbar,label=labelZV + rangeLabel,color=color,fmt='--')
	ax.plot(rBinStackCentres,np.ones(rBinStackCentres.shape),'k--')
	ax.plot([1,1],[0,3],'k--')
	ax.set_title(title)
	ax.set_xlabel("$R/R_{\mathrm{eff}}$",fontsize=fontsize)
	ax.set_ylabel("$\\rho/\\bar{\\rho}$",fontsize=fontsize)
	ax.tick_params(axis='both',labelsize=fontsize)
	ax.set_ylim([0,yUpper])
	ax.legend(prop={"size":fontsize})

# Plot stack comparisons:
def plotStackComparison(listRanges,snap,nbar,antiHaloRadii,antiHaloCentres,voidRadii,voidCentres,filterListAH,filterListZV,pairCountsAH,volumesListAH,volumesListZV,pairCountsZV,colourList,rBins,listType = "radius",ylim=[0,1.5],fontsize=15,returnAx = False):
	if listType == "radius":
		rangeText = "r/r_{\\mathrm{eff}}"
	elif listType == "mass":
		rangeText = "M/(M_{\\mathrm{sol}}h^{-1})"
	fig, ax = plt.subplots()
	rBinStackCentres = plot.binCentres(rBins)
	for k in range(0,len(filterListAH)):
		if filterListAH[k] is None:
			filterToUse = slice(len(antiHaloRadii))
		else:
			filterToUse = filterListAH[k]
		[nbarj_AHh,sigma_AH] = stacking.stackVoidsWithFilter(antiHaloCentres,antiHaloRadii,filterAH,snap,rBins=rBinStack,nPairsList = pairCountsAH,volumesList=volumesListAH)
		[nbarj_ZV,sigma_ZV] = stackVoidsWithFilter(voidCentres,voidRadii,filterZV,snap,rBins=rBins,nPairsList = pairCountsZV,volumesList=volumesListZV)
		ax.errorbar(rBinStackCentres,nbarj_AH/nbar,yerr=sigma_AH/nbar,label='Anti-halos, $' + str(listRanges[k,0]) + ' < ' + rangeText + ' < ' + str(listRanges[k,1]) + '$',fmt='-',color=colourList[k])
		ax.errorbar(rBinStackCentres,nbarj_ZV/nbar,yerr=sigma_ZV/nbar,label='ZOBOV voids, $'  + str(listRanges[k,0]) + ' < ' + rangeText + ' < ' + str(listRanges[k,1]) + '$',fmt='--',color=colourList[k])
	ax.plot(rBinStackCentres,np.ones(rBinStackCentres.shape),'k--')
	ax.plot([1,1],[0,1.5],'k--')
	ax.set_title("Stacked void profiles")
	ax.set_xlabel("$R/R_{\mathrm{eff}}$",fontsize=fontsize)
	ax.set_ylabel("$\\rho/\\bar{\\rho}$",fontsize=fontsize)
	ax.tick_params(axis='both',labelsize=fontsize)
	ax.legend(prop={"size":fontsize})
	ax.set_ylim(ylim)
	if returnAx:
		return ax


# Compare profiles to predictions:
from void_analysis.stacking_plots import plotProfileVsPrediction
def compareProfilePredictions(filterList,radii,centres,snap,rBins,pairCounts,volumesList,nbar,listRanges,colourList,listType="radius",fontsize=15):
	if listType == "radius":
		rangeText = "r/r_{\\mathrm{eff}}"
	elif listType == "mass":
		rangeText = "M/(M_{\\mathrm{sol}}h^{-1})"
	fig, ax = plt.subplots()
	for k in range(0,len(filterList)):
		[nbarj,sigma] = stacking.stackVoidsWithFilter(centres,radii,filterList[k],snap,rBins=rBins,nPairsList = pairCounts,volumesList=volumesList)
		plotProfileVsPrediction(nbarj,sigma,rBins,nbar,ax=ax,formatAxis=False,color=colourList[k],labelSuffix = "$, " + str(listRanges[k,0]) +  " < " + listType + " < " + str(listRanges[k,1]) + "$")
	ax.set_xlabel('$r/r_{\\mathrm{eff}}$',fontsize=fontsize)
	ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=fontsize)
	ax.tick_params(axis='both',labelsize=fontsize)
	ax.legend(prop={"size":fontsize})

# Histogram of densities:
def densityHistogram(deltaAH,deltaZV,centralAH,centralZV,nBins=21,deltaLow=-1,deltaHigh=0,valuesAH=None,valuesZV = None,valueFilter=None,ax=None,fontsixze=14,title=""):
	if valuesAH is not None:
		filterAH = (valuesAH > valueFilter[0]) & (valuesAH <= valueFilter[1])
	else:
		filterAH = np.ones(len(deltaAH),dtype=np.bool)
	if centralAH is not None:
		filterAH = filterAH & (centralAH < 0.0)
	if valuesZV is not None:
		filterZV = (valuesZV > valueFilter[0]) & (valuesZV <= valueFilter[1])
	else:
		filterZV = np.ones(len(deltaZV),dtype=np.bool)
	if centralZV is not None:
		filterZV = filterZV & (centralZV < 0.0)
	bins = np.linspace(deltaLow,deltaHigh,nBins)
	[pDeltaZV,sigmaDeltaZV,nDeltaZV,nBinsZV] = plot.computeHistogram(deltaZV[np.where(filterZV)],bins)
	[pDeltaAH,sigmaDeltaAH,nDeltaAH,nBinsAH] = plot.computeHistogram(deltaAH[np.where(filterAH)],bins)
	if ax is None:
		fig, ax = plt.subplots()
	
	plot.histWithErrors(pDeltaAH,sigmaDeltaAH,bins,label = "Antihalos",ax=ax)
	plot.histWithErrors(pDeltaZV,sigmaDeltaZV,bins,label = "ZOBOV Voids",ax=ax)
	ax.set_xlabel('$\\delta$',fontsize=fontsize)
	ax.set_ylabel('Probability density',fontsize=fontsize)
	ax.tick_params(axis='both',labelsize=fontsize)
	ax.set_title(title,fontsize=fontsize)
	ax.legend(prop={"size":fontsize})
	
# Plot velocity profiles
def plotVelocityProfile(vRShells,volumesList,pairCounts,voidRadii,rBins,nbar,voidFilterCondition = None,Om=0.279,Ol=0.721,z=0,ax=None,fontsize=14,errorType="mean",labelLinear = 'Linear Velocity Profile',labelNonlinear = 'Non-linear velocity profile',formatPlot = True,returnAx = False,lc = 'r--',nlc = 'b'):
	if voidFilterCondition is None:
		voidFilterCondition = (vRShells[:,0] == vRShells[:,0])
	voidFilter = np.where(voidFilterCondition)[0]
	delta_cum = deltaCumulative(pairCounts,volumesList,nbar)
	delta_cumStacked = stacking.weightedMean(delta_cum[voidFilter,:],volumesList[voidFilter,:],axis = 0)
	delta_cumError = np.sqrt(stacking.weightedVariance(delta_cum[voidFilter,:],volumesList[voidFilter,:],axis = 0)/(len(voidFilter)-1))
	rMean = np.mean(voidRadii[voidFilter])
	if errorType == "mean":
		rStd = np.std(voidRadii)/np.sqrt(len(voidFilter)-1)
	else:	
		rStd = np.std(voidRadii)
	rBinCentres = plot.binCentres(rBins)
	#vRlinearLow = vLinear(rBinCentres,delta_cumStacked,Om=Om,Ol=Ol,z=z)
	#vRlinearUpp = vLinear(rBinCentres,delta_cumStacked,Om=Om,Ol=Ol,z=z)
	vRlinear = vLinear(rBinCentres,delta_cumStacked,Om=Om,Ol=Ol,z=z)
	vRStacked = stacking.weightedMean(vRShells[voidFilter,:]/voidRadii[voidFilter,None],volumesList[voidFilter],axis=0)
	vRError = np.sqrt(stacking.weightedVariance(vRShells[voidFilter,:]/voidRadii[voidFilter,None],volumesList[voidFilter],axis=0))/np.sqrt(len(voidFilter) - 1)
	# Plot:
	if ax is None:
		fig, ax = plt.subplots()
	ax.errorbar(rBinCentres,vRStacked,yerr=vRError,fmt='-',color=nlc,label=labelNonlinear)
	ax.plot(rBinCentres,vRlinear,lc,label=labelLinear)
	#ax.fill_between(rBinCentres,vRlinearLow,vRlinearUpp,facecolor='r',alpha=0.5)
	if formatPlot:
		ax.legend()
		ax.set_xlabel("$r/r_{\\mathrm{eff}}$",fontsize=fontsize)
		ax.set_ylabel("$v_{\\mathrm{radial}}/R_{\\mathrm{eff}}$ [$\\mathrm{kms}^{-1}\\mathrm{Mpc}^{-1}h$]",fontsize=fontsize)
	if returnAx:
		return ax


# Compare different velocity profiles
def velocityProfileComparison(vRShellsAH,vRShellsZV,volumesListAH,volumesListZV,radiiAH,radiiZV,rBins,nbar,conditionAH=None,conditionZV = None,Om=0.279,Ol=0.721,z=0,ax=None,fontsize=14,errorType="mean",binType = "mass",binRange = [1e14,1e15],valuesAH = None,valuesZV = None):
	if (valuesAH is None):
		valuesAH = (radiiAH == radiiAH)
	if (valuesZV is None):
		valuesZV = (radiiZV == radiiZV)
	if binType == "mass":
		labelType = "M/(M_{\\mathrm{sol}}h^{-1})"
	elif binType == "radius":
		labelType = "R/R_{\\mathrm{eff}}"
	else:
		raise Exception("Invalid bin type.")
	rangeLabel = "$" + plot.scientificNotation(binRange[0]) + " < " + labelType + " < " + plot.scientificNotation(binRange[1]) + "$"
	voidFilterConditionAH = (valuesAH  > binRange[0]) & (valuesAH < binRange[1])
	voidFilterConditionZV = (valuesZV  > binRange[0]) & (valuesZV < binRange[1])
	if conditionAH is not None:
		voidFilterConditionAH = voidFilterConditionAH & conditionAH
	if conditionZV is not None:
		voidFilterConditionZV = voidFilterConditionZV & conditionZV
	ax = plotVelocityProfile(vRShellsAH,volumesListAH,pairCountsAH,radiiAH,rBins,voidFilterCondition=voidFilterConditionAH,returnAx=True,formatPlot=False,labelLinear = "Anti-halos, linear velocity profile " + rangeLabel,labelNonlinear = "Anti-halos, non-linear velocity profile " +  rangeLabel)
