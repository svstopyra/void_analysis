# Code for plotting various things related to stacking, such as void profiles.

# PLOTS USED FOR THE LINEAR VOIDS PAPER:

from .stacking import *

# Histogram representing the distribution of central and average density:
def plotDensityHistograms(deltaBarAH,deltaBarZV,deltaCentralAH,deltaCentralZV,massesAH,massesZV,fontname="serif",textwidth=7.1014,textheight=9.0971,fontsize=8,deltaLow=-1,deltaHigh=1,valueFilter1=[1e13,1e14],valueFilter2=[1e14,1e15],title00="",title01="",title10="",title11="",nBins=41,density=True,ylabel='Probability density',xlabel='$\\delta_c$',aspect_ratio=(3.0/4.0),titleLeft='Average Densities',titleRight='Central Densities',hspace=0.0,wspace=0.09,bottom=0.1,top=0.95,right=0.95,left=0.08,textUpper="$10^{13}\\mathrm{-}10^{14}\\,M_{\\mathrm{\odot}}h^{-1}$",textLower="$10^{14}\\mathrm{-}10^{15}\\,M_{\\mathrm{\odot}}h^{-1}$"):
	fig, ax = plt.subplots(2,2,figsize=(textwidth,textwidth*aspect_ratio))
	densityHistogram(deltaBarAH,deltaBarZV,nBins=nBins,deltaLow=deltaLow,deltaHigh=deltaHigh,valuesAH=massesAH,valuesZV = massesZV,valueFilter=valueFilter1,centralAH=deltaCentralAH,centralZV = deltaCentralZV,ax=ax[0,0],fontsize=fontsize,title=title00,density=density,ylabel=ylabel,xlabel=None,includeLegend=False,fontname=fontname)
	ax[0,0].axes.get_xaxis().set_visible(False)
	ax[0,0].yaxis.get_major_ticks()[0].set_visible(False)
	densityHistogram(deltaCentralAH,deltaCentralZV,nBins=nBins,deltaLow=deltaLow,deltaHigh=deltaHigh,valuesAH=massesAH,valuesZV = massesZV,valueFilter=valueFilter1,centralAH=deltaCentralAH,centralZV = deltaCentralZV,ax=ax[0,1],fontsize=fontsize,title=title01,density=density,ylabel=None,xlabel=None,includeLegend=True,fontname=fontname,numbersAH=False,numbersZV = False)
	ax[0,1].yaxis.get_major_ticks()[0].set_visible(False)
	ax[0,1].axes.get_xaxis().set_visible(False)
	densityHistogram(deltaBarAH,deltaBarZV,nBins=nBins,deltaLow=deltaLow,deltaHigh=deltaHigh,valuesAH=massesAH,valuesZV = massesZV,valueFilter=valueFilter2,centralAH=deltaCentralAH,centralZV = deltaCentralZV,ax=ax[1,0],fontsize=fontsize,title=title10,density=density,ylabel=ylabel,xlabel=xlabel,includeLegend=False,fontname=fontname)
	densityHistogram(deltaCentralAH,deltaCentralZV,nBins=nBins,deltaLow=deltaLow,deltaHigh=deltaHigh,valuesAH=massesAH,valuesZV = massesZV,valueFilter=valueFilter2,centralAH=deltaCentralAH,centralZV = deltaCentralZV,ax=ax[1,1],fontsize=fontsize,title=title11,density=density,ylabel=None,xlabel=xlabel,includeLegend=False,fontname=fontname,numbersAH=False,numbersZV = False)
	ax[1,1].text(1.07,0.25,textLower,fontsize=fontsize,fontfamily=fontname,transform=ax[1,1].transAxes,rotation='vertical',horizontalalignment='center')
	ax[0,1].text(1.07,0.25,textUpper,fontsize=fontsize,fontfamily=fontname,transform=ax[0,1].transAxes,rotation='vertical',horizontalalignment='center')
	ax[0,0].set_title(titleLeft,fontsize=fontsize,fontfamily=fontname)
	ax[0,1].set_title(titleRight,fontsize=fontsize,fontfamily=fontname)
	plt.subplots_adjust(hspace=hspace,bottom=bottom,top=top,right=right,left=left,wspace=wspace)

# Compare the linear and non-linear profiles (use conditionAH and conditionZV to apply cuts):
def plotLinearComparison(rBinStack,snap,massesAH,massesZV,radiiAHNL,radiiZVNL,radiiAHL,radiiZVL,centresAHNL,centresZVNL,centresAHL,centresZVL,pairCountsAHNL,pairCountsZVNL,pairCountsAHL,pairCountsZVL,volumesListAHNL,volumesListZVNL,volumesListAHL,volumesListZVL,conditionAH,conditionZV,fontname="serif",textwidth=7.1014,textheight=9.0971,fontsize=8,aspect_ratio=0.4,massBoundaries=np.array([1e13,1e14,1e15]),colourList=['#F5793A','#0F2080'],listType = "mass",ylim=[0,1.5],titleLeft="Anti-halos",titleRight="ZOBOV Voids",guideColor='0.75',set1Label='Non-linear',set2Label='Linear',formatList=['-',':'],method="poisson",errorType="Weighted",styleNL='-',styleL=':',wspace=0.0,bottom=0.17,left=0.1,right=0.97,top=None,legLoc='lower right',frameon=False):
	fig, ax = plt.subplots(1,2,figsize=(textwidth,aspect_ratio*textwidth))
	nbar = len(snap)/(snap.properties['boxsize'].ratio("Mpc a h**-1"))**3
	plotStackComparison(massBoundaries,massesAH,massesAH,snap,nbar,radiiAHNL,centresAHNL,radiiAHL,centresAHL,pairCountsAHNL,volumesListAHNL,volumesListAHL,pairCountsAHL,['#F5793A','#0F2080'],rBinStack,listType = listType,ylim=ylim,fontsize=fontsize,returnAx = False,title=titleLeft,conditionAH = conditionAH,conditionZV = conditionAH,cycle="colour",set1Label=set1Label,set2Label=set2Label,ax=ax[0],fontname=fontname,includeLegend=False,legendFontSize=fontsize,shortFormLegend=True,formatList=formatList,legendCols=1,guideColor=guideColor,method=method,errorType=errorType)
	plotStackComparison(massBoundaries,massesZV,massesZV,snap,nbar,radiiZVNL,centresZVNL,radiiZVL,centresZVL,pairCountsZVNL,volumesListZVNL,volumesListZVL,pairCountsZVL,['#F5793A','#0F2080'],rBinStack,listType = listType,ylim=ylim,fontsize=fontsize,returnAx = False,title=titleRight,conditionAH = conditionZV,conditionZV = conditionZV ,cycle="colour",set1Label=set1Label,set2Label=set2Label,ax=ax[1],includeLegend=False,fontname=fontname,legendFontSize=fontsize,includeYLabel=False,shortFormLegend=True,legendCols=1,formatList=formatList,guideColor=guideColor,method=method,errorType=errorType)
	fakeLines = [Line2D([0],[0],color='k',linestyle=styleNL),Line2D([0],[0],color='k',linestyle=styleL),Line2D([0],[0],color=colourList[0],linestyle=styleNL),Line2D([0],[0],color=colourList[1],linestyle=styleNL)]
	ax[0].legend(fakeLines[0:2],[set1Label,set2Label],prop={"size":fontsize,"family":fontname},loc=legLoc,frameon=frameon)
	ax[1].legend(fakeLines[2:4],['$' + scientificNotation(massBoundaries[0]) + '\\mathrm{-}' + scientificNotation(massBoundaries[1]) + '\\,M_{\odot}\\,h^{-1}$','$' + scientificNotation(massBoundaries[1]) + '\\mathrm{-}' + scientificNotation(massBoundaries[2]) + '\\,M_{\odot}\\,h^{-1}$'],prop={"size":fontsize,"family":fontname},loc=legLoc,frameon=frameon)
	ax[1].axes.get_yaxis().set_visible(False)
	plt.subplots_adjust(wspace=wspace,bottom=bottom,left=left,right=right,top=top)
	ax[0].xaxis.get_major_ticks()[-2].set_visible(False)

# Plot Density cut variation:
def plotDensityCutVariation(nbar,rBinStack,massesAH,massesZV,radiiAH,radiiZV,centresAH,centresZV,pairCountsAH,pairCountsZV,volumesListAH,volumesListZV,deltaBarAH,deltaBarZV,deltaCentralAH,deltaCentralZV,threshList = np.linspace(-0.6,1,6),massBoundaries=np.array([1e13,1e14,1e15]),fontname="serif",textwidth=7.1014,textheight=9.0971,fontsize=8,legLoc = 'upper right',legendCols = 2,colorMin=0.3,colorMax=1.0,colourMapType = 3,baseColourMap='Reds',aspect_ratio=0.4,method="poisson",errorType="Weighted",formatList = ['-',':'],right=0.84,left=0.10,wspace=0.0,bottom=0.17,top=None,hspace=None,cbaxParams=[0.85,0.17,0.03,0.73],cbaxLabel='$\\bar{\delta}_v$ (Upper Bound)',legendLabelsList = ['Anti-halos','ZOBOV Voids'],logMapping=False,colRange=np.linspace(0.3,1,256),conditionAHLow=True,conditionZVLow=True,conditionAHHigh=True,conditionZVHigh=True,extraText1="",extraText2="",removeDenseCentresAH = True,dBl1314=None,dBl1415=None,dCl1314=None,dCl1415=None,dCu1314=None,dCu1415=None):
	if removeDenseCentresAH:
		conditionAHLow = conditionAHLow & (deltaCentralAH < 0.0)
		conditionAHHigh = conditionAHHigh & (deltaCentralAH < 0.0)
	if logMapping:
		mapMin = np.min(np.log(1.0 + threshList))
		mapMax = np.max(np.log(1.0 + threshList))
	else:
		mapMin = np.min(threshList)
		mapMax = np.max(threshList)
	if(mapMin - mapMax == 0.0):
		mappedRange = colorMin + (colorMax-colorMin)*0.5*np.ones(len(threshList))
	else:	
		if logMapping:
			mappedRange = colorMin + (colorMax - colorMin)*(np.log(1.0 + threshList) - mapMin)/(mapMax - mapMin)
		else:
			mappedRange = colorMin + (colorMax - colorMin)*(threshList - mapMin)/(mapMax - mapMin)
	
	# Truncated colour-list, and discrete colour normalisation for the colourbar:
	#colourList = truncate_colormap(plt.get_cmap(baseColourMap),minval=colorMin,maxval=colorMax,n=len(threshList),colourList=colourBoundaries(mappedRange),uneven=True)
	#colourListNorm = colors.BoundaryNorm(colourBoundaries(threshList),len(threshList))
	colorRange = cmap(colRange)
	colourList = colors.ListedColormap(colorRange)
	mappedColors = colourList(mappedRange)
	new_cmap = colors.ListedColormap(mappedColors)
	colourListNorm = colors.BoundaryNorm(colourBoundaries(threshList),len(threshList))
	# Scalar mappable for constructing the colourbar:
	#sm = plt.cm.ScalarMappable(cmap=colourList,norm=colourListNorm)
	sm = plt.cm.ScalarMappable(cmap=new_cmap,norm=colourListNorm)
	# Create the plot:
	fig, ax = plt.subplots(1,2,figsize=(textwidth,aspect_ratio*textwidth))
	# Anti-halos in the LH plot, zobov voids in the RH plot:
	plotDensityCutComparisons(rBinStack,radiiAH,radiiZV,centresAH,centresZV,pairCountsAH,pairCountsZV,volumesListAH,volumesListZV,deltaBarAH,deltaBarZV,deltaCentralAH,deltaCentralZV,nbar,conditionAH = (massesAH > massBoundaries[0]) & (massesAH < massBoundaries[1]) & conditionAHLow,conditionZV = (massesZV > massBoundaries[0]) & (massesZV < massBoundaries[1]) & conditionZVLow,dBu=threshList,dBl=dBl1314,dCl=dCl1314,dCu=dCu1314,colourList = colourList,title='$' + scientificNotation(massBoundaries[0]) + ' \\mathrm{-} ' + plot.scientificNotation(massBoundaries[1]) + '\\,M_{\\mathrm{\odot}}h^{-1}$' + extraText1,fontsize=fontsize,fontname=fontname,ax=ax[0],includeColorBar=False,includeLegend=False,method=method,errorType=errorType,logSpacedColours=logMapping)
	plotDensityCutComparisons(rBinStack,radiiAH,radiiZV,centresAH,centresZV,pairCountsAH,pairCountsZV,volumesListAH,volumesListZV,deltaBarAH,deltaBarZV,deltaCentralAH,deltaCentralZV,nbar,conditionAH = (massesAH > massBoundaries[1]) & (massesAH < massBoundaries[2]) & conditionAHHigh,conditionZV = (massesZV > massBoundaries[1]) & (massesZV < massBoundaries[2]) & conditionZVHigh,dBu=threshList,dBl=dBl1415,dCl=dCl1415,dCu=dCu1415,colourList = colourList,title='$' + scientificNotation(massBoundaries[1]) + ' \\mathrm{-} ' + plot.scientificNotation(massBoundaries[2]) + '\\,M_{\\mathrm{\odot}}h^{-1}$' + extraText2,fontsize=fontsize,fontname=fontname,ax=ax[1],includeLegend=False,includeColourLabels=True,includeColorBar=False,includeYLabel=False,method=method,errorType=errorType,logSpacedColours=logMapping)
	ax[1].axes.get_yaxis().set_visible(False) # Hide the y  axis for the RH plot
	fakeLines = [Line2D([0],[0],color='k',linestyle=formatList[0]),Line2D([0],[0],color='k',linestyle=formatList[1])] # Easier than directly accessing the lines created by plotDensityCutComparisons
	toCompareLength = len(threshList)
	# Create labels for each entry on the discrete colourbar:
	for k in range(0,toCompareLength):
		fakeLines.append(Line2D([0],[0],color=colourList(mappedRange[k]),linestyle='-'))
		legendLabelsList.append(scientificNotation(threshList[k]))
	# Construct the legend:
	if legendCols > 1:
		ax[1].legend(fakeLines[0:2],legendLabelsList[0:2],prop={"size":legFontSize,"family":fontname},loc=legLoc,frameon=False)
	else:
		ax[0].legend(fakeLines[0:4],legendLabelsList[0:4],prop={"size":legFontSize,"family":fontname},loc=legLoc,frameon=False)
		ax[1].legend(fakeLines[4:8],legendLabelsList[4:8],prop={"size":legFontSize,"family":fontname},loc=legLoc,frameon=False)
	# Final adjustments, and to make room for the colourbar:
	plt.subplots_adjust(right=right,left=left,wspace=wspace,bottom=bottom,hspace=hspace,top=top)
	# Add the colourbar:
	cbax = fig.add_axes(cbaxParams)
	cbax.tick_params(axis='both',labelsize=fontsize)
	cbar = fig.colorbar(sm,cax=cbax,boundaries=colourBoundaries(threshList))
	cbar.set_label(cbaxLabel,fontsize=fontsize,fontfamily=fontname)
	cbar.set_ticks(threshList)

# 2D histograms of central and average density:
def plotDensity2DHistograms(deltaBarAH,deltaBarZV,deltaCentralAH,deltaCentralZV,conditionAH1314,conditionZV1314,conditionAH1415,conditionZV1415,fontname="serif",textwidth=7.1014,textheight=9.0971,fontsize=8,colourMapBase='Blues',ylabel = '$\\delta_c$',xlabel = '$\\bar{\\delta}_v$',vmin=0,vmax=50,deltaRange = [[-1,0],[-1,-0.6]],nBins1314 = 41,nBins1415 = 41,titleY = 0.9,titleX = 0.5,titleLeft="Anti-halos",titleRight="ZOBOV voids",right=0.8,left=0.13,wspace=0.0,hspace=0.0,bottom=None,top=None,cbarParams=[0.85,0.11,0.03,0.77],cbaxLabel='Void Count',rowLabelX= 1.10,rowLabelY = 0.25,rowLabelLower="$10^{14}\\mathrm{-}10^{15}\\,M_{\\mathrm{\odot}}h^{-1}$",rowLabelUpper="$10^{13}\\mathrm{-}10^{14}\\,M_{\\mathrm{\odot}}h^{-1}$"):
	# Create figure:		
	fig, ax = plt.subplots(2,2,figsize=(textwidth,textwidth))
	# Construct scalar mappable for the colourbar:
	sm = plt.cm.ScalarMappable(cmap=colourMapBase,norm=plt.Normalize(vmin=vmin,vmax=vmax))
	# Top left:
	hist00 = ax[0,0].hist2d(deltaBarAH[conditionAH1314],deltaCentralAH[conditionAH1314],range=deltaRange,bins=nBins1314,cmap=colourMapBase)
	ax[0,0].set_ylabel(ylabel,fontsize=fontsize,fontfamily=fontname)
	ax[0,0].axes.get_xaxis().set_visible(False)
	ax[0,0].set_title(titleLeft,fontsize=fontsize,fontfamily=fontname) # Left column label
	ax[0,0].tick_params(axis='both',labelsize=fontsize)
	# Top right:
	hist01 = ax[0,1].hist2d(deltaBarZV[conditionZV1314],deltaCentralZV[conditionZV1314],range=deltaRange,bins=nBins1314,cmap=colourMapBase)
	ax[0,1].axes.get_yaxis().set_visible(False)
	ax[0,1].axes.get_xaxis().set_visible(False)
	ax[0,1].tick_params(axis='both',labelsize=fontsize)
	ax[0,1].set_title(titleRight,fontsize=fontsize,fontfamily=fontname) # Right column label
	# Bottom left:
	ax[1,0].hist2d(deltaBarAH[conditionAH1415],deltaCentralAH[conditionAH1415],range=deltaRange,bins=nBins1415,cmap=colourMapBase)
	ax[1,0].set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontname)
	ax[1,0].set_ylabel(ylabel,fontsize=fontsize,fontfamily=fontname)
	ax[1,0].tick_params(axis='both',labelsize=fontsize)
	ax[1,0].xaxis.get_major_ticks()[-1].set_visible(False) # Hide top ticks to prevent clashing
	ax[1,0].yaxis.get_major_ticks()[-1].set_visible(False)
	# Bottom right:
	ax[1,1].hist2d(deltaBarZV[conditionZV1415],deltaCentralZV[conditionZV1415],range=deltaRange,bins=nBins1415,cmap=colourMapBase)
	ax[1,1].set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontname)
	ax[1,1].axes.get_yaxis().set_visible(False)
	# Adjustments and colourbar:
	plt.subplots_adjust(right=right,left=left,wspace=wspace,hspace=hspace,bottom=bottom,top=top)
	cbax = fig.add_axes(cbarParams)
	cbax.tick_params(axis='both',labelsize=fontsize)
	cbar = fig.colorbar(sm,cax=cbax)
	cbar.set_label(cbaxLabel,fontsize=fontsize,fontfamily=fontname)
	# Row labels:
	ax[1,1].text(rowLabelX,rowLabelY,rowLabelLower,fontsize=fontsize,fontfamily=fontname,transform=ax[1,1].transAxes,rotation='vertical',horizontalalignment='center')
	ax[0,1].text(rowLabelX,rowLabelY,rowLabelUpper,fontsize=fontsize,fontfamily=fontname,transform=ax[0,1].transAxes,rotation='vertical',horizontalalignment='center')
	ax[1,1].tick_params(axis='both',labelsize=fontsize)

# 1D Histogram of the radii:
def plotRadiiHistogram(radiiAH,radiiZV,filterAH1314,filterAH1415,filterZV1314,filterZV1415,aspect_ratio=0.5,fontname="serif",textwidth=7.1014,textheight=9.0971,fontsize=8,rBinsLow=np.linspace(0,20,30),rBinsHigh=np.linspace(4,20,10),label1="Anti-halos",label2="ZOBOV Voids",xlabel="$R/R_{\mathrm{eff}}$",ylabel="Number of Voids",titleLeft="$10^{13}\\mathrm{-}10^{14}\\,M_{\\mathrm{\odot}}h^{-1}$",titleRight="$10^{14}\\mathrm{-}10^{15}\\,M_{\\mathrm{\odot}}h^{-1}$",frameon=False,rightAxisPostion = "right",right=0.9,left=0.13,wspace=0.1,hspace=0.0,bottom=0.15,top=None):
	# Bin the data:
	[p1314ZV,sigma1314ZV,noInBins1314ZV,inBins1314ZV] = computeHistogram(radiiZV[filterZV1314],rBinsLow,density=False)
	[p1415ZV,sigma1415ZV,noInBins1415ZV,inBins1415ZV] = computeHistogram(radiiZV[filterZV1415],rBinsHigh,density=False)
	[p1314AH,sigma1314AH,noInBins1314AH,inBins1314AH] = computeHistogram(radiiAH[filterAH1314],rBinsLow,density=False)
	[p1415AH,sigma1415AH,noInBins1415AH,inBins1415AH] = computeHistogram(radiiAH[filterAH1415],rBinsHigh,density=False)
	# Construct plot:
	fig, ax = plt.subplots(1,2,figsize=(textwidth,textwidth*aspect_ratio))
	# Left panel:
	plot.histWithErrors(p1314AH,sigma1314AH,bins=rBinsLow,label=label1,ax=ax[0])
	plot.histWithErrors(p1314ZV,sigma1314ZV,bins=rBinsLow,label=label2,ax=ax[0])
	ax[0].tick_params(axis='both',labelsize=fontsize)
	ax[0].set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontname)
	ax[0].set_ylabel(ylabel,fontsize=fontsize,fontfamily=fontname)
	ax[0].set_title(titleLeft,fontsize=fontsize,fontfamily=fontname)
	ax[0].legend(prop={"size":fontsize,"family":fontname},frameon=frameon)
	# Right panel:
	plot.histWithErrors(p1415AH,sigma1415AH,bins=np.linspace(4,20,10),label=label1,ax=ax[1])
	plot.histWithErrors(p1415ZV,sigma1415ZV,bins=np.linspace(4,20,10),label=label2,ax=ax[1])
	ax[1].tick_params(axis='both',labelsize=fontsize)
	ax[1].set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontname)
	ax[1].set_ylabel(ylabel,fontsize=fontsize,fontfamily=fontname)
	ax[1].set_title(titleRight,fontsize=fontsize,fontfamily=fontname)
	ax[1].yaxis.tick_right()
	ax[1].yaxis.set_label_position(rightAxisPostion)
	plt.subplots_adjust(right=right,left=left,wspace=wspace,hspace=hspace,bottom=bottom,top=top)

# Distribution of individual void profiles
def plotProfileDistribution(filterToUse,centres,radii,snap,rBinStack,pairCounts,volumesList,method="poisson",errorType="Weighted",fontname="serif",textwidth=7.1014,textheight=9.0971,fontsize=8,heightRatio=0.5,aspect_ratio=0.5*(3.0/4.0),ylim = [0.0,1.4],shadeColour='b',shadeAlpha=0.2,guideColour='0.75',guideStyle='--',individualLineColour='r',individualLineStyle=':',labels=['Std. Dev.','Mean Profile','Samples'],frameon=False,xlabel="$R/R_{\mathrm{eff}}$",ylabel="$\\rho/\\bar{\\rho}$",left=0.2,bottom=0.2,right=0.95,top=None):
	fig, ax = plt.subplots(figsize=(heightRatio*textwidth,aspect_ratio*textwidth))
	[nbarj,sigma] = stackVoidsWithFilter(centres,radii,filterToUse,snap,rBinStack,nPairsList = pairCounts,volumesList=volumesList,method=method,errorType=errorType)
	N = len(filterToUse)
	rBinStackCentres = plot_utilities.binCentres(rBinStack)
	line1 = ax.fill_between(rBinStackCentres,(nbarj + sigma*np.sqrt(N))/nbar,y2=(nbarj - sigma*np.sqrt(N))/nbar,alpha=shadeAlpha,color=shadeColour,linestyle='None')
	ax.plot(rBinStack,np.ones(rBinStack.shape),color=guideColour,linestyle=guideStyle)
	ax.plot([1,1],ylim,color=guideColour,linestyle=guideStyle)
	line2 = ax.errorbar(rBinStackCentres,nbarj/nbar,yerr=sigma/nbar)
	for k in range(0,10):
		ax.plot(rBinStackCentres,pairCountsAH[filterToUse[k]]/(nbar*volumesListAH[filterToUse[k]]),color=individualLineColour,linestyle=individualLineStyle)
	line3 = Line2D([0],[0],color=individualLineColour,linestyle=individualLineStyle)
	ax.legend([line1,line2,line3],labels,prop={"size":fontsize,"family":fontname},frameon=frameon)
	ax.set_ylim(ylim)
	ax.set_xlim([np.min(rBinStack),np.max(rBinStack)])
	ax.tick_params(axis='both',labelsize=fontsize)
	ax.set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontname)
	ax.set_ylabel(ylabel,fontsize=fontsize,fontfamily=fontname)
	plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top)

# Violin plots with density profile overlayed:
def plotViolinsAndProfile(filterToUse,rho,rBinCentres,rhoMean,rhoSigma,fontname="serif",textwidth=7.1014,textheight=9.0971,fontsize=8,heightRatio=0.45,widthRatio=0.3,violinColour = (1.0, 0.6470588235294118, 0.0, 0.5),ylim=2,inner=None,linewidth=0.1,saturation=1.0,palette=None,alpha1=0.5,alpha2=0.2,meanColour='#0F2080',label1="Profile mean",label2="Weighted error",label3='$\\rho$ distribution',left=0.16,right=0.98,bottom=0.18,top=0.96,frameon=False,legLoc="upper left"):
	fig, ax = plt.subplots(figsize=(heightRatio*textwidth,widthRatio*textwidth))
	plotViolins(rho[filterToUse,0:],rBinCentres,ylim=ylim,ax=ax,fontsize=fontsize,fontname=fontname,color=violinColour,inner=inner,linewidth=linewidth,saturation=saturation,palette=palette)
	plt.setp(ax.collections,alpha=alpha1)
	xticks = plt.xticks()[0]
	pl1 = ax.plot(xticks,rhoMean,label=label1,color=meanColour)
	pl2 = ax.fill_between(plt.xticks()[0],(rhoMean + rhoSigma),y2=(rhoMean - rhoSigma),alpha=alpha2,color=meanColour,linestyle='None',label=label2)
	pl3 = ax.fill_between(plt.xticks()[0],np.NaN*np.ones(len(plt.xticks()[0])),label=label3,color=violinColour)
	plt.subplots_adjust(left=left,right=right,bottom=bottom,top=top)
	ax.legend([pl1[0],pl2,pl3],[label1,label2,label3],prop={"size":fontsize,"family":fontname},frameon=frameon,loc=legLoc)


# Compare linear and non-linear profiles in different radial bins:
def plotRadialBinComparison(rLowLeft,rHighLeft,massesAH,massesZV,radiiAH,radiiZV,radiiAHL,radiiZVL,centresAH,centresAHL,centresZV,centresZVL,pairCountsAH,pairCountsAHL,pairCountsZV,pairCountsZVL,volumesListAH,volumesListAHL,volumesListZV,volumesListZVL,snap,massBoundaries = np.array([1e13,1e15]),rLowRight=None,rHighRight = None,widthRatio=0.4,heightRatio=1.0,fontname="serif",textwidth=7.1014,textheight=9.0971,fontsize=8,colourList=['#F5793A','#0F2080'],listType="mass",ylim=[0,1.7],titleLeftBase="Anti-halos, ",titleRightBase="ZOBOV Voids, ",conditionAH=True,conditionZV=True,set1Label='Non-linear',set2Label='Linear',formatList=['-',':'],guideColor='0.75',method="poisson",errorType="Weighted",legLoc='lower right',frameon=False,wspace=0.0,bottom=0.17,left=0.1,right=0.97,top=None,hspace=None):
	fig, ax = plt.subplots(1,2,figsize=(heightRatio*textwidth,widthRatio*textwidth))
	if rLowRight is None:
		rLowRight = rLowLeft
	if rHighRight is None:
		rHighRight = rHighLeft
	if rLowLeft == 0:
		titleLeft = titleLeftBase + "$<" + str(rHighLeft) + "\\,\\mathrm{Mpc}\\,h^{-1}$"
	else:
		titleLeft = titleLeftBase + "$" + str(rLowLeft) + "\\mathrm{-}" + str(rHighLeft) + "\\,\\mathrm{Mpc}\\,h^{-1}$"
	if rLowRight == 0:
		titleRight = titleRightBase + "$<" + str(rHighRight) + "\\,\\mathrm{Mpc}\\,h^{-1}$"
	else:
		titleRight = titleRightBase + "$" + str(rLowRight) + "\\mathrm{-}" + str(rHighRight) + "\\,\\mathrm{Mpc}\\,h^{-1}$"
	nbar = len(snap)/(snap.properties['boxsize'].ratio("Mpc a h**-1"))**3
	plotStackComparison(massBoundaries,massesAH,massesAH,snap,nbar,radiiAH,centresAH,radiiAHL,centresAHL,pairCountsAH,volumesListAH,volumesListAHL,pairCountsAHL,colourList,rBinStack,listType = listType,ylim=ylim,fontsize=fontsize,returnAx = False,title=titleLeft,conditionAH = conditionAH & (radiiAH < rHighLeft) & (radiiAH >= rLowLeft),conditionZV = conditionAH  & (radiiAH < rHighLeft) & (radiiAH >= rLowLeft),cycle="colour",set1Label=set1Label,set2Label=set2Label,ax=ax[0],fontname=fontname,includeLegend=False,legendFontSize=fontsize,shortFormLegend=True,formatList=formatList,legendCols=1,guideColor=guideColor,method=method,errorType=errorType)
	plotStackComparison(massBoundaries,massesZV,massesZV,snap,nbar,radiiZV,centresZV,radiiZVL,centresZVL,pairCountsZV,volumesListZV,volumesListZVL,pairCountsZVL,colourList,rBinStack,listType = listType,ylim=ylim,fontsize=fontsize,returnAx = False,title=titleRight,conditionAH = conditionZV & (radiiZV < rHighRight) & (radiiZV >= rLowRight),conditionZV = conditionZV & (radiiZV < rHighRight) & (radiiZV >= rLowRight),cycle="colour",set1Label=set1Label,set2Label=set2Label,ax=ax[1],includeLegend=False,fontname=fontname,legendFontSize=fontsize,includeYLabel=False,shortFormLegend=True,legendCols=1,formatList=formatList,guideColor=guideColor,method=method,errorType=errorType)
	fakeLines = [Line2D([0],[0],color='k',linestyle='-'),Line2D([0],[0],color='k',linestyle=':'),Line2D([0],[0],color=colourList[0],linestyle='-'),Line2D([0],[0],color=colourList[1],linestyle='-')]
	ax[0].legend(fakeLines[0:2],[set1Label,set2Label],prop={"size":fontsize,"family":fontname},loc=legLoc,frameon=frameon)
	ax[1].axes.get_yaxis().set_visible(False)
	plt.subplots_adjust(wspace=wspace,bottom=bottom,left=left,right=right,top=top,hspace=hspace)
	ax[0].xaxis.get_major_ticks()[-2].set_visible(False)


# Quick helper function to compute and plot a histogram:
def plotHistogram(x,bins,density=True):
	[pDelta,sigmaDelta,nDelta,nBins] = plot_utilities.computeHistogram(x,bins,density=density)
	plot.histWithErrors(pDelta,sigmaDelta,bins)
	
# Compare data to a gamma distribution and plot best fit distribution:	
def testGammaAndPlot(x,bins,density=True,arguments="fit"):
	mu = np.mean(x)
	sigma = np.sqrt(np.var(x))
	if arguments=="fit":
		args = scipy.stats.gamma.fit(x)
	elif arguments == "mean":
		args=(mu**2/sigma**2,0,sigma**2/mu)
	else:
		args=arguments
	test = kstest(x,'gamma',args=args)
	print(test)
	plotHistogram(x,bins,density=density)
	rangeList = np.linspace(np.min(bins),np.max(bins),1000)
	plt.plot(rangeList,scipy.stats.gamma.pdf(rangeList,a=args[0],scale=args[2]))

# Compare data to a normal distribution and plot best fit distribution:	
def testNormalAndPlot(x,bins,density=True):
	mu = np.mean(x)
	sigma = np.sqrt(np.var(x))
	args=(mu,sigma)
	test = kstest(x,'norm',args=args)
	print(test)
	plotHistogram(x,bins,density=density)
	rangeList = np.linspace(np.min(bins),np.max(bins),1000)
	plt.plot(rangeList,scipy.stats.norm.pdf(rangeList,loc=args[0],scale=args[1]))


# Construct a truncated colourmap, removing some colours from a map and returning a new colourmap:
import matplotlib.colors as colors
def truncate_colormap(cmap,minval=0.0,maxval=1.0,n=100,colourList = None,uneven=False):
	if colourList is None:
		colourList = np.linspace(minval,maxval,n)
	if uneven:
		colours = cmap(colourList)
		new_cmap = colors.ListedColormap(colours)
	else:
		colours = cmap(colourList)
		new_cmap = colors.LinearSegmentedColormap.from_list(cmap.name,colours,N=n)
	return new_cmap

# Return the boundaries of each discrete colour:	
def colourBoundaries(midpoints):
	boundaries = np.zeros(len(midpoints)+1)
	boundaries[-1] = midpoints[-1]
	boundaries[0] = midpoints[0]
	boundaries[1:-1] = (midpoints[0:-1] + midpoints[1:])/2
	return boundaries


# Plot the correlation function:
def plotCrossCorrelations(rBins,xiAM,xiVM,xiAV,ax=None,rMin=0,rMax = np.inf):
	rBinCentres = plot_utilities.binCentres(rBins)
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

#Plot a void profile against its prediction:
def plotProfileVsPrediction(nbarj,sigambarj,rBins,nbar,fontsize=14,ax=None,formatAxis=True,model="Hamaus",color='r',labelSuffix = ""):
	if model == "Hamaus":
		[rs,dc,srs,sdc] = profileParamsHamaus(nbarj,sigambarj,rBins,nbar)
		params = [rs,dc]
	elif model == "Nadathur":
		[alpha,beta,rs,deltac] = profileParamsNadathur(nbarj,rBins,nbar)
		params = [alpha,beta,rs,deltac]
	else:
		raise Exception("Unknown profile model.")
	if ax is None:
		fig, ax = plt.subplots()
	rBinCentres = plot_utilities.binCentres(rBins)
	ax.errorbar(rBinCentres,nbarj/nbar,yerr = sigambarj/nbar,fmt='x-',color=color,label='Computed profile' + labelSuffix)
	ax.plot(rBinCentres,profileModel(rBinCentres,model,params),'--',color=color,label='Predicted profile' + labelSuffix)
	if model == "Hamaus":
		paramspp = [rs + srs,dc + sdc]
		paramspm = [rs + srs,dc - sdc]
		paramsmp = [rs - srs,dc + sdc]
		paramsmm = [rs - srs,dc - sdc]
		profpp = profileModel(rBinCentres,model,paramspp)
		profpm = profileModel(rBinCentres,model,paramspm)
		profmp = profileModel(rBinCentres,model,paramsmp)
		profmm = profileModel(rBinCentres,model,paramsmm)
		proffComb = np.vstack((profpp,profpm,profmp,profmm))
		profUpp = np.max(proffComb,0)
		profLow = np.min(proffComb,0)
		ax.fill_between(rBinCentres,profLow,profUpp,facecolor=color,alpha=0.5)
	if formatAxis:
		ax.set_xlabel('$r/r_{\\mathrm{eff}}$',fontsize=fontsize)
		ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=fontsize)
		ax.tick_params(axis='both',labelsize=fontsize)
		ax.legend(prop={"size":fontsize})


