# CONFIGURATION
from void_analysis import plot, snapedit, tools, simulation_tools, halos
from void_analysis import stacking, real_clusters, survey, cosmology
from void_analysis.tools import loadOrRecompute
from void_analysis.simulation_tools import ngPerLBin
from void_analysis.simulation_tools import biasNew, biasOld
from void_analysis.survey import radialCompleteness, surveyMask
from void_analysis import plot_utilities
import seaborn as sns
seabornColormap = sns.color_palette("colorblind",as_cmap=True)
import numpy as np
import scipy
import pynbody
import astropy
import pickle
import gc
import copy
import h5py
import healpy
import matplotlib.pylab as plt
import os
import sys
import alphashape


# KE correction used to compute magnitudes. Used by PPTs:
def keCorr(z,fit = [-1.456552772320231,-0.7687913554110967]):
    #return -fit[0]*z# - fit[1]
    return 2.9*z

# Convert per-voxel to per-healpix patch average galaxy counts:
def getAllNgsToHealpix(ngList,hpIndices,sampleList,sampleFolder,nside,\
        recomputeData=False,nres=256):
    return [loadOrRecompute(sampleFolder + "sample" + str(sampleList[k]) + \
        "/ngHP.p",tools.getCountsInHealpixSlices,ngList[k],hpIndices,\
        nside=nside,nres=nres,_recomputeData=recomputeData) \
        for k in range(0,len(sampleList))]


# Compute the average posterior value of the amplitude:
def getAalpha(mask,hpIndices,ngHPMCMC,nsamples,nMagBins,npixels,nRadialSlices):
    Aalpha = np.zeros((nsamples,nMagBins,npixels*nRadialSlices))
    inverseLambdaTot = np.zeros((nsamples,nMagBins,npixels*nRadialSlices))
    # Sum the mask of the voxels in each healpix patch. Only where this 
    # sums to zero is lambda_bar_tot actually zero:
    healpixMask = tools.getCountsInHealpixSlices(mask,hpIndices,\
        nside=nside,nres=N)
    # collection properly???
    nz = np.where(healpixMask > 1e-300)
    for k in range(0,nsamples):
        #nz = np.where(ngHPMCMC[k] != 0.0)
        # Formally, it's probably better to remove these pixels using the
        # mask instead, in case there are any errors that make lambda zero:
        # NB - rounding errors in the mask calculation sometimes give 
        # tiny values of the mask instead of zero. These should also be 
        # formally removed, so we should cut to some small value rather than
        # doing a floating point comparison with zero which can be error 
        # prone.
        inverseLambdaTot[k][nz] = 1.0/ngHPMCMC[k][nz]
    Aalpha = inverseLambdaTot*ngHP
    return [Aalpha,inverseLambdaTot]

# Generate data for the PPT plots:
def getPPTPlotData(nBins = 31,nClust=9,nMagBins = 16,N=256,\
        restartFile = 'new_chain_restart/merged_restart.h5',\
        snapNumList = [7000, 7200, 7400],samplesFolder = 'new_chain/',\
        surveyMaskPath = "./2mpp_data/",\
        Om0 = 0.3111,Ode0 = 0.6889,boxsize = 677.7,h=0.6766,Mstarh = -23.28,\
        mmin = 0.0,mmax = 12.5,recomputeData = False,rBinMin = 0.1,\
        rBinMax = 20,abell_nums = [426,2147,1656,3627,3571,548,2197,2063,1367],\
        nside = 4,nRadialSlices=10,rmax=600,tmppFile = "2mpp_data/2MPP.txt",\
        reductions = 4,iterations = 20,verbose=True,hpIndices=None,\
        centreMethod="density",catFolder="",\
        snapname="/gadget_full_forward_512/snapshot_001",\
        data_folder = './',invVarWeighting=False,error="bootstrap",\
        num_samples=10000,bootstrapInterval=[2.5,97.5]):
    # Parameters:
        # nBins - Number of radial bins for galaxy counts in PPT
        # nClust - Number of clusters to do PPTs for.
        # nMagBins - Total number of absolute and apparent magnitude cuts used
        #            by BORG.
        # N - resolution of BORG constraints
        # restartFile - restart file for the BORG MCMC run. Used to extract the 
        #               healpix map.
        # snapNumList - MCMC samples used in the PPT analysis.
        # samplesFolder - folder where MCMC samples are found. 
        # surveyMaskPath - folder where 2MPP survey mask data is stored.
        # Om0 - Matter density fraction LCDM parameter
        # Ode0 - Dark energy density fraction LCDM parameter
        # boxsize - size of periodic domain in constrained simulations.
        # h - Hubble rate in units of 100 kms^{-1}Mpc^{-1}
        # Mstarh - Parameter of the Schecter function used by the survey mask
        # mmin - minimum apparent magnitude of objects used by BORG
        # mmax - maximum apparent magnitude of objects used by BORG
        # recomputeData - If true, recompute all data from scratch, rather than
        #               using cached data.
        # rBinMin - Minimum radius for PPTs
        # rBinMax - Maximum radius for PPTs.
        # abell_nums - Abell numbers of the clusters to perform PPTs for.
        # nside - nside parameter for healpix patches on the sky.
        # nRadialSlices - number of radial slices over each healpix patch
        # rmax - maximum radius for healpix slices
        # tmppFile - 2M++ catalogue file.
        # reductions - Number of times to reduce in half the sphere radius
        #               when computing the centres of mass.
        # iterations - Number of iterations per reduction when computing the
        #               centre of mass.
        # verbose - print indications of progress
    # Cosmology:
    try:
        cosmo = astropy.cosmology.LambdaCDM(100*h,Om0,Ode0)
        # 2M++ catalogue data, and cluster locations:
        [combinedAbellN,combinedAbellPos,abell_nums] = \
            real_clusters.getCombinedAbellCatalogue(Om0 = 0.3111,Ode0 = 0.6889,\
                h=0.6766,catFolder=catFolder)
        clusterInd = [np.where(combinedAbellN == n)[0] for n in abell_nums]
        clusterLoc = np.zeros((len(clusterInd),3))
        for k in range(0,len(clusterInd)):
            if len(clusterInd[k]) == 1:
                clusterLoc[k,:] = combinedAbellPos[clusterInd[k][0],:]
            else:
                # Average positions:
                clusterLoc[k,:] = np.mean(combinedAbellPos[clusterInd[k],:],0)
        # Bias model data for each MCMC sample, and density fields:
        if verbose:
            print("Loading bias data...")
        biasData = [h5py.File(samplesFolder + "/sample" + str(k) + "/mcmc_" + \
            str(k) + ".h5",'r') for k in snapNumList]
        mcmcDen = [1.0 + sample['scalars']['BORG_final_density'][()] \
            for sample in biasData]
        if N < mcmcDen[0].shape[0]:
            print("Warning!!: requested resolution is lower than " + 
            "the MCMC file. Density field will be downgraded.")
            mcmcDen = [tools.downsample(den,int(den.shape[0]/N)) \
                for den in mcmcDen]
        elif N > mcmcDen[0].shape[0]:
            print("Warning!!: requested resolution is higher than " + 
            "the MCMC file. Density field will be interpolated.")
            mcmcDen = [scipy.ndimage.zoom(den,N/den.shape[0]) \
                for den in mcmcDen]
        mcmcDenLin = [np.reshape(den,N**3) for den in mcmcDen]
        mcmcDen_r = [np.reshape(den,(N,N,N),order='F') for den in mcmcDenLin]
        mcmcDenLin_r = [np.reshape(den,N**3) for den in mcmcDen_r]
        biasParam = [np.array([[sample['scalars']['galaxy_bias_' + str(k)][()] \
            for k in range(0,nMagBins)]]) for sample in  biasData]
        # Positions of density voxels, and KDtree for rapid access:
        if verbose:
            print("Computing KD-tree...")
        grid = snapedit.gridListPermutation(N,perm=(2,1,0))
        centroids = grid*boxsize/N + boxsize/(2*N)
        positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
        tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,\
            boxsize),boxsize=boxsize)
        # Survey mask data:
        if verbose:
            print("Computing survey mask...")
        surveyMask11 = healpy.read_map(surveyMaskPath + \
            "completeness_11_5.fits")
        surveyMask12 = healpy.read_map(surveyMaskPath + \
            "completeness_12_5.fits")
        [mask,angularMask,radialMas,mask12,mask11] = tools.loadOrRecompute(\
            data_folder + "surveyMask.p",surveyMask,\
            positions,surveyMask11,surveyMask12,cosmo,-0.94,\
            Mstarh,keCorr = keCorr,mmin=mmin,numericalIntegration=True,\
            mmax=mmax,splitApparent=True,splitAbsolute=True,\
            returnComponents=True,_recomputeData=recomputeData)
        # Obtain galaxy counts:
        nsamples = len(snapNumList)
        galaxyNumberCountExp = np.zeros((nBins,nClust,nMagBins))
        galaxyNumberCountExpShells = np.zeros((nBins,nClust,nMagBins))
        interval2MPPBootstrap = np.zeros((nBins,nClust,len(bootstrapInterval),\
            nMagBins))
        interval2MPPBootstrapShells = np.zeros((nBins,nClust,\
            len(bootstrapInterval),nMagBins))
        galaxyNumberCountsRobustAll = np.zeros((nBins,nClust,nsamples,\
            nMagBins))
        galaxyNumberCountsRobust = np.zeros((nBins,nClust,nMagBins))
        galaxyNumberCountsRobustAllShells = \
            np.zeros((nBins,nClust,nsamples,nMagBins))
        galaxyNumberCountsRobustShells = np.zeros((nBins,nClust,nMagBins))
        if error == "bootstrap":
            # Need space for a lower and upper limit:
            varianceAL = np.zeros((nBins,nClust,len(bootstrapInterval),\
                nMagBins))
            varianceALShell = np.zeros((nBins,nClust,len(bootstrapInterval),\
                nMagBins))
        else:
            varianceAL = np.zeros((nBins,nClust,nMagBins))
            varianceALShell = np.zeros((nBins,nClust,nMagBins))
        posteriorMassAll = np.zeros((nBins,nClust,nsamples))
        mUnit = Om0*2.7754e11*(boxsize/N)**3
        # Obtain properties:
        rBins = np.linspace(rBinMin,rBinMax,nBins+1)
        wrappedPos = snapedit.wrap(clusterLoc + boxsize/2,boxsize)
        # Healpix indices for each voxel:
        if verbose:
            print("Loading healpix data...")
        if hpIndices is None:
            restart = h5py.File(restartFile)
            hpIndices = restart['scalars']['colormap3d'][()]
        hpIndicesLinear = hpIndices.reshape(N**3)
        # Get predicted galaxy counts in each voxel, for each MCMC sample:
        if verbose:
            print("Computing posterior predicted galaxy counts...")
        ngMCMC = np.vstack([tools.loadOrRecompute(samplesFolder + "sample" + \
                str(snapNumList[k]) + "/ngMCMC.p",ngPerLBin,\
                biasParam,return_samples=True,mask=mask,\
                accelerate=True,N=N,\
                delta = [mcmcDenLin_r[k]],contrast=False,sampleList=[0],\
                beta=biasParam[k][:,:,1],rhog = biasParam[k][:,:,3],\
                epsg=biasParam[k][:,:,2],\
                nmean=biasParam[k][:,:,0],biasModel = biasNew,\
                _recomputeData = recomputeData) \
                for k in range(0,nsamples)])
        # Convert MCMC galaxy counts in voxels to counts in each healpix patch:
        if verbose:
            print("Converting voxel counts to healpix patch counts...")
        ngHPMCMC = tools.loadOrRecompute(data_folder + "ngHPMCMC.p",\
            getAllNgsToHealpix,ngMCMC,\
            hpIndices,snapNumList,samplesFolder,nside,nres=N,\
            _recomputeData=recomputeData)
        # Compute counts in each healpix pixel for 2M++ survey:
        if verbose:
            print("Computing 2M++ galaxy counts...")
        ng2MPP = np.reshape(tools.loadOrRecompute(data_folder + "mg2mppK3.p",\
            survey.griddedGalCountFromCatalogue,\
            cosmo,tmppFile=tmppFile,Kcorrection = True,N=N,\
            _recomputeData=recomputeData),(nMagBins,N**3))
        ngHP = tools.loadOrRecompute(data_folder + "ngHP3.p",\
            tools.getCountsInHealpixSlices,\
            ng2MPP,hpIndices,nside=nside,nres=N,_recomputeData=recomputeData)
        # Amplitudes of the bias model in each healpix patch:
        if verbose:
            print("Computing bias patch amplitudes...")
        npixels = 12*(nside**2)
        [Aalpha,inverseLambdaTot] = getAalpha(mask,hpIndices,ngHPMCMC,\
            nsamples,nMagBins,npixels,nRadialSlices)
        # Get centres of clusters in each MCMC sample:
        if verbose:
            print("Computing cluster centres in each sample...")
        clusterCentresSim = []
        densityList = [np.reshape(den,N**3) for den in mcmcDen_r]
        for k in range(0,nsamples): 
            snapPath = samplesFolder + "sample" + str(snapNumList[k]) + snapname
            clusterCentresSim.append(\
                simulation_tools.getClusterCentres(clusterLoc,\
                snapPath = snapPath,fileSuffix = "clusters1",\
                recompute=recomputeData,density=densityList[k],boxsize=boxsize,\
                positions=positions,positionTree=tree,\
                method="density",reductions=reductions,\
                iterations=iterations))
        if verbose:
            print("Computing radial galaxy counts...")
        # Loop over all bins and give spatial averages of the voxels:
        for k in range(0,nBins):
            print("Doing bin " + str(k+1) + " of " + str(nBins))
            # Indices in the previous radial bin:
            if k == 0:
                indicesLast = [[] for centres in range(0,nClust)]
            else:
                indicesLast = indices
            # Indices at <r:
            indices = np.array(tree.query_ball_point(wrappedPos,rBins[k+1]))
            # Indices in a single shell:
            indicesShell = [np.array(np.setdiff1d(ind1,ind2),dtype=int) \
                for ind1,ind2 in zip(indices,indicesLast)]
            # This was used for summing about individual centres if they 
            # fluctuated between samples, but I don't really think that makes 
            # sense any more if we're looking at the variance of the sum of 
            # samples:
            if centreMethod == "snapshot":
                indicesSample = [np.array(tree.query_ball_point(\
                    -np.fliplr(snapedit.wrap(centres,boxsize)),\
                    rBins[k+1])) for centres in clusterCentresSim]
            elif centreMethod == "density":
                indicesSample = [np.array(tree.query_ball_point(\
                    snapedit.wrap(centres,boxsize),rBins[k+1])) \
                    for centres in clusterCentresSim]
            elif centreMethod == "fixed":
                indicesSample = [indices for centres in clusterCentresSim]
            if np.any(np.array(indices,dtype=bool)):
                for l in range(0,nClust):
                    voxels = np.array(indices[l],dtype=int)
                    if len(voxels != 0):
                        # Evaluate the variance and expectation value
                        # using different methods, for the N(<r) case:
                        [expALSum,varAL] = getVoxelSums(\
                            voxels,hpIndicesLinear,Aalpha,ngMCMC,\
                            inverseLambdaTot,ngHP,error,num_samples,nsamples,\
                            nMagBins,bootstrapInterval=bootstrapInterval)
                        if error == "bootstrap":
                            varianceAL[k,l,:,:] = varAL.reshape((\
                                len(bootstrapInterval),nMagBins))
                        else:
                            varianceAL[k,l,:] = varAL.reshape(nMagBins)
                        galaxyNumberCountsRobust[k,l,:] = expALSum
                        # Now do the N(r), at a fixed shell:
                        [expALSumShell,varALShell] = getVoxelSums(\
                            np.array(indicesShell[l],dtype=int),\
                            hpIndicesLinear,Aalpha,ngMCMC,inverseLambdaTot,\
                            ngHP,error,num_samples,nsamples,nMagBins,\
                            bootstrapInterval=bootstrapInterval)
                        if error == "bootstrap":
                            varianceALShell[k,l,:,:] = \
                                varALShell.reshape((\
                                    len(bootstrapInterval),nMagBins))
                        else:
                            varianceALShell[k,l,:] = varALShell.reshape(\
                                nMagBins)
                        galaxyNumberCountsRobustShells[k,l,:] = expALSumShell
                    # Note - we are sometimes interested in looking at the 
                    # counts in individual samples, so we compute these as well.
                    bootstrapSums = bootstrapGalaxyCounts(ng2MPP,\
                            np.array(indices[l],dtype=int),num_samples)
                    bootstrapSumsShell = bootstrapGalaxyCounts(ng2MPP,\
                            np.array(indicesShell[l],dtype=int),num_samples)
                    interval2MPPBootstrap[k,l,:,:] = np.percentile(\
                            bootstrapSums,bootstrapInterval,\
                            axis=1).reshape((len(bootstrapInterval),nMagBins))
                    interval2MPPBootstrapShells[k,l,:,:] = np.percentile(\
                            bootstrapSumsShell,bootstrapInterval,\
                            axis=1).reshape((len(bootstrapInterval),nMagBins))
                    for m in range(0,nMagBins):
                        galaxyNumberCountExp[k,l,m] = \
                            np.sum(ng2MPP[m][indices[l]])
                        galaxyNumberCountExpShells[k,l,m] = \
                            np.sum(ng2MPP[m][indicesShell[l]])
                        for n in range(0,nsamples):
                            # Counts in a specific sample:
                            galaxyNumberCountsRobustAll[k,l,n,m] += np.sum(\
                                Aalpha[n,m,hpIndicesLinear[indices[l]]]*\
                                ngMCMC[n,m][indices[l]])
                            galaxyNumberCountsRobustAllShells[k,l,n,m] += \
                                np.sum(\
                                Aalpha[n,m,hpIndicesLinear[indicesShell[l]]]*\
                                ngMCMC[n,m][indicesShell[l]])
                            posteriorMassAll[k,l,n] = np.sum(\
                                mcmcDenLin_r[n][indices[l]]*mUnit)
    except:
        # Make sure we delete any variables
        raise Exception("An exception occured.")
    finally:
        # Delete variables. Python doesn't seem to garbage collect these
        # manually for some reason:
        del mask, ngMCMC, ng2MPP, ngHP, ngHPMCMC, hpIndicesLinear
        del positions, surveyMask11, surveyMask12, tree, centroids, grid
        del angularMask, radialMas, mask12, mask11
    return [galaxyNumberCountExp,galaxyNumberCountExpShells,\
        interval2MPPBootstrap,interval2MPPBootstrapShells,\
        galaxyNumberCountsRobust,\
        galaxyNumberCountsRobustAll,posteriorMassAll,\
        Aalpha,varianceAL,varianceALShell,\
        galaxyNumberCountsRobustAllShells,\
        galaxyNumberCountsRobustShells]

def bootstrapGalaxyCounts(counts,voxels,numBootstrapSamples,randomSeed=1000):
    # Set random seed so we get consistent results on re-running:
    np.random.seed(randomSeed)
    # Choose voxels with replacement for numBootstrapSamples bootstrap samples:
    bootstrap_indices = voxels[np.random.choice(len(voxels),\
        size=(numBootstrapSamples,len(voxels)),replace=True)]
    # Select these voxels. counts will be a (numMagBins,256^3), so this
    # sampling gives us a (numMagBins,numBootstrapSamples,numVoxels) array:
    bootstrap_samples = counts[:,bootstrap_indices]
    # Sum over all the voxels in every bootstrap sample of voxels to obtain
    # the sum for that bootstrap sample:
    bootstrapSums = np.sum(bootstrap_samples,2)
    return bootstrapSums

# Sum up the specified voxels, computing variance using different methods:
def getVoxelSums(voxels,hpIndicesLinear,Aalpha,ngMCMC,inverseLambdaTot,ngHP,\
        error,num_samples,nsamples,nMagBins,bootstrapInterval=[2.5,97.5],\
        returnDistributionData = False):
    # First, compute the average of A\bar{\lambda} over samples:
    # Healpix indices associated with these voxels:
    hpInd = hpIndicesLinear[voxels]
    # Expectation value of A\bar{\lambda}_i:
    expAL = np.mean(Aalpha[:,:,hpInd]*ngMCMC[:,:,voxels],0)
    # Spatial average:
    # Straight sum (no weights):
    expALSum = np.sum(expAL,1)
    if error == "variance":
        # Boot strap variance 
        varAL = np.var(Aalpha[:,:,hpInd]*ngMCMC[:,:,voxels],0,\
            ddof=1)
    elif error == "bootstrap":
        # We want to sample the possible sums in this set:
        bootstrap_samples = np.zeros((num_samples, nsamples,\
             nMagBins,len(voxels)))
        means = Aalpha[:,:,hpInd]*ngMCMC[:,:,voxels]
        bootstrap_indices = np.random.choice(nsamples,\
            size=(num_samples,nsamples),replace=True)
        bootstrap_samples = means[bootstrap_indices,:,:]
        # TODO: We could probably optimise the following code 
        # a bit better:
        #for i in range(num_samples):
        #    bootstrap_indices = np.random.choice(nsamples,\
        #        size=nsamples,replace=True)
        #    bootstrap_samples[i,:,:,:] = \
        #        means[bootstrap_indices,:,:]
        # Spatial sum over voxels for all samples:
        bootstrapSums = np.sum(bootstrap_samples,3)
        bootstrap_means = np.mean(bootstrapSums,1)
        bootstrap_stds = np.std(bootstrapSums,1,ddof=1)
        # Confidence interval for the means:
        varAL = np.percentile(bootstrap_means, bootstrapInterval,\
            axis = 0)
    else:
        varAL = ngHP[:,hpInd]*(ngHP[:,hpInd] + 1.0)*\
            np.mean(\
            (inverseLambdaTot[:,:,hpInd]*\
            ngMCMC[:,:,voxels])**2,0)+ - expAL**2
    if error == "bootstrap" and returnDistributionData:
        return [expALSum,varAL,bootstrap_indices,bootstrap_samples,\
            bootstrapSums,bootstrap_means,bootstrap_stds]
    else:
        return [expALSum,varAL]

def getPPTForPoints(points,nBins = 31,nClust=9,nMagBins = 16,N=256,\
        restartFile = 'new_chain_restart/merged_restart.h5',\
        snapNumList = [7000, 7200, 7400],samplesFolder = 'new_chain/',\
        surveyMaskPath = "./2mpp_data/",\
        Om0 = 0.3111,Ode0 = 0.6889,boxsize = 677.7,h=0.6766,Mstarh = -23.28,\
        mmin = 0.0,mmax = 12.5,recomputeData = False,rBinMin = 0.1,\
        rBinMax = 20,abell_nums = [426,2147,1656,3627,3571,548,2197,2063,1367],\
        nside = 4,nRadialSlices=10,rmax=600,tmppFile = "2mpp_data/2MPP.txt",\
        reductions = 4,iterations = 20,verbose=True,hpIndices=None,\
        centreMethod="density",catFolder="",data_folder = './',\
        snapname="/gadget_full_forward_512/snapshot_001",\
        returnAll=False):
    # Get tree of density voxel positions:
    grid = snapedit.gridListPermutation(N,perm=(2,1,0))
    centroids = grid*boxsize/N + boxsize/(2*N)
    positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
    tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
        boxsize=boxsize)
    # Cosmology:
    cosmo = astropy.cosmology.LambdaCDM(100*h,Om0,Ode0)
    nsamples = len(snapNumList)
    # Heapix indices:
    if hpIndices is None:
        restart = h5py.File(restartFile)
        hpIndices = restart['scalars']['colormap3d'][()]
    hpIndicesLinear = hpIndices.reshape(N**3)
    # Get bias data:
    biasData = [h5py.File(samplesFolder + "/sample" + str(k) + "/mcmc_" + \
        str(k) + ".h5",'r') for k in snapNumList]
    mcmcDen = [1.0 + sample['scalars']['BORG_final_density'][()] \
        for sample in biasData]
    if N < mcmcDen[0].shape[0]:
        print("Warning!!: requested resolution is lower than " + 
        "the MCMC file. Density field will be downgraded.")
        mcmcDen = [tools.downsample(den,int(den.shape[0]/N)) \
            for den in mcmcDen]
    elif N > mcmcDen[0].shape[0]:
        print("Warning!!: requested resolution is higher than " + 
        "the MCMC file. Density field will be interpolated.")
        mcmcDen = [scipy.ndimage.zoom(den,N/den.shape[0]) \
            for den in mcmcDen]
    mcmcDenLin = [np.reshape(den,N**3) for den in mcmcDen]
    mcmcDen_r = [np.reshape(den,(N,N,N),order='F') for den in mcmcDenLin]
    mcmcDenLin_r = [np.reshape(den,N**3) for den in mcmcDen_r]
    biasParam = [np.array([[sample['scalars']['galaxy_bias_' + str(k)][()] \
        for k in range(0,nMagBins)]]) for sample in  biasData]
    # Survey mask:
    surveyMask11 = healpy.read_map(surveyMaskPath + "completeness_11_5.fits")
    surveyMask12 = healpy.read_map(surveyMaskPath + "completeness_12_5.fits")
    [mask,angularMask,radialMas,mask12,mask11] = tools.loadOrRecompute(\
        data_folder + "surveyMask.p",surveyMask,\
        positions,surveyMask11,surveyMask12,cosmo,-0.94,\
        Mstarh,keCorr = keCorr,mmin=mmin,numericalIntegration=True,\
        mmax=mmax,splitApparent=True,splitAbsolute=True,returnComponents=True,\
        _recomputeData=recomputeData)
    # Get the posterior galaxy counts:
    ngMCMC = np.vstack([tools.loadOrRecompute(samplesFolder + "sample" + \
            str(snapNumList[k]) + "/ngMCMC.p",ngPerLBin,\
            biasParam,return_samples=True,mask=mask,\
            accelerate=True,N=N,\
            delta = [mcmcDenLin_r[k]],contrast=False,sampleList=[0],\
            beta=biasParam[k][:,:,1],rhog = biasParam[k][:,:,3],\
            epsg=biasParam[k][:,:,2],\
            nmean=biasParam[k][:,:,0],biasModel = biasNew,\
            _recomputeData = recomputeData) \
            for k in range(0,nsamples)])
    ngHPMCMC = tools.loadOrRecompute(data_folder + "ngHPMCMC.p",\
        getAllNgsToHealpix,ngMCMC,\
        hpIndices,snapNumList,samplesFolder,nside,nres=N,\
        _recomputeData=recomputeData)
    # Compute counts in each healpix pixel for 2M++ survey:
    ng2MPP = np.reshape(tools.loadOrRecompute(data_folder + "mg2mppK3.p",\
        survey.griddedGalCountFromCatalogue,\
        cosmo,tmppFile=tmppFile,Kcorrection = True,N=N,\
        _recomputeData=recomputeData),(nMagBins,N**3))
    ngHP = tools.loadOrRecompute(data_folder + "ngHP3.p",\
        tools.getCountsInHealpixSlices,\
        ng2MPP,hpIndices,nside=nside,nres=N,_recomputeData=recomputeData)
    # Aalpha:
    npixels = 12*(nside**2)
    Aalpha = np.zeros((nsamples,nMagBins,npixels*nRadialSlices))
    # Sum the mask of the voxels in each healpix patch. Only where this sums to
    # zero is lambda_bar_tot actually zero:
    healpixMask = tools.getCountsInHealpixSlices(mask,hpIndices,nside=nside,\
        nres=N)
    for k in range(0,nsamples):
        #nz = np.where(ngHPMCMC[k] != 0.0)
        # Formally, it's probably better to remove these pixels using the mask
        # instead, in case there are any errors that make lambda zero:
        nz = np.where(healpixMask > 1e-300)
        # NB - rounding errors in the mask calculation sometimes give 
        # tiny values of the mask instead of zero. These should also be 
        # formally removed, so we should cut to some small value rather than
        # doing a floating point comparison with zero which can be error prone.
        Aalpha[k][nz] = ngHP[nz]/ngHPMCMC[k][nz]
    # Perform the PPT:
    galaxyCountExp = np.zeros((nBins,nClust,nMagBins))
    galaxyCountsRobustAll = np.zeros((nBins,nClust,nsamples,nMagBins))
    rBins = np.linspace(rBinMin,rBinMax,nBins+1)
    wrappedPos = snapedit.wrap(points + boxsize/2,boxsize)
    for k in range(0,nBins):
        indices = tree.query_ball_point(wrappedPos,rBins[k+1])
        if np.any(np.array(indices,dtype=bool)):
            for l in range(0,nClust):
                for m in range(0,nMagBins):
                    galaxyCountExp[k,l,m] = np.sum(ng2MPP[m][indices[l]])/\
                        (4*np.pi*rBins[k+1]**3/3)
                    for n in range(0,nsamples):
                        galaxyCountsRobustAll[k,l,n,m] += np.sum(\
                            Aalpha[n,m,hpIndicesLinear[indices[l]]]*\
                            ngMCMC[n,m][indices[l]])/\
                            (4*np.pi*rBins[k+1]**3/3)
    galaxyCountsRobust = np.mean(galaxyCountsRobustAll,2)
    # Convert density to galaxy counts:
    galaxyNumberCountsRobust = (4*np.pi*rBins[1:,None,None]**3/3)*\
        galaxyCountsRobust
    galaxyNumberCountExp = np.array((4*np.pi*rBins[1:,None,None]**3/3)*\
        galaxyCountExp,dtype=int)
    if not returnAll:
        return [galaxyNumberCountExp,galaxyNumberCountsRobust]
    else:
        galaxyNumberCountsAllRobust = (4*np.pi*rBins[1:,None,None,None]**3/3)*\
            galaxyCountsRobustAll
        return [galaxyNumberCountExp,galaxyNumberCountsRobust,\
            galaxyNumberCountsAllRobust]

def getHMFAMFDataFromSnapshots(snapNumList,snapname,snapnameRev,samplesFolder,\
        fileSuffix = '',recomputeData = False,reCentreSnap=True,rSphere=135,\
        boxsize=677.7,verbose=True,recomputeCentres=False,\
        data_folder="./"):
    # Load snapshots:
    if verbose:
        print("Loading snapshots...")
    snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" + \
        snapname) for snapNum in snapNumList]
    snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + snapnameRev) for snapNum in snapNumList]
    if reCentreSnap:
        for snap in snapList:
            tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
            snap.recentred = True
    else:
        for snap in snaplist:
            snap.recentred = False
    # Load or recompute mass and centre data:
    if recomputeCentres:
        if verbose:
            print("Computing halo centres...")
        massesAndCentres512 = loadOrRecompute(data_folder + \
            "all_halo_properties_512" + fileSuffix + ".p",\
            halos.getAllHaloCentresAndMasses,\
            snapList,boxsize,recompute=recomputeData,\
            _recomputeData=recomputeData)
        if verbose:
            print("Computing antihalo centres...")
        antihaloMassesAndCentres512 = loadOrRecompute(data_folder + \
            "all_antihalo_properties_512" + fileSuffix + ".p",\
            halos.getAllAntihaloCentresAndMassesFromSnaplists,\
            snapList,snapListRev,\
            recompute=recomputeData,_recomputeData=recomputeData)
    else:
        # Load from the pre-computed list:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapList]
        massesAndCentres512 = [[\
            tools.remapAntiHaloCentre(props[0],boxsize),props[1]] \
            for props in ahProps]
        antihaloMassesAndCentres512 = [[\
            tools.remapAntiHaloCentre(props[5],boxsize),props[3]] \
            for props in ahProps]
    haloCentres512 = [massesAndCentres512[k][0] \
        for k in range(0,len(snapNumList))]
    antihaloCentres512 = [antihaloMassesAndCentres512[k][0] \
        for k in range(0,len(snapNumList))]
    haloMasses512 = [massesAndCentres512[k][1] \
        for k in range(0,len(snapNumList))]
    antihaloMasses512 = [antihaloMassesAndCentres512[k][1] \
        for k in range(0,len(snapNumList))]
    # Get the halos and antihalos in the centre of the simulations:
    centralHalos512 = [tools.getAntiHalosInSphere(hcentres,135) \
        for hcentres in haloCentres512]
    constrainedHaloMasses512 = [haloMasses512[k][centralHalos512[k][0]] \
        for k in range(0,len(centralHalos512))]
    centralAntihalos512 = [tools.getAntiHalosInSphere(hcentres,135) \
        for hcentres in antihaloCentres512]
    constrainedAntihaloMasses512 = [antihaloMasses512[k][\
        centralAntihalos512[k][0]] \
        for k in range(0,len(centralAntihalos512))]
    # Get Mean density in local supervolume:
    if verbose:
        print("Computing local supervolume density...")
    mUnit = snapList[0]['mass'][0]*1e10
    volSphere = 4*np.pi*rSphere**3/3
    rhoCrit = 2.7754e11
    Om0 = snapList[0].properties['omegaM0']
    rhoMean = rhoCrit*Om0
    deltaList = []
    for k in range(0,len(snapList)):
        snap = pynbody.load(samplesFolder + "sample" \
            + str(snapNumList[k]) + "/" + snapname)
        gc.collect() # Clear memory of the previous snapshot
        tree = tools.getKDTree(snap)
        gc.collect()
        deltaList.append(mUnit*tree.query_ball_point(\
            [boxsize/2,boxsize/2,boxsize/2],rSphere,\
            workers=-1,return_length=True)/(volSphere*rhoMean) - 1.0)
    deltaListMean = np.mean(deltaList)
    deltaListError = np.std(deltaList)/np.sqrt(len(snapList))
    if verbose:
        print("Done.")
    return [constrainedHaloMasses512,constrainedAntihaloMasses512,\
        deltaListMean,deltaListError]

def getUnconstrainedHMFAMFData(snapNumListUncon,snapName,snapNameRev,\
        unconstrainedFolder,deltaListMean,deltaListError,\
        fileSuffix = '',recomputeData=False,boxsize=677.7,verbose=True,\
        reCentreSnaps = True,rSphere=135,nRandCentres = 10000,\
        recomputeCentres = False,randomSeed=1000,meanDensityMethod="selection",\
        meanThreshold=0.02,nRandMax=100):
    # Load unconstrained snaps:
    if verbose:
        print("Loading snapshots...")
    snapListUnconstrained = [pynbody.load(unconstrainedFolder + "sample" \
        + str(snapNum) + "/" + snapName) for snapNum in snapNumListUncon]
    snapListUnconstrainedRev = [pynbody.load(unconstrainedFolder + \
            "sample" + str(snapNum) + "/" +  snapNameRev) \
            for snapNum in snapNumListUncon]
    # recentre snaps:
    # Get halo and antihalo centres:
    if recomputeCentres:
        if reCentreSnaps:
            for snap in snapListUnconstrained:
                tools.remapBORGSimulation(snap)
                snap.recentred = True
        else:
            for snap in snapListUnconstrained:
                snap.recentred = False
        if verbose:
            print("Computing halo centres...")
        massesAndCentres_512uncon = tools.loadOrRecompute(\
            unconstrainedFolder + "all_halo_properties_512" + fileSuffix + \
            ".p",halos.getAllHaloCentresAndMasses,snapListUnconstrained,\
            boxsize,_recomputeData=recomputeData)
        if verbose:
            print("Computing antihalo centres...")
        antihaloMassesAndCentresUn512 = loadOrRecompute(unconstrainedFolder + \
                "all_antihalo_properties_512" + fileSuffix + ".p",\
                halos.getAllAntihaloCentresAndMassesFromSnaplists,\
                snapListUnconstrained,snapListUnconstrainedRev,\
                recompute=recomputeData,_recomputeData=recomputeData)
    else:
        # Load from the pre-computed list:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapListUnconstrained]
        massesAndCentres_512uncon = [[\
            tools.remapAntiHaloCentre(props[0],boxsize),props[1]] \
            for props in ahProps]
        antihaloMassesAndCentresUn512 = [[\
            tools.remapAntiHaloCentre(props[5],boxsize),props[3]] \
            for props in ahProps]
    haloMasses512uncon = [massesAndCentres_512uncon[k][1] \
        for k in range(0,len(snapNumListUncon))]
    antihaloMasses512uncon = [antihaloMassesAndCentresUn512[k][1] \
        for k in range(0,len(snapNumListUncon))]
    haloCentres512uncon = [massesAndCentres_512uncon[k][0] \
        for k in range(0,len(snapNumListUncon))]
    antihaloCentres512uncon = [antihaloMassesAndCentresUn512[k][0] \
        for k in range(0,len(snapNumListUncon))]
    # Find regions with comparable underdensity:
    if verbose:
        print("Computing random centres and overdensities...")
    rhoCrit = 2.7754e11
    Om0 = snapListUnconstrained[0].properties['omegaM0']
    rhoMean = rhoCrit*Om0
    volSphere = 4*np.pi*rSphere**3/3
    # Seed RNG:
    np.random.seed(randomSeed)
    # Get random selection of centres:
    randCentres = np.random.random((nRandCentres,3))*boxsize
    mUnit = snapListUnconstrained[0]['mass'][0]*1e10
    randOverDen = []
    for k in range(0,len(snapListUnconstrained)):
        snap = pynbody.load(unconstrainedFolder + "sample" \
            + str(snapNumListUncon[k]) + "/" + snapName)
        gc.collect() # Clear memory of the previous snapshot
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
        gc.collect()
        randOverDen.append(mUnit*tree.query_ball_point(randCentres,rSphere,\
            workers=-1,return_length=True)/(volSphere*rhoMean) - 1.0)
    # Halos/antihalos in regions with similar underdensity:
    if verbose:
        print("Finding halos in similar density regions...")
    comparableUnfilt = [np.where((delta >= deltaListMean - deltaListError) & \
        (delta < deltaListMean + deltaListError))[0] \
        for delta in randOverDen]
    comparable = [comp[0:np.min([nRandMax,len(comp)])] \
        for comp in comparableUnfilt]
    comparableCentresList = [randCentres[indices,:] for indices in comparable]
    comparableHalos = [[tools.getAntiHalosInSphere(\
        snapedit.wrap(haloCentres512uncon[k] + boxsize/2,boxsize),rSphere,\
        origin = comparableCentresList[k][l]) \
        for l in range(0,len(comparableCentresList[k]))]
        for k in range(0,len(snapListUnconstrained))]
    comparableHaloMasses = [[haloMasses512uncon[k][comparableHalos[k][l][0]] \
        for l in range(0,len(comparableHalos[k]))] \
        for k in range(0,len(snapListUnconstrained))]
    comparableAntihalos = [[tools.getAntiHalosInSphere(\
        snapedit.wrap(antihaloCentres512uncon[k] + boxsize/2,boxsize),rSphere,\
        origin = comparableCentresList[k][l]) \
        for l in range(0,len(comparableCentresList[k]))]
        for k in range(0,len(snapListUnconstrained))]
    comparableAntihaloMasses = [[\
        antihaloMasses512uncon[k][comparableAntihalos[k][l][0]] \
        for l in range(0,len(comparableAntihalos[k]))] \
        for k in range(0,len(snapListUnconstrained))]
    if meanDensityMethod == "central":
        centralHalos = [tools.getAntiHalosInSphere(hcentres,rSphere) \
                for hcentres in haloCentres512uncon]
        centralAntihalos = [tools.getAntiHalosInSphere(hcentres,rSphere) \
                for hcentres in antihaloCentres512uncon]
        centralHaloMasses = [\
                haloMasses512uncon[k][centralHalos[k][0]] \
                for k in range(0,len(centralHalos))]
        centralAntihaloMasses = [\
                antihaloMasses512uncon[k][centralAntihalos[k][0]] \
                for k in range(0,len(centralAntihalos))]
    elif meanDensityMethod == "selection":
        meanDensityRegionsUnfilt = [\
            np.where((delta >= - meanThreshold) & \
            (delta < meanThreshold))[0] \
            for delta in randOverDen]
        meanDensityRegions = [comp[0:np.min([nRandMax,len(comp)])] \
        for comp in meanDensityRegionsUnfilt]
        meanDensityCentresList = [randCentres[indices,:] \
            for indices in meanDensityRegions]
        centralHalos = [[tools.getAntiHalosInSphere(\
            snapedit.wrap(haloCentres512uncon[k] + boxsize/2,boxsize),rSphere,\
            origin = meanDensityCentresList[k][l]) \
            for l in range(0,len(meanDensityCentresList[k]))]
            for k in range(0,len(snapListUnconstrained))]
        centralHaloMasses = [[haloMasses512uncon[k][\
            centralHalos[k][l][0]] \
            for l in range(0,len(centralHalos[k]))] \
            for k in range(0,len(snapListUnconstrained))]
        centralAntihalos = [[tools.getAntiHalosInSphere(\
            snapedit.wrap(antihaloCentres512uncon[k] + boxsize/2,boxsize),\
            rSphere,origin = meanDensityCentresList[k][l]) \
            for l in range(0,len(meanDensityCentresList[k]))]
            for k in range(0,len(snapListUnconstrained))]
        centralAntihaloMasses = [[\
            antihaloMasses512uncon[k][centralAntihalos[k][l][0]] \
            for l in range(0,len(centralAntihalos[k]))] \
            for k in range(0,len(snapListUnconstrained))]
    else:
        raise Exception("Unrecognise mean density method.")
    if verbose:
        print("Done.")
    return [comparableHalos,comparableHaloMasses,\
            comparableAntihalos,comparableAntihaloMasses,\
            centralHalos,centralAntihalos,\
            centralHaloMasses,centralAntihaloMasses]


# Generate HMF/AMF data:
def getHMFAMFData(snapNumList,snapNumListOld,snapNumListUncon,\
        snapNumListUnconOld,samplesFolder="new_chain/",\
        samplesFolderOld = "./",recomputeData=False,\
        boxsize = 677.7,snapnameNew = "gadget_full_forward_512/snapshot_001",\
        snapnameNewRev = "gadget_full_reverse_512/snapshot_001",\
        snapnameOld = "forward_output/snapshot_006",\
        snapnameOldRev = "reverse_output/snapshot_006",\
        unconstrainedFolderNew = "new_chain/unconstrained_samples/",\
        unconstrainedFolderOld = "unconstrainedSamples/",verbose=True,\
        reCentreSnaps = True,Om0=0.3111,rSphere=135,nRandCentres = 10000,\
        meanDensityMethod = "selection",meanThreshold=0.02,\
        data_folder="./"):
    if type(recomputeData) == bool:
        recomputeDataList = [recomputeData for k in range(0,4)]
    elif type(recomputeData) == list:
        if len(recomputeData) != 4:
            raise Exception("Invalid number of elements in recomputeData.")
        recomputeDataList = recomputeData
    else:
        raise Exception("Invalid recomputeData argument")
    # New snapshots, constrained:
    [constrainedHaloMasses512New,constrainedAntihaloMasses512New,\
        deltaListMeanNew,deltaListErrorNew] = tools.loadOrRecompute(\
        data_folder + "constrained_new.p",\
        getHMFAMFDataFromSnapshots,\
        snapNumList,snapnameNew,snapnameNewRev,samplesFolder,\
        recomputeData = recomputeDataList[0],reCentreSnap=reCentreSnaps,\
        rSphere=rSphere,boxsize=boxsize,verbose=verbose,\
        _recomputeData=recomputeDataList[0])
    gc.collect()
    # Old snapshots, constrained:
    [constrainedHaloMasses512Old,constrainedAntihaloMasses512Old,\
        deltaListMeanOld,deltaListErrorOld] = tools.loadOrRecompute(\
        data_folder + "constrained_old.p",\
        getHMFAMFDataFromSnapshots,\
        snapNumListOld,snapnameOld,snapnameOldRev,samplesFolderOld,\
        recomputeData=recomputeDataList[1],reCentreSnap=reCentreSnaps,\
        rSphere=rSphere,boxsize=boxsize,verbose=verbose,\
        fileSuffix = '_old',_recomputeData=recomputeDataList[1])
    gc.collect()
    # Unconstrained halos/antihalos with similar underdensity:
    [comparableHalosNew,comparableHaloMassesNew,\
            comparableAntihalosNew,comparableAntihaloMassesNew,\
            centralHalosNew,centralAntihalosNew,\
            centralHaloMassesNew,centralAntihaloMassesNew] = \
                tools.loadOrRecompute(data_folder + "unconstrained_new.p",\
                    getUnconstrainedHMFAMFData,\
                    snapNumListUncon,snapnameNew,\
                    snapnameNewRev,unconstrainedFolderNew,deltaListMeanNew,\
                    deltaListErrorNew,boxsize=boxsize,\
                    reCentreSnaps = reCentreSnaps,rSphere=rSphere,\
                    nRandCentres=nRandCentres,verbose=verbose,\
                    recomputeData=recomputeDataList[2],\
                    meanThreshold=meanThreshold,\
                    meanDensityMethod=meanDensityMethod,\
                    _recomputeData=recomputeDataList[2])
    gc.collect()
    [comparableHalosOld,comparableHaloMassesOld,\
            comparableAntihalosOld,comparableAntihaloMassesOld,\
            centralHalosOld,centralAntihalosOld,\
            centralHaloMassesOld,centralAntihaloMassesOld] = \
                tools.loadOrRecompute(data_folder + "unconstrained_old.p",\
                    getUnconstrainedHMFAMFData,\
                    snapNumListUnconOld,snapnameOld,\
                    snapnameOldRev,unconstrainedFolderOld,deltaListMeanOld,\
                    deltaListErrorOld,boxsize=boxsize,\
                    reCentreSnaps = reCentreSnaps,rSphere=rSphere,\
                    nRandCentres=nRandCentres,verbose=verbose,\
                    fileSuffix = '_old',recomputeData=recomputeDataList[3],\
                    meanThreshold=meanThreshold,\
                    meanDensityMethod = meanDensityMethod,\
                    _recomputeData=recomputeDataList[3])
    gc.collect()
    return [constrainedHaloMasses512New,constrainedAntihaloMasses512New,\
        deltaListMeanNew,deltaListErrorNew,\
        constrainedHaloMasses512Old,constrainedAntihaloMasses512Old,\
        deltaListMeanOld,deltaListErrorOld,\
        comparableHalosNew,comparableHaloMassesNew,\
        comparableAntihalosNew,comparableAntihaloMassesNew,\
        centralHalosNew,centralAntihalosNew,\
        centralHaloMassesNew,centralAntihaloMassesNew,\
        comparableHalosOld,comparableHaloMassesOld,\
        comparableAntihalosOld,comparableAntihaloMassesOld,\
        centralHalosOld,centralAntihalosOld,\
        centralHaloMassesOld,centralAntihaloMassesOld]



def getPartialPairCountsAndVols(snapNameList,antihaloRadii,antihaloMassesList,\
        ahCentresList,vorVols,rBins,method,\
        rMin,rMax,mMin,mMax,boxsize,filterListToApply=None):
    newPairCounts = []
    newVolumesList = []
    filtersList = []
    for ns in range(0,len(snapNameList)):
        if filterListToApply is None:
            filterToApply = np.where((antihaloRadii[ns] > rMin) & \
                (antihaloRadii[ns] < rMax) & \
                (antihaloMassesList[ns] > mMin) & \
                (antihaloMassesList[ns] <= mMax))[0]
        else:
            filterToApply = filterListToApply[ns]
        snap = tools.getPynbodySnap(snapNameList[ns])
        gc.collect()
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
        gc.collect()
        [pairs,vols] = stacking.getPairCounts(\
            ahCentresList[ns][filterToApply,:],\
            antihaloRadii[ns][filterToApply],snap,rBins,\
            nThreads=-1,tree=tree,\
            method=method,vorVolumes=vorVols[ns])
        newPairCounts.append(pairs)
        newVolumesList.append(vols)
        filtersList.append(filterToApply)
    return [newPairCounts,newVolumesList,filtersList]


def getCentreListUnconstrained(snapListUnconstrained,
        randomSeed = 1000,numDenSamples = 1000,rSphere = 135,\
        densityRange = [-0.051,-0.049]):
    np.random.seed(randomSeed)
    boxsize = snapListUnconstrained[0].properties['boxsize'].ratio(\
        "Mpc a h**-1")
    sampleCentresUnmapped = np.random.random((numDenSamples,3))*boxsize
    sampleCentresMapped = tools.remapAntiHaloCentre(sampleCentresUnmapped,\
        boxsize)
    N = int(np.cbrt(len(snapListUnconstrained[0])))
    Om = snapListUnconstrained[0].properties['omegaM0']
    rhoM = 2.7754e11*Om
    mUnit = rhoM*(boxsize/N)**3
    densitiesInCentres = []
    centreListUn = []
    denListUn = []
    for ns in range(0,len(snapListUnconstrained)):
        snap = pynbody.load(snapListUnconstrained[ns].filename)
        gc.collect()
        tree = tools.getKDTree(snap)
        gc.collect()
        den = mUnit*tree.query_ball_point(\
            sampleCentresUnmapped,\
            rSphere,return_length=True,workers=-1)/\
            (4*np.pi*rhoM*rSphere**3/3) - 1.0
        densitiesInCentres.append(den)
        centreListUn.append(tools.remapAntiHaloCentre(\
            sampleCentresMapped[(den > densityRange[0]) & \
            (den <= densityRange[1]),:],boxsize))
        denListUn.append(den[(den > densityRange[0]) & \
            (den <= densityRange[1])])
        gc.collect()
    return [centreListUn,densitiesInCentres,denListUn]

def getVoidProfilesData(snapNumList,snapNumListUncon,\
        snapList = None,snapListRev = None,\
        samplesFolder="new_chain/",\
        unconstrainedFolder="new_chain/unconstrained_samples/",\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        reCentreSnaps = False,N=512,boxsize=677.7,mMin = 1e14,mMax = 1e15,\
        rMin=5,rMax=25,verbose=True,combineSims=False,\
        method="poisson",errorType = "Weighted",\
        unconstrainedCentreList = np.array([[0,0,0]]),rSphere=135,\
        additionalConditions = None,densityRange=None,numDenSamples = 1000,\
        randomSeed = 1000,redoPairCounts = False,\
        rEffMax=3.0,rEffMin=0.0,nBins=31,pairCountsListUn=None,\
        volumesListUn=None,pairCountsList=None,volumesList=None,\
        ahPropsConstrained = None,ahPropsUnconstrained = None,\
        snapListUnconstrained=None,snapListUnconstrainedRev=None,\
        data_folder = "./",recomputeData=True):
    # Load snapshots:
    if verbose:
        print("Loading snapshots...")
    if snapList is None:
        snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapname) for snapNum in snapNumList]
    if snapListRev is None:
        snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapnameRev) for snapNum in snapNumList]
    # Load unconstrained snaps:
    if verbose:
        print("Loading snapshots...")
    if snapListUnconstrained is None:
        snapListUnconstrained = [pynbody.load(unconstrainedFolder + "sample" \
            + str(snapNum) + "/" + snapname) for snapNum in snapNumListUncon]
    if snapListUnconstrainedRev is None:
        snapListUnconstrainedRev = [pynbody.load(unconstrainedFolder + \
                "sample" + str(snapNum) + "/" + snapnameRev) \
                for snapNum in snapNumListUncon]
    if ahPropsConstrained is None:
        ahPropsConstrained = [tools.loadPickle(snap.filename + \
            ".AHproperties.p") \
            for snap in snapList]
    if ahPropsUnconstrained is None:
        ahPropsUnconstrained = [tools.loadPickle(snap.filename + \
            ".AHproperties.p")\
            for snap in snapListUnconstrained]
    snapNameList = [snap.filename for snap in snapList]
    snapNameListUn = [snap.filename for snap in snapListUnconstrained]
    nbar = (N/boxsize)**3
    # Constrained antihalo properties:
    ahCentresList = [props[5] \
        for props in ahPropsConstrained]
    ahCentresListRemap = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahPropsConstrained]
    antihaloMassesList = [props[3] for props in ahPropsConstrained]
    antihaloRadii = [props[7] for props in ahPropsConstrained]
    deltaCentralList = [props[11] for props in ahPropsConstrained]
    deltaMeanList = [props[12] for props in ahPropsConstrained]
    rBins = np.linspace(rEffMin,rEffMax,nBins)
    rBinStackCentres = plot.binCentres(rBins)
    centralAntihalosCon = [tools.getAntiHalosInSphere(hcentres,rSphere) \
        for hcentres in ahCentresListRemap]
    # Unconstrained antihalo properties:
    ahCentresListUn = [props[5] \
        for props in ahPropsUnconstrained]
    ahCentresListRemapUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahPropsUnconstrained]
    antihaloMassesListUn = [props[3] for props in ahPropsUnconstrained]
    antihaloRadiiUn = [props[7] for props in ahPropsUnconstrained]
    deltaCentralListUn = [props[11] for props in ahPropsUnconstrained]
    deltaMeanListUn = [props[12] for props in ahPropsUnconstrained]
    vorVols = [props[4] for props in ahPropsConstrained]
    vorVolsUn = [props[4] for props in ahPropsUnconstrained]
    if not redoPairCounts:
        if pairCountsListUn is None:
            pairCountsListUn = [props[9] for props in ahPropsUnconstrained]
        if volumesListUn is None:
            volumesListUn = [props[10] for props in ahPropsUnconstrained]
        if pairCountsList is None:
            pairCountsList = [props[9] for props in ahPropsConstrained]
        if volumesList is None:
            volumesList = [props[10] for props in ahPropsConstrained]
    # Density sampling, if specified:
    if densityRange is None:
        centreListUn = unconstrainedCentreList
    else:
        # Sample densities in each sphere to get densities in a particular
        # range:
        np.random.seed(randomSeed)
        if len(densityRange) != 2:
            raise Exception("Invalid densityRange")
        sampleCentresUnmapped = np.random.random((numDenSamples,3))*boxsize
        sampleCentresMapped = tools.remapAntiHaloCentre(sampleCentresUnmapped,\
            boxsize)
        unconstrainedTrees = [tools.getKDTree(snap) \
            for snap in snapListUnconstrained]
        Om = snapListUnconstrained[0].properties['omegaM0']
        rhoM = 2.7754e11*Om
        mUnit = rhoM*(boxsize/N)**3
        densitiesInCentres = [mUnit*tree.query_ball_point(\
            sampleCentresUnmapped,\
            rSphere,return_length=True,workers=-1)/\
            (4*np.pi*rhoM*rSphere**3/3) - 1.0 \
            for tree in unconstrainedTrees]
        centreListUn = [tools.remapAntiHaloCentre(\
            sampleCentresMapped[(den > densityRange[0]) & \
            (den <= densityRange[1]),:],boxsize) for den in densitiesInCentres]
        denListUn = [den[(den > densityRange[0]) & \
            (den <= densityRange[1])] for den in densitiesInCentres]
        lengths = np.array([len(cen) for cen in centreListUn])
        if np.any(lengths < 1):
            print(lengths)
            raise Exception("Did not find random centres with appropriate " + \
                "density. Expand density range, or increase number of samples.")
    if type(centreListUn) == list:
        centralAntihalosUn = [[tools.getAntiHalosInSphere(ahCentresListRemapUn[ns],\
            rSphere,origin=centre) for centre in centreListUn[ns]] \
            for ns in range(0,len(snapListUnconstrained))]
    elif type(centreListUn) == np.ndarray:
        centralAntihalosUn = [[tools.getAntiHalosInSphere(ahCentresListRemapUn[ns],\
            rSphere,origin=centre) for centre in centreListUn] \
            for ns in range(0,len(snapListUnconstrained))]
    else:
        raise Exception("Unconstrained centre list format not understood.")
    if additionalConditions is None:
        additionalConditions = [np.ones(len(centres),dtype=bool) \
            for centres in ahCentresListRemap]
    # Select antihalos in the central region:
    conditionList = [(deltaCentralList[ns] < 0) & \
        (centralAntihalosCon[ns][1]) & additionalConditions[ns] \
        for ns in range(0,len(snapNumList))]
    conditionListUn = [[(deltaCentralListUn[ns] < 0) & \
        (centralAHs[1]) for centralAHs in centralAntihalosUn[ns]] \
        for ns in range(0,len(snapNumListUncon))]
    conditionListMrange = [(deltaCentralList[ns] < 0) & \
        (centralAntihalosCon[ns][1]) & (antihaloMassesList[ns] > mMin) & \
        (antihaloMassesList[ns] <= mMax) & additionalConditions[ns] & \
        (antihaloRadii[ns] > rMin) & \
        (antihaloRadii[ns] <= rMax)
        for ns in range(0,len(snapNumList))]
    conditionListMrangeUn = [[(deltaCentralListUn[ns] < 0) & \
        (centralAHs[1]) & (antihaloMassesListUn[ns] > mMin) & \
        (antihaloMassesListUn[ns] <= mMax) & \
        (antihaloRadiiUn[ns] > rMin) & \
        (antihaloRadiiUn[ns] <= rMax)
        for centralAHs in centralAntihalosUn[ns]] \
        for ns in range(0,len(snapNumListUncon))]
    # Stacked profile data:
    stackedRadii = np.hstack(antihaloRadii)
    stackedMasses = np.hstack(antihaloMassesList)
    stackedConditions = np.hstack(conditionListMrange)
    if (pairCountsList is not None) and (volumesList is not None):
        stackedPairCounts = np.vstack(pairCountsList)
        stackedVolumesList = np.vstack(volumesList)
    else:
        # Recompute only those which we need:
        [newPairCounts,newVolumesList,filtersList] = \
            tools.loadOrRecompute(data_folder + "pair_counts_data.p",\
            getPartialPairCountsAndVols,\
            snapNameList,antihaloRadii,antihaloMassesList,\
            ahCentresList,vorVols,rBins,method,\
            rMin,rMax,mMin,mMax,boxsize,\
            filterListToApply=conditionListMrange,\
            _recomputeData=recomputeData)
        if len(newPairCounts) != len(snapList):
            raise Exception("Pair counts list does not match sample list.")
        stackedPairCounts = np.vstack(newPairCounts)
        stackedVolumesList = np.vstack(newVolumesList)
    [nbarjAllStacked,sigmaAllStacked] = stacking.stackScaledVoids(\
        np.vstack(ahCentresList)[stackedConditions,:],\
        stackedRadii[stackedConditions],\
        snapList[0],rBins,\
        nPairsList = stackedPairCounts,\
        volumesList = stackedVolumesList,\
        method=method,errorType=errorType)
    allPairCountsUn = [[] for ns in range(0,len(snapListUnconstrained))]
    allVolumesListsUn = [[] for ns in range(0,len(snapListUnconstrained))]
    if (pairCountsListUn is not None) and (volumesListUn is not None):
        # Combined stacked withe known pair lists:
        stackedPairCountsUn = np.vstack(pairCountsListUn)
        stackedVolumesListUn = np.vstack(volumesListUn)
        stackedRadiiUn = np.hstack(antihaloRadiiUn)
        stackedMassesUn = np.hstack(antihaloMassesListUn)
        stackedCentresUn = np.vstack(ahCentresListUn)
        combinedUnAll = np.array([],dtype=bool)
    else:
        # Pre-define here. Will be filled below.
        stackedPairCountsUn = np.zeros((0,nBins-1))
        stackedVolumesListUn = np.zeros((0,nBins-1))
        stackedRadiiUn= np.array([])
        stackedMassesUn= np.array([])
        stackedCentresUn = np.zeros((0,3))
    for ns in range(0,len(snapListUnconstrained)):
        for l in range(0,len(centreListUn[ns])):
            if (pairCountsListUn is not None) and (volumesListUn is not None):
                # Use pre-generated pair counts:
                newPairCountsUn = pairCountsListUn[\
                    conditionListMrangeUn[ns][l],:]
                newVolumesListUn = volumesListUn[\
                    conditionListMrangeUn[ns][l],:]
            else:
                # Regenerate them from scratch:
                [newPairCountsUn,newVolumesListUn,filtersListUn] = \
                    tools.loadOrRecompute(data_folder + \
                        "pair_counts_data_unconstrained_sample_" + \
                        str(ns) + "_region_" + str(l) + ".p",\
                        getPartialPairCountsAndVols,\
                        [snapNameListUn[ns]],[antihaloRadiiUn[ns]],\
                        [antihaloMassesListUn[ns]],\
                        [ahCentresListUn[ns]],[vorVolsUn[ns]],rBins,method,\
                        rMin,rMax,mMin,mMax,boxsize,\
                        filterListToApply=[conditionListMrangeUn[ns][l]],\
                        _recomputeData=recomputeData)
                stackedPairCountsUn = np.vstack(\
                    [stackedPairCountsUn,newPairCountsUn[0]])
                stackedVolumesListUn = np.vstack(\
                    [stackedVolumesListUn,newVolumesListUn[0]])
                stackedRadiiUn = np.hstack([stackedRadiiUn,\
                    antihaloRadiiUn[ns][conditionListMrangeUn[ns][l]]])
                stackedMassesUn = np.hstack([stackedMassesUn,\
                    antihaloMassesListUn[ns][conditionListMrangeUn[ns][l]]])
                stackedCentresUn = np.vstack([stackedCentresUn,\
                    ahCentresListUn[ns][conditionListMrangeUn[ns][l],:]])
            allPairCountsUn[ns].append(newPairCountsUn[0])
            allVolumesListsUn[ns].append(newVolumesListUn[0])
    # Stack voids from all bins for the combined unconstrained profile:
    if (pairCountsListUn is not None) and (volumesListUn is not None):
        # Combined stacked withe known pair lists:
        combinedUnAll = np.array([],dtype=bool)
        for ns in range(0,len(snapListUnconstrained)):
            union = np.zeros(len(antihaloRadii[ns]),dtype=bool)
            for l in range(0,len(conditionListMrangeUn[ns])):
                union = np.logical_or(union,conditionListMrangeUn[ns][l])
            combinedUnAll = np.hstack([combinedUnAll,union])
        [nbarjAllStackedUn,sigmaAllStackedUn] = stacking.stackVoidsWithFilter(\
            stackedCentresUn,stackedRadiiUn,combinedUnAll,\
            snapListUnconstrained[0],rBins,\
            nPairsList = stackedPairCountsUn,\
            volumesList = stackedVolumesListUn,\
            method=method,errorType=errorType)
    else:
        # Recompute only those antihalo profiles we need:
        [nbarjAllStackedUn,sigmaAllStackedUn] = stacking.stackScaledVoids(\
            stackedCentresUn,stackedRadiiUn,snapListUnconstrained[0],rBins,\
            nPairsList = stackedPairCountsUn,volumesList = stackedVolumesListUn,\
            method=method,errorType=errorType)
    # Profiles in each constrained region separately:
    if (pairCountsList is not None) and (volumesList is not None):
        pCounts = pairCountsList
        vList = volumesList
        centres = ahCentresList
        radii = antihaloRadii
        masses = antihaloMassesList
        cond = conditionListMrange
    else:
        pCounts = newPairCounts
        vList = newVolumesList
        centres = [ahCentresList[ns][conditionListMrange[ns],:] \
            for ns in range(0,len(snapList))]
        radii = [antihaloRadii[ns][conditionListMrange[ns]] \
            for ns in range(0,len(snapList))]
        masses = [antihaloMassesList[ns][conditionListMrange[ns]] \
            for ns in range(0,len(snapList))]
        cond = None
    [nbarjSepStack,sigmaSepStack] = stacking.computeMeanStacks(\
        centres,radii,masses,cond,pCounts,\
        vList,snapNameList,nbar,rBins,rMin,rMax,mMin,mMax)
    # Profiles in selected unconstrained regions:
    indStack = []
    pCounts = allPairCountsUn
    vList = allVolumesListsUn
    if type(centreListUn) == np.ndarray:
        centreListUn = [centreListUn \
            for ns in range(0,len(snapListUnconstrained))]
    for ns in range(0,len(snapNumListUncon)):
        for l in range(0,len(centreListUn[ns])):
            condition = conditionListMrangeUn[ns][l]
            indStack.append(stacking.stackScaledVoids(
                ahCentresListUn[ns][condition,:],\
                antihaloRadiiUn[ns][condition],\
                snapListUnconstrained[ns],rBins,\
                nPairsList = pCounts[ns][l],\
                volumesList=vList[ns][l]))
    nbarjSepStackUn = np.vstack([sim[0] for sim in indStack])
    sigmaSepStackUn = np.vstack([sim[1] for sim in indStack])
    return [rBinStackCentres,nbarjSepStack,\
        sigmaSepStack,nbarjSepStackUn,sigmaSepStackUn,\
        nbarjAllStacked,sigmaAllStacked,nbarjAllStackedUn,sigmaAllStackedUn,\
        nbar,rMin,mMin,mMax]


# Constructs a grid of points for a given boxsize at which we can evaluate
# densities, SNR etc...
# NB - these positions are designed to work with the positions
# output by the tools.remapAntiHaloCentres fuction when applies to the 
# constrained simulations. If the layout changes, then this function
# will no longer work!
def getGridPositionsAndTreeForSims(boxsize,Nden=256,perm=(2,1,0)):
    grid = snapedit.gridListPermutation(Nden,perm=(2,1,0))
    centroids = grid*boxsize/Nden + boxsize/(2*Nden)
    positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
    tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
        boxsize=boxsize)
    return [positions,tree]


# Construct an SNR filter from a chain file:
def getSNRFilterFromChainFile(chainFile,snrThresh,snapNameList,boxsize,\
        Nden = 256,allProps=None):
    [mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
        std_field,hmc_Elh,hmc_Eprior,hades_accept_count,\
        hades_attempt_count] = tools.loadPickle(chainFile)
    snrField = mean_field**2/std_field**2
    snrFieldLin = np.reshape(snrField,Nden**3)
    if allProps is None:
        allProps = [tools.loadPickle(snapname + ".AHproperties.p") \
           for snapname in snapNameList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in allProps]
    antihaloCentresUnmapped = [props[5] for props in allProps]
    antihaloMasses = [props[3] for props in allProps]
    antihaloRadii = [props[7] for props in allProps]
    # Get positions at which to compute the SNR:
    [positions,tree] = getGridPositionsAndTreeForSims(boxsize,Nden=Nden)
    # Locate the points within one anti-halo effective radii of each void
    # centre. We define the void SNR as the average SNR of these points:
    nearestPointsList = [tree.query_ball_point(\
            snapedit.wrap(antihaloCentres[k] + boxsize/2,boxsize),\
            antihaloRadii[k],workers=-1) \
            for k in range(0,len(antihaloCentres))]
    snrAllCatsList = [np.array([np.mean(snrFieldLin[points]) \
            for points in nearestPointsList[k]]) \
            for k in range(0,len(snapNameList))]
    # Filter those above the threshold:
    snrFilter = [snr > snrThresh for snr in snrAllCatsList]
    return [snrFilter,snrAllCatsList]

# Get alpha shapes for the combined catalogue:
def getFinalCatalogueAlphaShapes(snapNumList,finalCat,verbose=True,\
        samplesFolder = "new_chain/",recomputeData = False,\
        snapForFolder = "gadget_full_forward_512",\
        snapRevFolder = "gadget_full_reverse_512",\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        reCentreSnaps = True,figuresFolder='',\
        snapList = None,snapListRev = None,antihaloCatalogueList=None,\
        snapsortList = None,ahProps = None,massRange = None,rRange = None,\
        additionalFilters = None,rSphere=135,data_folder="./",\
        centralAntihalos = None,alphaVal = 7):
    # Load snapshots:
    try:
        if verbose:
            print("Loading snapshots...")
        if snapList is None:
            snapList = [pynbody.load(samplesFolder + "sample" + \
                str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        if snapListRev is None:
            snapListRev = [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapnameRev) for snapNum in snapNumList]
        if reCentreSnaps:
            for snap in snapList:
                tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
                snap.recentred = True
        if verbose:
            print("Loading antihalo properties...")
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        if antihaloCatalogueList is None:
            antihaloCatalogueList = [snap.halos() for snap in snapListRev]
        if ahProps is None:
            ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
                for snap in snapList]
        antihaloRadii = [props[7] for props in ahProps]
        antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize,\
            swapXZ=False,reverse=True) for props in ahProps]
        antihaloMasses = [props[3] for props in ahProps]
        print("Getting snapsort lists...")
        if snapsortList is None:
            snapsortList = [np.argsort(snap['iord']) \
                for snap in snapList]
        if centralAntihalos is None:
            print("Constructing filters...")
            if rRange is None:
                rRangeCond = [np.ones(len(antihaloRadii[k]),dtype=bool) \
                    for k in range(0,len(snapNumList))]
            else:
                rRangeCond = [(antihaloRadii[k] > rRange[0]) & \
                    (antihaloRadii[k] <= rRange[1]) \
                    for k in range(0,len(snapNumList))]
            if massRange is None:
                mRangeCond = [np.ones(len(antihaloRadii[k]),dtype=bool) \
                    for k in range(0,len(snapNumList))]
            else:
                mRangeCond = [(antihaloMasses[k] > massRange[0]) & \
                    (antihaloMasses[k] <= massRange[1]) \
                    for k in range(0,len(snapNumList))]
            filterCond = [rRangeCond[k] & mRangeCond[k] \
                for k in range(0,len(snapNumList))]
            if additionalFilters is not None:
                filterCond = [filterCond[k] & additionalFilters[k] \
                    for k in range(0,len(snapNumList))]
            print("Selecting antihalos...")
            centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],\
                    rSphere,filterCondition = filterCond[k]) \
                    for k in range(0,len(snapNumList))]
        massSortCentral = [\
            np.flip(np.argsort(antihaloMasses[k][centralAntihalos[k][0]])) \
            for k in range(0,len(centralAntihalos))]
        largeAntihalos = [np.array(centralAntihalos[ns][0],dtype=int)[\
                massSortCentral[ns]] \
                for ns in range(0,len(snapList))]
        if verbose:
            print("Computing alpha shapes...")
        # From here, we have to combined the positions of ALL voids:
        positionLists = [] # Positions of all particles in all voids
        centralAntihaloMasses = [\
                antihaloMasses[k][centralAntihalos[k][0]] \
                for k in range(0,len(centralAntihalos))]
        sortedList = [np.flip(np.argsort(centralAntihaloMasses[ns])) \
            for ns in range(0,len(snapNumList))]
        fullListAll = [np.array(centralAntihalos[ns][0])[sortedList[ns]] \
            for ns in range(0,len(snapNumList))]
        alpha_shapes = []
        ahMWPos = []
        for k in range(0,finalCat.shape[0]):
            allPosXYZ = np.full((0,3),0)
            for ns in range(0,finalCat.shape[1]):
                # Select the correct anti-halo
                fullList = fullListAll[ns]
                listPosition = finalCat[k,ns]-1
                if listPosition >= 0:
                    # Only include anti-halos which we have representatives for
                    # in a given catalogue
                    ahNumber = fullList[listPosition]
                    posXYZ = snapedit.unwrap(
                        snapList[ns]['pos'][snapsortList[ns][\
                        antihaloCatalogueList[ns][\
                        largeAntihalos[ns][ahNumber]+1]['iord']],:],boxsize)
                    allPosXYZ = np.vstack((allPosXYZ,posXYZ))
            posMW = plot_utilities.computeMollweidePositions(allPosXYZ)
            ahMWPos.append(posMW)
            alpha_shapes.append(alphashape.alphashape(
                    np.array([posMW[0],posMW[1]]).T,alphaVal))
    except:
        # Make sure we delete the snapshots!
        for snap in snapList:
            del snap
        for snap in snapListRev:
            del snap
        raise Exception("Error occurred. Deleting snapshots")
    return [ahMWPos,alpha_shapes]

# Get alpha shapes for individual voids:
def getAntihaloSkyPlotData(snapNumList,nToPlot=None,verbose=True,\
        samplesFolder = "new_chain/",recomputeData = False,\
        snapForFolder = "gadget_full_forward_512",\
        snapRevFolder = "gadget_full_reverse_512",\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        reCentreSnaps = True,figuresFolder='',\
        snapList = None,snapListRev = None,antihaloCatalogueList=None,\
        snapsortList = None,ahProps = None,massRange = None,rRange = None,\
        additionalFilters = None,rSphere=135,data_folder="./",\
        centralAntihalos = None):
    # Load snapshots:
    if verbose:
        print("Loading snapshots...")
    if snapList is None:
        snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapname) for snapNum in snapNumList]
    if snapListRev is None:
        snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapnameRev) for snapNum in snapNumList]
    if reCentreSnaps:
        for snap in snapList:
            tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
            snap.recentred = True
    if verbose:
        print("Loading antihalo properties...")
    boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
    if antihaloCatalogueList is None:
        antihaloCatalogueList = [snap.halos() for snap in snapListRev]
    if ahProps is None:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapList]
    antihaloRadii = [props[7] for props in ahProps]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize,\
        swapXZ=False,reverse=True) for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    print("Getting snapsort lists...")
    if snapsortList is None:
        snapsortList = [np.argsort(snap['iord']) \
            for snap in snapList]
    if centralAntihalos is None:
        print("Constructing filters...")
        if rRange is None:
            rRangeCond = [np.ones(len(antihaloRadii[k]),dtype=bool) \
                for k in range(0,len(snapNumList))]
        else:
            rRangeCond = [(antihaloRadii[k] > rRange[0]) & \
                (antihaloRadii[k] <= rRange[1]) \
                for k in range(0,len(snapNumList))]
        if massRange is None:
            mRangeCond = [np.ones(len(antihaloRadii[k]),dtype=bool) \
                for k in range(0,len(snapNumList))]
        else:
            mRangeCond = [(antihaloMasses[k] > massRange[0]) & \
                (antihaloMasses[k] <= massRange[1]) \
                for k in range(0,len(snapNumList))]
        filterCond = [rRangeCond[k] & mRangeCond[k] \
            for k in range(0,len(snapNumList))]
        if additionalFilters is not None:
            filterCond = [filterCond[k] & additionalFilters[k] \
                for k in range(0,len(snapNumList))]
        print("Selecting antihalos...")
        centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],\
                rSphere,filterCondition = filterCond[k]) \
                for k in range(0,len(snapNumList))]
    massSortCentral = [\
        np.flip(np.argsort(antihaloMasses[k][centralAntihalos[k][0]])) \
        for k in range(0,len(centralAntihalos))]
    if nToPlot is None:
        largeAntihalos = [np.array(centralAntihalos[ns][0],dtype=int)[\
            massSortCentral[ns]] \
            for ns in range(0,len(snapList))]
    else:
        largeAntihalos = [np.array(centralAntihalos[ns][0],dtype=int)[\
            massSortCentral[ns][0:nToPlot]] \
            for ns in range(0,len(snapList))]
    if verbose:
        print("Computing alpha shapes...")
    alpha_shape_list = [tools.computeMollweideAlphaShapes(\
        snapList[ns],largeAntihalos[ns],antihaloCatalogueList[ns]) \
        for ns in range(0,len(snapList))]
    return [alpha_shape_list,largeAntihalos,\
        snapsortList]

def zobovVolumesToPhysical(zobovVolumes,snap,dtype=np.double):
    N = np.round(np.cbrt(len(snap))).astype(int)
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if type(zobovVolumes) == type(""):
        vols = np.fromfile(zobovVolumes,dtype=dtype,offset=4)
    else:
        vols = zobovVolumes
    return vols*(boxsize/N)**3


def getMatchPynbody(snap1,snap2,cat1,cat2,quantity1,quantity2,\
        max_index = 200,threshold = 0.5,\
        quantityThresh=0.5,fractionType='normal'):
    bridge = pynbody.bridge.OrderBridge(snap1,snap2,monotonic=False)
    # Ensure ratio is above the threshold:
    match = [-2]
    candidatesList = []
    # get shared particle fractions:
    catTransfer = bridge.catalog_transfer_matrix(groups_1 = cat1,\
        groups_2 = cat2,max_index=max_index)
    cat2Lengths = np.array([len(halo) for halo in cat2])
    for k in range(0,np.min([len(cat1),max_index])):
        # get quantity (mass or radius) ratio, defined as lower/highest
        # so that it is always <= 1:
        bigger = np.where(quantity2 > quantity1[k])[0]
        quantRatio = quantity2/quantity1[k]
        quantRatio[bigger] = quantity1[k]/quantity2[bigger]
        if fractionType == 'normal':
            fraction = catTransfer[k]/len(cat1[k+1])
        elif fractionType == 'symmetric':
            fraction = catTransfer[k]/np.sqrt(len(cat1[k+1]*cat2Lengths))
        else:
            raise Exception("Unrecognised fraction type requested.")
        # Find candidates that satisfy the threshold requirements:
        candidates = np.where((quantRatio >= quantityThresh) & \
            (fraction > threshold))[0]
        candidatesList.append(candidates)
        if len(candidates) < 1:
            match.append(-1)
        else:
            # Select the match with the highest shared particle fraction
            # as the most probable:
            mostProbable = candidates[np.argmax(fraction[candidates])]
            match.append(mostProbable + 1)
        return [np.array(match,dtype=int),candidatesList]

# Function to match a single void:
def getCandidatesForVoid(centre,searchRadius,boxsize,otherCentres,\
        otherTree=None):
    if otherTree is None:
        otherTree = scipy.spatial.cKDTree(snapedit.wrap(otherCentres,boxsize),\
            boxsize=boxsize)
    candidates = otherTree.query_ball_point(snapedit.wrap(centre,boxsize),\
        searchRadius,workers=-1)
    return candidates

# Get the radius/mass ratios of all candidates, without sorting:
def getUnsortedRatios(candidates,searchQuantity,otherQuantities):
    if not np.isscalar(searchQuantity):
        # This happens if we are checking multiple quantities, or possibly
        # multiple thresholds, simultaneously.
        nQuantLen = len(searchQuantity) # Number of quantities being
            # tested
        quantRatio = np.zeros((len(candidates),nQuantLen))
        for l in range(0,nQuantLen):
            bigger = np.where(\
                otherQuantities[candidates,l] > searchQuantity[l])[0]
            quantRatio[:,l] = otherQuantities[candidates,l]/\
                searchQuantity[l]
            quantRatio[bigger,l] = searchQuantity[l]/\
                otherQuantities[candidates,l][bigger]
    else:
        bigger = np.where(\
            otherQuantities[candidates] > searchQuantity)[0]
        quantRatio = otherQuantities[candidates]/searchQuantity
        quantRatio[bigger] = searchQuantity/\
            otherQuantities[candidates][bigger]
    return quantRatio

# Sort the candidates in descending order of radius/mass ratio:

def computeQuantityRatio(otherQuantities,searchQuantity):
    bigger = np.where(\
        otherQuantities > searchQuantity)[0]
    quantRatio = otherQuantities/searchQuantity
    quantRatio[bigger] = searchQuantity/otherQuantities[bigger]
    return quantRatio

def sortQuantRatiosByRatio(quantRatio,distances,candidates,sortMethod,\
        sortQuantity):
    # sort the quantRatio from biggest to smallest, so we find
    # the most similar anti-halo within the search distance:
    if len(quantRatio.shape) > 1:
        indSort = np.flip(np.argsort(quantRatio[:,sortQuantity]))
    else:
        indSort = np.flip(np.argsort(quantRatio))
    quantRatio = quantRatio[indSort]
    sortedCandidates = np.array(candidates,dtype=int)[indSort]
    return [quantRatio,sortedCandidates,indSort]

def sortCandidatesByDistance(candidates,quantRatio,distances,otherQuantities,\
        searchQuantity):
    indSort = np.argsort(distances)
    sortedCandidates = np.array(candidates,dtype=int)[indSort]
    quantRatio = quantRatio[indSort]
    return [quantRatio,sortedCandidates,indSort]

def sortCandidatesByVolumes(candidates,quantRatio,overlapForVoid):
    volOverlapFrac = overlapForVoid[candidates]
    indSort = np.flip(np.argsort(volOverlapFrac))
    quantRatio = quantRatio[indSort]
    sortedCandidates = np.array(candidates,dtype=int)[indSort]
    return [quantRatio,sortedCandidates,indSort]

def getSortedQuantRatio(sortMethod,candidates,quantRatio,distances,
        otherQuantities,searchQuantity,overlapForVoid,sortQuantity):
    if sortMethod == 'distance':
        # Sort the antihalos by distance. Candidate is the closest
        # halo which satisfies the threshold criterion:
        [quantRatio,sortedCandidates,indSort] = sortCandidatesByDistance(\
            candidates,quantRatio,distances,otherQuantities,searchQuantity)
    elif sortMethod == 'ratio':
        # sort the quantRatio from biggest to smallest, so we find
        # the most similar anti-halo within the search distance:
        [quantRatio,sortedCandidates,indSort] = sortQuantRatiosByRatio(\
            quantRatio,distances,candidates,sortMethod,sortQuantity)
    elif sortMethod == "volumes":
        [quantRatio,sortedCandidates,indSort] = sortCandidatesByVolumes(\
            candidates,quantRatio,overlapForVoid)
    else:
        raise Exception("Unrecognised sorting method")
    return [quantRatio,sortedCandidates,indSort]

def getVoidsAboveThresholds(quantRatio,distances,quantityThresh,\
        distMax,searchQuantity,otherQuantities,sortedCandidates,nQuantLen,\
        indSort,mode="fractional"):
    if mode == "fractional":
        if not np.isscalar(searchQuantity):
            candRadii = otherQuantities[sortedCandidates,0]
            # Geometric mean of radii, to ensure symmetry.
            geometricRadii = np.sqrt(searchQuantity[0]*candRadii)
            condition = (quantRatio[:,0] >= quantityThresh[0]) & \
                (distances[indSort] <= geometricRadii*distMax)
            for l in range(1,nQuantLen):
                condition = condition & \
                    (quantRatio[:,l] >= quantityThresh[l])
            matchingVoids = np.where(condition)[0]
        else:
            candRadii = otherQuantities[sortedCandidates]
            # Geometric mean of radii, to ensure symmetry.
            geometricRadii = np.sqrt(searchQuantity*candRadii)
            matchingVoids = np.where((quantRatio >= quantityThresh) & \
                (distances[indSort] <= geometricRadii*distMax))[0]
    else:
        if not np.isscalar(searchQuantity):
            condition = np.ones(quantRatio.shape[0],dtype=bool)
            for l in range(0,nQuantLen):
                condition = condition & \
                    (quantRatio[:,l] >= quantityThresh[l])
            matchingVoids = np.where(condition)[0]
        else:
            matchingVoids = np.where((quantRatio >= quantityThresh))[0]
    return matchingVoids

# Function to process candidates for a match to a given void. We compute their
# radius ratio and distance ratio, to check whether they are within the
# thresholds required:
def findAndProcessCandidates(centre,otherCentres,searchQuantity,\
        otherQuantities,boxsize,searchRadii,sortQuantity,candidates=None,\
        sortMethod='distance',overlapForVoid=None,quantityThresh=0.5,\
        distMax = 20.0,mode="fractional",treeOther=None):
    # Number of search quantities (radius or mass) to process for this void:
    if np.isscalar(searchQuantity):
        nQuantLen = 1
    else:
        nQuantLen = len(searchQuantity)
    # If we don't have candidates already, then we should find them:
    if candidates is None:
        if treeOther is None:
            treeOther = scipy.spatial.cKDTree(\
                snapedit.wrap(otherCentres,boxsize),boxsize=boxsize)
        candidates = treeOther.query_ball_point(snapedit.wrap(centre,boxsize),\
            searchRadii,workers=-1)
    # Check we have a sensible overlap map:
    if (overlapForVoid is None) and (sortMethod == "volumes"):
        raise Exception("overlap map required for volume sort method.")
    if len(candidates) > 0:
        # Sort indices:
        distances = np.sqrt(np.sum((\
                otherCentres[candidates,:] - centre)**2,1))
        # Unsorted radius (or mass) ratios:
        quantRatio = getUnsortedRatios(candidates,searchQuantity,\
            otherQuantities)
        [quantRatio,sortedCandidates,indSort] = getSortedQuantRatio(\
            sortMethod,candidates,quantRatio,distances,
            otherQuantities,searchQuantity,overlapForVoid,sortQuantity)
        # Get voids above the specified thresholds for these candidates:
        matchingVoids = getVoidsAboveThresholds(\
            quantRatio,distances,quantityThresh,\
            distMax,searchQuantity,otherQuantities,sortedCandidates,nQuantLen,\
            indSort,mode=mode)
        selectCandidates = np.array(candidates)[matchingVoids]
        selectedQuantRatios = quantRatio[matchingVoids]
        selectedDistances = distances[matchingVoids]
        if len(matchingVoids) > 0:
            # Add the most probable - remembering the +1 offset for 
            # pynbody halo catalogue IDs:
            selectedMatches = sortedCandidates[matchingVoids[0]] + 1
        else:
            selectedMatches = -1
    else:
        selectedMatches = -1
        selectCandidates = np.array([])
        selectedQuantRatios = []
        selectedDistances = []
    return [selectedMatches,selectCandidates,selectedQuantRatios,\
        selectedDistances]

# Get the candidates in all other catalogues:
def getCandidatesForVoidInAllCatalogues(centre,radius,centresList,\
        quantitiesList,boxsize,sortQuantity,sortMethod,
        quantityThresh,distMax,mode,overlapForVoid=None,treeList=None):
    newCatalogueRow = []
    numCats = len(centresList)
    if treeList is None:
        treeList = [scipy.spatial.cKDTree(\
                snapedit.wrap(centres,boxsize),boxsize=boxsize) \
                for centres in centresList]
    for ns in range(0,numCats):
        searchRadii = getSearchRadii(radius,quantitiesList[ns],\
            quantityThresh,distMax,mode=mode)
        [selectedMatches,selectCandidates,selectedQuantRatios,\
            selectedDistances] = findAndProcessCandidates(\
                centre,centresList[ns],radius,\
                quantitiesList[ns],boxsize,searchRadii,sortQuantity,\
                candidates=None,sortMethod=sortMethod,\
                overlapForVoid=overlapForVoid,\
                quantityThresh=quantityThresh,distMax = distMax,mode=mode,\
                treeOther=treeList[ns])
        newCatalogueRow.append(selectedMatches)
    return np.array(newCatalogueRow)

def getSearchRadii(quantity1,quantity2,quantityThresh,distMax,\
        mode="fractional"):
    if mode == "fractional":
        radii1 = quantity1/quantityThresh
        radii2 = quantity2/quantityThresh
        if np.isscalar(radii1):
            searchRadii = radii1
        elif len(radii1.shape) > 1:
            searchRadii = radii1[:,0]
        else:
            searchRadii = radii1
    else:
        searchRadii = distMax
    return searchRadii

def getAllCandidatesFromTrees(centres1,quantity1,quantity2,quantityThresh,\
        distMax,tree1,tree2,boxsize,mode = "fractional"):
    searchRadii = getSearchRadii(quantity1,quantity2,quantityThresh,distMax,\
        mode = mode)
    if mode == "fractional":
        # Interpret distMax as a fraction of the void radius, not the 
        # distance in Mpc/h.
        # Choose a search radius that is no greater than the void radius divided
        # by the radius ratio. If the other anti-halo is further away than this
        # then it wouldn't match to us anyway, so we don't need to consider it.
        searchOther = tree2.query_ball_point(snapedit.wrap(centres1,boxsize),\
            searchRadii,workers=-1)
    else:
        searchOther = tree1.query_ball_tree(tree2,distMax)
    return [searchRadii,searchOther]

# Function to match all voids:
def getMatchDistance(snap1,snap2,centres1,centres2,\
        quantity1,quantity2,tree1=None,tree2=None,distMax = 20.0,\
        max_index=200,quantityThresh=0.5,sortMethod='distance',\
        mode="fractional",sortQuantity = 0,cat1=None,cat2=None,volumes1=None,\
        volumes2=None,overlap = None):
    boxsize = snap1.properties['boxsize'].ratio("Mpc a h**-1")
    if tree1 is None:
        tree1 = scipy.spatial.cKDTree(snapedit.wrap(centres1,boxsize),\
            boxsize=boxsize)
    if tree2 is None:
        tree2 = scipy.spatial.cKDTree(snapedit.wrap(centres2,boxsize),\
            boxsize=boxsize)
    # Our procedure here is to get the closest anti-halo that lies within the 
    # threshold:
    match = [-2] # Always include -2, for compatibility with pynbody output
    candidatesList = []
    ratioList = []
    distList = []
    # Fina candidates for all anti-halos:
    [searchRadii,searchOther] = getAllCandidatesFromTrees(centres1,\
        quantity1,quantity2,quantityThresh,distMax,tree1,tree2,boxsize,
        mode = mode)
    # Build an overlap map, if we are using this method:
    if overlap is None and sortMethod == "volumes":
        if cat1 is None or cat2 is None or \
        volumes1 is None or volumes2 is None:
            raise Exception("Anti-halo catalogue required for " + \
                "volumes based matching.")
        overlap = overlapMap(cat1,cat2,volumes1,volumes2)
    # Process all the candidates, to find which are above the specified 
    # thresholds:
    for k in range(0,np.min([len(centres1),max_index])):
        candidates = searchOther[k]
        centre = centres1[k]
        if overlap is None:
            overlapForVoid = None
        else:
            overlapForVoid = overlap[k]
        [selectedMatches,selectCandidates,selectedQuantRatios,\
            selectedDistances] = findAndProcessCandidates(\
            centre,centres2,quantity1[k],\
            quantity2,boxsize,searchRadii,sortQuantity,candidates=candidates,\
            sortMethod=sortMethod,overlapForVoid=overlapForVoid,\
            quantityThresh=quantityThresh,distMax = distMax,mode=mode,\
            treeOther=tree2)
        candidatesList.append(selectCandidates)
        ratioList.append(selectedQuantRatios)
        distList.append(selectedDistances)
        match.append(selectedMatches)
    return [np.array(match,dtype=int),candidatesList,ratioList,distList]

# Check if an antihalo centre lies in the ZoA:
def isInZoA(centre,inUnits="equatorial",galacticCentreZOA = [-30,30],\
        bRangeCentre = [-10,10],bRange = [-5,5]):
    if len(centre.shape) < 2:
        R = np.sqrt(np.sum(centre**2))
        ra = np.arctan2(centre[1],centre[0])
        dec = np.arcsin(centre[2]/R)
    else:
        R = np.sqrt(np.sum(centre**2,1))
        ra = np.arctan2(centre[:,1],centre[:,0])
        dec = np.arcsin(centre[:,2]/R)
    coords = astropy.coordinates.SkyCoord(ra=ra*astropy.units.rad,
        dec=dec*astropy.units.rad)
    l = coords.galactic.l.value
    b = coords.galactic.b.value
    if len(centre.shape) < 2:
        if (l > galacticCentreZOA[0]) and (l < galacticCentreZOA[1]):
            result = (b > bRangeCentre[0]) and (b < bRangeCentre[1])
        else:
            result = (b > bRange[0]) and (b < bRange[1])
    else:
        result = np.zeros(len(centre),dtype=bool)
        result[(l > galacticCentreZOA[0]) & (l < galacticCentreZOA[1]) \
            & (b > bRangeCentre[0]) & (b < bRangeCentre[1])] = True
        result[((l < galacticCentreZOA[0]) | (l > galacticCentreZOA[1])) \
            & ((b > bRange[0]) & (b < bRange[1]))] = True
    return result

def overlapMap(cat1,cat2,volumes1,volumes2,checkFirst = False,verbose=False):
    overlap = np.zeros((len(cat1),len(cat2)))
    vol1 = np.array([np.sum(volumes1[halo['iord']]) for halo in cat1])
    vol2 = np.array([np.sum(volumes2[halo['iord']]) for halo in cat2])
    if checkFirst:
        for k in range(0,len(cat1)):
            for l in range(0,len(cat2)):
                if checkOverlap(cat1[k+1]['iord'],cat2[l+1]['iord']):
                    intersection = np.intersect1d(cat1[k+1]['iord'],\
                        cat2[l+1]['iord'])
                    overlap[k,l] = np.sum(\
                        volumes1[intersection])/np.sqrt(vol1[k]*vol2[l])
    else:
        for k in range(0,len(cat1)):
            for l in range(0,len(cat2)):
                intersection = np.intersect1d(cat1[k+1]['iord'],\
                    cat2[l+1]['iord'])
                if len(intersection) > 0:
                    overlap[k,l] = np.sum(\
                        volumes1[intersection])/np.sqrt(vol1[k]*vol2[l])
            if verbose:
                print(("%.3g" % (100*(k*len(cat1) + l + 1)/\
                    (len(cat1)*len(cat2)))) + "% complete")
    return overlap

# Check whether two halos have any overlap:
def checkOverlap(list1,list2):
    [min1,max1] = tools.minmax(list1)
    [min2,max2] = tools.minmax(list2)
    if max1 < min2:
        return False
    elif min1 > max2:
        return False
    else:
        if len(list1) < len(list2):
            listMin = list1
            listMax = list2
        else:
            listMin = list2
            listMax = list1
        for k in range(0,len(listMin)):
            if np.isin(listMin[k],listMax):
                return True
        return False

def linearFromIJ(i,j,N):
    if i >= N or i < 0:
        raise Exception("i out of range.")
    if j >= N or j < 0:
        raise Exception("j out of range.")
    if i < j:
        return int(i*N - i*(i+1)/2 + j - i - 1)
    if j < i:
        return int(j*N - j*(j+1)/2 + i - j - 1)
    else:
        raise Exception("i = j not valid.")


# Select the voids which will be included in catalogue matching, applying
# radius and mass cuts, and other arbitrary cuts (such as signel-to-noise):
def computeShortCentresList(snapNumList,antihaloCentres,antihaloRadii,\
        antihaloMasses,rSphere,rMin,rMax,massRange=None,additionalFilters=None,\
        sortBy = "mass",max_index=None):
    # Build a filter from the specified radius range:
    filterCond = [(antihaloRadii[k] > rMin) & (antihaloRadii[k] <= rMax) \
            for k in range(0,len(snapNumList))]
    # Further filter on mass, if mass limits are specified:
    if massRange is not None:
        if len(massRange) < 2:
            raise Exception("Mass range must have an upper and a lower " + \
                "limit.")
        for k in range(0,len(snapNumList)):
            filterCond[k] = filterCond[k] & \
                (antihaloMasses[k] > massRange[0]) & \
                (antihaloMasses[k] <= massRange[1])
    # Apply any additional filters (such as signal-to-noise):
    if additionalFilters is not None:
        for k in range(0,len(snapNumList)):
            filterCond[k] = filterCond[k] & additionalFilters[k]
    # Select anti-halos within rSphere of the centre of the box, applying
    # this filter:
    centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],\
        rSphere,filterCondition = filterCond[k]) \
        for k in range(0,len(snapNumList))]
    # Sort the list on either mass or radius:
    if sortBy == "mass":
        centralAntihaloMasses = [\
            antihaloMasses[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
        sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
            for k in range(0,len(snapNumList))]
    elif sortBy == "radius":
        centralAntihaloRadii = [\
            antihaloRadii[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
        sortedList = [np.flip(np.argsort(centralAntihaloRadii[k])) \
            for k in range(0,len(snapNumList))]
    else:
        raise Exception("sortBy argument not recognised.")
    # Include the option to impose an artificial cut on the length of the
    # void list in each catalogue:
    ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
    if max_index is None:
        max_index = np.max(ahCounts)
    # Construct the list of centres for all voids:
    centresListShort = [np.array([antihaloCentres[l][\
        centralAntihalos[l][0][sortedList[l][k]],:] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]
    return [centresListShort,centralAntihalos,sortedList,ahCounts,max_index]


# Take a shortened list of anti-halos in each sample, and compute lists of 
# properties for each such as mass or radius:
def getShortenedQuantity(quantity,centralAntihalos,shortenedList,sortedList,\
        ahCounts,max_index):
    return [np.array([quantity[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(shortenedList))]

# Get the two way matches of a void in other catalogues:
def getTwoWayMatches(nVoid,nCat,otherColumns,numCats,\
        oneWayMatchesAllCatalogues,alreadyMatched,oneWayMatchesOther=None,\
        enforceExclusive=False):
    oneWayMatches = oneWayMatchesAllCatalogues[nCat]
    if oneWayMatchesOther is None:
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
    twoWayMatch = np.zeros(oneWayMatchesOther[nVoid].shape,dtype=bool)
    for m in range(0,numCats-1):
        if oneWayMatches[nVoid,otherColumns[m]] < 0:
            # Fails if we don't match to anything
            twoWayMatch[m] = False
        else:
            # 2-way only if the other matches back to this:
            twoWayMatch[m] = (\
                oneWayMatchesAllCatalogues[otherColumns[m]][\
                oneWayMatches[nVoid,otherColumns[m]] - 1,nCat] == nVoid+1)
            if enforceExclusive:
                twoWayMatch[m] = twoWayMatch[m] and \
                    (not alreadyMatched[m,oneWayMatches[nVoid][m]-1])
    return twoWayMatch

# Check which of a void's matches are new:
def getNewMatches(nVoid,nCat,oneWayMatches,alreadyMatched,blockDuplicates=True):
    isNewMatch = np.zeros(\
        oneWayMatches[nVoid].shape,dtype=bool)
    for m in range(0,len(isNewMatch)):
        if (oneWayMatches[nVoid][m] > 0) and (m != nCat):
            if blockDuplicates:
                isNewMatch[m] = \
                not alreadyMatched[m,oneWayMatches[nVoid][m]-1]
            else:
                isNewMatch[m] = True
        if (oneWayMatches[nVoid][m] < 0):
            isNewMatch[m] = False
    return isNewMatch

# Check whether we should include a void or not:
def checkIfVoidIsNeeded(nVoid,nCat,alreadyMatched,twoWayMatch,otherColumns,
        candidateCounts,oneWayMatches,twoWayOnly=True,blockDuplicates=True):
    voidAlreadyFound = alreadyMatched[nCat,nVoid]
    atLeastOneTwoWayMatch = np.any(twoWayMatch)
    atLeastOneMatchWithUniqueCandidate = np.any(\
        candidateCounts[nCat][otherColumns,nVoid] == 1)
    haveNewMatch = getNewMatches(nVoid,nCat,oneWayMatches,alreadyMatched,\
        blockDuplicates=True)
    atLeastOneNewMatch = np.any(haveNewMatch[otherColumns])
    if twoWayOnly:
        needed = (not voidAlreadyFound) and atLeastOneTwoWayMatch and \
            atLeastOneMatchWithUniqueCandidate and atLeastOneNewMatch
    else:
        needed = (not voidAlreadyFound) and atLeastOneMatchWithUniqueCandidate \
            and atLeastOneNewMatch
    return needed

def getUniqueEntriesInCatalogueRow(catalogueRow,alreadyMatched):
    numCats = len(catalogueRow)
    isNeededList = np.zeros(numCats,dtype=bool)
    for ns in range(0,numCats):
        if catalogueRow[ns] > -1:
            # Missing voids automatically fail. For others, they pass
            # only if they haven't been found:
            isNeededList[ns] = (not alreadyMatched[ns,catalogueRow[ns]-1])
    return isNeededList

def gatherCandidatesRatiosAndDistances(numCats,nCat,nVoid,allCandidates,\
        allRatios,allDistances):
    candm = []
    ratiosm = []
    distancesm = []
    for m in range(0,numCats):
        if (m != nCat):
            candm.append(allCandidates[nCat][m][nVoid])
            ratiosm.append(allRatios[nCat][m][nVoid])
            distancesm.append(allDistances[nCat][m][nVoid])
    return [candm,ratiosm,distancesm]

# Mark the two-way matches of a void as already found, so that we don't 
# accidentally include them:
def markCompanionsAsFound(nVoid,nCat,numCats,voidMatches,\
        oneWayMatchesAllCatalogues,allCandidates,alreadyMatched):
    for m in range(0,numCats):
        if (m != nCat) and (voidMatches[m] > 0):
            # Only deem something to be already matched
            # if it maps back to this with a single unique 
            # candidate
            alreadyMatched[m][voidMatches[m] - 1] = \
                (oneWayMatchesAllCatalogues[m]\
                [voidMatches[m] - 1,nCat] == nVoid+1) \
                and (len(allCandidates[m][nCat]\
                [voidMatches[m] - 1]) == 1)
        if m == nCat:
            alreadyMatched[m][nVoid] = True

# Get all the possible matches of a given void by following chains of 
# two way matches:
def followAllMatchChains(nVoid,nCat,numCats,oneWayMatches,alreadyMatched,\
        diffMap,allCandidates):
    # Track down all possible matches that are connected
    # to this one.
    # First, get an initial scan of possible candidates:
    allCands = [[] for m in range(0,numCats)] # Candidates that could
        # be connected to this void
    lengthsList = np.zeros(numCats,dtype=int) # Number of candidates in each
        # catalogue
    for m in range(0,numCats):
        twoWayCand = oneWayMatches[nVoid][m]
        haveMatch = twoWayCand > -1
        alreadyIncluded = alreadyMatched[m][twoWayCand - 1]
        if haveMatch and (not alreadyIncluded):
            allCands[m].append(twoWayCand-1)
    # Number of candidates we have in each catalogue:
    lengthsListNew = np.array(\
        [len(cand) for cand in allCands],dtype=int)
    # Keep iterating until we stop finding new matches:
    while not np.all(lengthsListNew == lengthsList):
        lengthsList = lengthsListNew
        # Loop over all catalogues:
        for n in range(0,numCats):
            if len(allCands[n]) > 0:
                # If we have candidates, follow their 
                # 2-way matches to the other catalogues.
                # Loop over all other catalogues, d, and 
                # the candidate anti-halos in catalogue n:
                for d in diffMap[n]:
                    for m in range(0,len(allCands[n])):
                        # For each antihalo in catalogue n, 
                        # get the candidates in catalogue d
                        # to which it has two-way matches:
                        otherCatCandidates = \
                            allCandidates[n][d][\
                            allCands[n][m]]
                        if len(otherCatCandidates) > 0:
                            # The first candidate is the
                            # two way match:
                            bestCand = otherCatCandidates[0]
                            # Add this iff we haven't
                            # already found it, and it
                            # hasn't already been marked
                            # as belonging to another
                            # void:
                            inOtherList = np.isin(\
                                    bestCand,allCands[d])
                            alreadyIncluded = \
                                alreadyMatched[d,bestCand]
                            if (not inOtherList) and \
                                    (not alreadyIncluded):
                                allCands[d].append(bestCand)
        lengthsListNew = np.array([len(cand) \
            for cand in allCands],dtype=int)
    return allCands

# Count the number of two way matches in a list with candidates for each
# mcmc sample, when using N-way matching
def getNumberOfTwoWayMatchesNway(numCats,allCands,allCandidates,diffMap):
    twoWayMatchesAllCands = [[] for m in range(0,numCats)]
    for m in range(0,numCats):
        for n in range(0,len(allCands[m])):
            nTW = np.sum(np.array(\
                [len(allCandidates[m][d][allCands[m][n]]) \
                for d in diffMap[m]]) > 0)
            twoWayMatchesAllCands[m].append(nTW)
    return twoWayMatchesAllCands


# Count the number of two way matches.
# TODO - can we merge this with getNumberOfTwoWayMatchesNway? They are 
# similar, but doing slightly different things:
def getTotalNumberOfTwoWayMatches(numCats,diffMap,\
        allCandidates,voidMatches):
    twoWayMatchCounts = 0
    for m in range(0,numCats):
        for d in diffMap[m]:
            allCands = allCandidates[m][d][\
                voidMatches[m]-1]
            if len(allCands) > 0:
                if allCands[0] == voidMatches[d]-1:
                    twoWayMatchCounts += 1
    return twoWayMatchCounts


# Compute a quantity (such as radius ratio or distance ratio) which is defined
# for all candidates connected to a particular void:
def computeQuantityForCandidates(quantity,numCats,allCands,diffMap):
    quantityAverages = [[] for k in range(0,numCats)]
    for m in range(0,numCats):
        for n in range(0,len(allCands[m])):
            individualQuantities = np.zeros(numCats-1)
            for d in range(0,numCats-1):
                if len(quantity[m][diffMap[m][d]][\
                        allCands[m][n]]) > 0:
                    individualQuantities[d] = quantity[m][\
                        diffMap[m][d]][allCands[m][n]][0]
            qR = np.mean(individualQuantities)
            quantityAverages[m].append(qR)
    return quantityAverages

def applyNWayMatching(nVoid,nCat,numCats,oneWayMatches,alreadyMatched,\
        diffMap,allCandidates,allRatios,allDistances):
    # Follow all chains of two way matches to get possible void candidates:
    allCands = followAllMatchChains(nVoid,nCat,numCats,oneWayMatches,\
        alreadyMatched,diffMap,allCandidates)
    # Count two way matches:
    twoWayMatchesAllCands = getNumberOfTwoWayMatchesNway(numCats,allCands,\
        allCandidates,diffMap)
    # Compute the average radius ratio:
    ratioAverages = computeQuantityForCandidates(allRatios,numCats,allCands,\
        diffMap)
    # Compute the distances:
    distAverages = computeQuantityForCandidates(allDistances,numCats,allCands,\
        diffMap)
    # Now figure out the best candidates to include:
    bestCandidates = -np.ones(numCats,dtype=int)
    bestRatios = np.zeros(numCats)
    bestDistances = np.zeros(numCats)
    numberOfLinks = 0
    for m in range(0,numCats):
        if len(allCands[m]) == 1:
            bestCandidates[m] = allCands[m][0] + 1
            bestRatios[m] = ratioAverages[m][0]
            bestDistances[m] = distAverages[m][0]
            numberOfLinks += twoWayMatchesAllCands[m][0]
        elif len(allCands[m]) > 1:
            maxTW = np.max(allCands[m])
            haveMaxTW = np.where(\
                np.array(allCands[m]) == maxTW)[0]
            if len(haveMaxTW) > 1:
                # Need to use the ratio criteria to choose
                # instead if we have a tie:
                maxRat = np.max(ratioAverages[m])
                haveMaxRat = np.where(\
                    np.array(ratioAverages[m]) == maxRat)[0]
                bestCandidates[m] = allCands[m][\
                    haveMaxRat[0]] + 1
                bestRatios[m] = ratioAverages[m][\
                    haveMaxRat[0]]
                bestDistances[m] = distAverages[m][\
                    haveMaxRat[0]]
                numberOfLinks += twoWayMatchesAllCands[m][\
                    haveMaxRat[0]]
            else:
                # Otherwise, we just use the one with the most number
                # of two way matches:
                bestCandidates[m] = allCands[m][\
                    haveMaxTW[0]] + 1
                bestRatios[m] = ratioAverages[m][\
                    haveMaxTW[0]]
                bestDistances[m] = distAverages[m][\
                    haveMaxTW[0]]
                numberOfLinks += twoWayMatchesAllCands[m][\
                    haveMaxTW[0]]
        # If no candidates, just leave it as -1
    # Now we mark the other voids as already included. Remember that we
    # only include the best matches as already included, freeing the
    # other candidates to possibly be included elsewhere:
    for m in range(0,numCats):
        alreadyMatched[m,bestCandidates[m] - 1] = True
    return [bestCandidates,bestRatios,bestDistances,numberOfLinks]

def getAllQuantityiesFromVoidMatches(voidMatches,numCats,quantitiesList):
    quantitiesArray = []
    for l in range(0,numCats):
        if voidMatches[l] > -1:
            quantitiesArray.append(quantitiesList[l][voidMatches[l] - 1])
    return quantitiesArray

def getStdCentreFromVoidMatches(voidMatches,numCats,centresList):
    centresArray = getAllQuantityiesFromVoidMatches(voidMatches,numCats,\
        centresList)
    stdCentres = np.std(centresArray,0)
    return stdCentres

def getMeanCentreFromVoidMatches(voidMatches,numCats,centresList):
    centresArray = getAllQuantityiesFromVoidMatches(voidMatches,numCats,\
        centresList)
    meanCentres = np.mean(centresArray,0)
    return meanCentres

def getMeanRadiusFromVoidMatches(voidMatches,numCats,radiusList):
    radiusArray = getAllQuantityiesFromVoidMatches(voidMatches,numCats,\
        radiusList)
    meanRadius = np.mean(radiusArray)
    return meanRadius

# Code to iterate on the centres of a given void, so that we are less
# dependent on matching to a particular void:
def refineVoidCentres(voidMatches,ratiosm,distancesm,numCats,centresList,\
        radiusList,boxsize,sortQuantity,sortMethod,quantityThresh,\
        distMax,mode,overlapForVoid=None,treeList = None,iterMax = 100):
    voidMatchesLast = np.array([-1 for k in range(0,numCats)])
    voidMatchesNew = voidMatches
    iterations = 0
    success = True
    while not np.all(voidMatchesLast == voidMatchesNew):
        voidMatchesLast = voidMatchesNew
        # First, compute the mean centre of the voids in this set:
        meanCentres = getMeanCentreFromVoidMatches(\
            voidMatchesNew,numCats,centresList)
        meanRadius = getMeanRadiusFromVoidMatches(\
            voidMatchesNew,numCats,radiusList)
        # Get all the voids within the thresholds from this centre:
        voidMatchesNew = getCandidatesForVoidInAllCatalogues(
            meanCentres,meanRadius,centresList,radiusList,boxsize,\
            sortQuantity,sortMethod,quantityThresh,distMax,mode,\
            overlapForVoid=overlapForVoid,treeList=treeList)
        # Check that we didn't run completely out of voids, as this will
        # make our centre meaningless.
        if np.all(voidMatchesNew < 0):
            break
        iterations += 1
        if iterations > iterMax:
            success = False
            break
    return [voidMatchesNew,ratiosm,distancesm,success]

# Add an entry to the catalogue:
def matchVoidToOtherCatalogues(nVoid,nCat,numCats,otherColumns,\
        oneWayMatchesOther,oneWayMatchesAllCatalogues,twoWayMatch,\
        allCandidates,alreadyMatched,candidateCounts,NWayMatch,\
        allRatios,allDistances,diffMap,finalCandidates,\
        finalCat,finalRatios,finalDistances,finalCombinatoricFrac,\
        finalCatFrac,refineCentres,centresList,radiusList,\
        boxsize,sortQuantity,sortMethod,quantityThresh,distMax,\
        mode,treeList=None):
    oneWayMatches = oneWayMatchesAllCatalogues[nCat]
    # Mark companions of this void as already found, to avoid duplication.
    # Additionally, store the candidates (candm), radius ratios (ratiosm) and 
    # distances to candidates (distancesm) of this void for output data:
    [candm,ratiosm,distancesm] = gatherCandidatesRatiosAndDistances(\
        numCats,nCat,nVoid,allCandidates,allRatios,allDistances)
    finalCandidates.append(candm)
    if NWayMatch:
        # Get the best candidates using the N-way matching code:
        [bestCandidates,bestRatios,bestDistances,numberOfLinks] = \
            applyNWayMatching(nVoid,nCat,numCats,oneWayMatches,alreadyMatched,\
                diffMap,allCandidates,allRatios,allDistances)
        finalCat.append(bestCandidates)
        finalRatios.append(bestRatios)
        finalDistances.append(bestDistances)
        finalCombinatoricFrac.append(float(numberOfLinks/\
            (numCats*(numCats-1))))
        finalCatFrac.append(float(\
            np.sum(bestCandidates > 0)/numCats))
    else:
        if refineCentres:
            [voidMatches,ratiosm,distancesm,success] = refineVoidCentres(\
                oneWayMatches[nVoid],ratiosm,distancesm,numCats,centresList,\
                radiusList,boxsize,sortQuantity,sortMethod,\
                quantityThresh,distMax,mode,overlapForVoid=None,\
                treeList = treeList,iterMax = 100)
            # Check the new entry is still unique:
            if success:
                success = np.any(getUniqueEntriesInCatalogueRow(\
                    voidMatches,alreadyMatched)) # Skip the void if it's just 
                    # a duplicate of something that already existed, or
                    # a subset.
        else:
            voidMatches = oneWayMatches[nVoid]
            success = True
        if not success:
            print("WARNING: void centre refining did not converge.")
            # Do nothing else - don't add a failed void to the catalogue!
        else:
            # Block the voids we have identified from appearing again:
            markCompanionsAsFound(nVoid,nCat,numCats,voidMatches,\
                oneWayMatchesAllCatalogues,allCandidates,alreadyMatched)
            # Provided we found at least two voids, then add it to the catalogue:
            if np.sum(voidMatches > 0) > 1:
                finalCat.append(voidMatches)
                finalRatios.append(ratiosm)
                finalDistances.append(distancesm)
                finalCatFrac.append(float(len(np.where(voidMatches > 0)[0])\
                    /numCats))
                # Compute the combinatoric fraction:
                twoWayMatchCounts = getTotalNumberOfTwoWayMatches(numCats,\
                    diffMap,allCandidates,voidMatches)
                finalCombinatoricFrac.append(twoWayMatchCounts/\
                    (numCats*(numCats-1)))

# Load simulations and catalogue data so that we can combine them. If these
# are already loaded, this function won't reload them.
def loadCatalogueData(snapList,snapListRev,ahProps,sortMethod,snapSortList,\
        hrList,verbose=False):
    # If snaplists are strings, then load them:
    if type(snapList[0]) == str:
        snapList = [pynbody.load(snap) for snap in snapList]
    if type(snapListRev[0]) == str:
        snapListRev = [pynbody.load(snap) for snap in snapListRev]
    # Load centres so that we can filter to the constrained region:
    boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
    if verbose:
        print("Extracting anti-halo properties...")
    if ahProps is None:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    antihaloRadii = [props[7] for props in ahProps]
    if sortMethod == "volumes":
        if snapSortList is None:
            snapSortList = [np.argsort(snap['iord']) for snap in snapList]
        volumesList = [ahProps[k][4][snapSortList[k]] \
            for k in range(0,len(ahProps))]
    else:
        volumesList = [None for k in range(0,len(ahProps))]
    # Load anti-halo catalogues:
    if hrList is None:
        hrList = [snap.halos() for snap in snapListRev]
    return [snapList,snapListRev,boxsize,ahProps,antihaloCentres,\
        antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList]

# For some methods, we need to create shortened anti-halo catalogues first
# otherwise we end up matching a lot of useless halos and wasting time:
def constructShortenedCatalogues(numCats,matchType,sortMethod,hrList,\
        centralAntihalos,sortedList):
    if matchType == "pynbody" or sortMethod == "volumes":
        # These methods need copies of the catalogues, which we will then
        # filter below:
        hrListCentral = [copy.deepcopy(halos) for halos in hrList]
    else:
        # Other methods don't need these catalogues, so we store place-holders:
        hrListCentral = [None for halos in hrList]
    # List of the halo numbers (in pynbody convention) for all anti-halos
    # being used for matching:
    for l in range(0,numCats):
        if matchType == "pynbody" or sortMethod == "volumes":
            # Manually edit the pynbody anti-halo catalogues to
            # only include the relevant anti-halos:
            hrListCentral[l]._halos = dict([(k+1,\
                hrList[l][centralAntihalos[l][0][sortedList[l][k]]+1]) \
                for k in range(0,len(centralAntihalos[l][0]))])
            hrListCentral[l]._nhalos = len(centralAntihalos[l][0])
    return hrListCentral

# If we are using volume overlaps to match voids, then we need to 
# create an overlapMap between all pairs of anti-halos in all catalogues:
def getOverlapList(numCats,hrListCentral,volumesList):
    overlapList = []
    for k in range(0,numCats):
        for l in range(0,numCats):
            if k >= l:
                continue
            overlapList.append(overlapMap(hrListCentral[k],\
                hrListCentral[l],volumesList[k],volumesList[l],\
                verbose=False))
    return overlapList

# Perform matching between two catalogues:
def getMatchCandidatesTwoCatalogues(n1,n2,matchType,snapListRev,hrListCentral,\
        centresListShort,quantityList,max_index,thresh,crossMatchThreshold,\
        quantityListRad,quantityListMass,crossMatchQuantity,treeList,distMax,\
        sortMethod,mode,volumesList):
    if matchType == 'pynbody':
        # Use pynbody's halo catalogue matching to identify likely
        # matches:
        [match, candidatesList] = getMatchPynbody(snapListRev[n1],\
                snapListRev[n2],hrListCentral[n1],hrListCentral[n2],\
                quantityList[n1],quantityList[n2],\
                max_index=max_index,threshold=thresh,\
                quantityThresh = crossMatchThreshold)
        ratioList = None
        distList = None
    elif matchType == 'distance':
        # This is the conventional approach: matching on distance
        # radius criteria:
        if crossMatchQuantity == "both":
            # More complicated version where we require both mass
            # and radius to match the given thresholds:
            if sortMethod == "volumes":
                linearIndex = linearFromIJ(n1,n2,numCats)
                if n1 < l:
                    overlap = overlapList[linearIndex]
                else:
                    overlap = overlapList[linearIndex].transpose()
            else:
                overlap = None
            [match, candidatesList,ratioList,distList] = \
                getMatchDistance(snapListRev[n1],\
                    snapListRev[n2],centresListShort[n1],\
                    centresListShort[n2],\
                    np.array([quantityListRad[n1],\
                        quantityListMass[n1]]).transpose(),\
                    np.array([quantityListRad[n2],\
                        quantityListMass[n2]]).transpose(),\
                    tree1=treeList[n1],\
                    tree2=treeList[n2],distMax = distMax,\
                    max_index=max_index,\
                    quantityThresh=crossMatchThreshold,\
                    sortMethod=sortMethod,mode=mode,\
                    cat1 = hrListCentral[n1],\
                    cat2 = hrListCentral[n2],\
                    volumes1 = volumesList[n1],\
                    volumes2 = volumesList[n2],\
                    overlap = overlap)
        else:
            # Simple version, where we match on either mass or 
            # radius;
            [match, candidatesList,ratioList,distList] = \
                getMatchDistance(snapListRev[n1],\
                    snapListRev[n2],centresListShort[n1],\
                    centresListShort[n2],quantityList[n1],\
                    quantityList[n2],tree1=treeList[n1],\
                    tree2=treeList[n2],distMax = distMax,\
                    max_index=max_index,\
                    quantityThresh=crossMatchThreshold,\
                    sortMethod=sortMethod,mode=mode)
    else:
        raise Exception("Unrecognised matching type requested.")
    return [match, candidatesList,ratioList,distList]

def getOneWayMatchesAllCatalogues(numCats,matchType,snapListRev,\
        hrListCentral,centresListShort,quantityList,max_index,thresh,\
        crossMatchThreshold,ahCounts,quantityListRad,quantityListMass,\
        crossMatchQuantity,treeList,distMax,sortMethod,mode,volumesList):
    matchArrayList = [[] for k in range(0,numCats)] # List of best 
        # matches in each one way pair
    allCandidates = [] # List of candidates for each one way pair
    allRatios = [] # List of the radius (or mass) ratios of each candidate
    allDistances = [] # List of distances from a void to all it's candidates
    for k in range(0,numCats):
        matchArrayListNew = matchArrayList[k]
        allCandidatesNew = []
        allRatiosNew = []
        allDistancesNew = []
        for l in range(0,numCats):
            if l != k:
                [match, candidatesList,ratioList,distList] = \
                    getMatchCandidatesTwoCatalogues(k,l,matchType,snapListRev,\
                        hrListCentral,centresListShort,quantityList,max_index,\
                        thresh,crossMatchThreshold,quantityListRad,\
                        quantityListMass,crossMatchQuantity,treeList,distMax,\
                        sortMethod,mode,volumesList)
                matchArrayListNew.append(match)
            elif l == k:
                # Placeholder entries when matching a catalogue to itself:
                matchArrayListNew.append([-2] + \
                    list(np.arange(1,np.min([ahCounts[l],max_index])+1)))
                candidatesList = np.array([[m] \
                    for m in np.arange(1,np.min([ahCounts[l],max_index])+1)])
                ratioList = np.ones(len(candidatesList))
                distList = np.zeros(len(candidatesList))
            allCandidatesNew.append(candidatesList)
            allRatiosNew.append(ratioList)
            allDistancesNew.append(distList)
        matchArrayList[k] = list(matchArrayListNew)
        allCandidates.append(allCandidatesNew)
        allRatios.append(allRatiosNew)
        allDistances.append(allDistancesNew)
    # Here we arrange the one-way matches of every void in every 
    # catalogue. This is a list of Ncat matrices, one for each catalogue.
    # Each matrix has dimensions Nvoids_i x Ncat (Nvoids_i being the number of
    # voids in catalogue i):
    oneWayMatchesAllCatalogues = \
        [np.array(matchArrayList[k]).transpose()[1:,:] \
        for k in range(0,numCats)]
    return [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
        allRatios,allDistances]

def constructSnapNameList(samplesFolder,snapNumList,snapname):
    return [samplesFolder + "sample" + str(snapNum) + "/" + \
            snapname for snapNum in snapNumList]

# Construct an anti-halo catalogue from reversed snapshots
def constructAntihaloCatalogue(snapNumList,samplesFolder="new_chain/",\
        verbose=True,rSphere=135,max_index=None,thresh=0.5,\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        fileSuffix= '',matchType='distance',crossMatchQuantity='radius',\
        crossMatchThreshold = 0.5,distMax=20.0,sortMethod='ratio',\
        blockDuplicates=True,twoWayOnly = True,\
        snapList=None,snapListRev=None,ahProps=None,hrList=None,\
        rMin = 5,rMax = 30,mode="fractional",massRange = None,\
        snapSortList = None,overlapList = None,NWayMatch = False,\
        additionalFilters = None,sortBy="mass",refineCentres=False,\
        sortQuantity = 0,enforceExclusive=False):
    # Load snapshots:
    if snapList is None:
        snapList = constructSnapNameList(samplesFolder,snapNumList,snapname)
    if snapListRev is None:
        snapListRev = constructSnapNameList(samplesFolder,snapNumList,\
            snapnameRev)
    [snapList,snapListRev,boxsize,ahProps,antihaloCentres,\
        antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList] = \
        loadCatalogueData(snapList,snapListRev,ahProps,sortMethod,\
            snapSortList,hrList,verbose=verbose)
    numCats = len(snapNumList)
    if verbose:
        print("Loading snapshots...")
    # Construct filtered anti-halo lists:
    [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
        computeShortCentresList(snapNumList,antihaloCentres,\
            antihaloRadii,antihaloMasses,rSphere,rMin,rMax,massRange=massRange,\
            additionalFilters=additionalFilters,sortBy = sortBy,\
            max_index=max_index)
    # Construct new antihalo catalogues from the filtered list:
    if verbose:
        print("Constructing constrained region catalogues...")
    # For some methods, we need to create shortened anti-halo catalogues first
    # otherwise we end up matching a lot of useless halos and wasting time:
    hrListCentral = constructShortenedCatalogues(numCats,matchType,sortMethod,\
        hrList,centralAntihalos,sortedList)
    # List of filtered void antih-halo numbers (pynbody offset):
    shortHaloList = [np.array(centralAntihalos[l][0])[sortedList[l]] + 1 \
        for l in range(0,numCats)]
    # If we are using volume overlaps to match voids, then we need to 
    # create an overlapMap between all pairs of anti-halos in all catalogues:
    if sortMethod == "volumes":
        if overLapList is None:
            overlapList = getOverlapList(numCats,hrListCentral,volumesList)
        else:
            # Check that the supplied overlap list is the correct size. If not,
            # this probably means that a bad overlap list was given:
            if len(overlapList) != int(numCats*(numCats - 1)/2):
                raise Exception("Invalid overlapList!")
    # Construct matches:
    # Create lists of the quantity to match voids with (mass or radius), chosen
    # to match centresListShort:
    quantityListRad = getShortenedQuantity(antihaloRadii,centralAntihalos,\
            centresListShort,sortedList,ahCounts,max_index)
    quantityListMass = getShortenedQuantity(antihaloMasses,\
        centralAntihalos,centresListShort,sortedList,ahCounts,max_index)
    if crossMatchQuantity == 'radius':
        quantityList = quantityListRad
    elif crossMatchQuantity == 'mass':
        quantityList = quantityListMass
    elif crossMatchQuantity == 'both':
        quantityList = quantityList
    else:
        raise Exception('Unrecognised cross-match quantity.')
    # Build a KD tree with the centres in centresListShort for efficient 
    # matching:
    treeList = [scipy.spatial.cKDTree(\
        snapedit.wrap(centres,boxsize),boxsize=boxsize) \
        for centres in centresListShort]
    if verbose:
        print("Computing matches...")
    # Main loop to compute candidate matches:
    [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
        allRatios,allDistances] = getOneWayMatchesAllCatalogues(numCats,\
            matchType,snapListRev,hrListCentral,centresListShort,quantityList,\
            max_index,thresh,crossMatchThreshold,ahCounts,quantityListRad,\
            quantityListMass,crossMatchQuantity,treeList,distMax,\
            sortMethod,mode,volumesList)
    # Combined to a single catalogue:
    if verbose:
        print("Combining to a single catalogue...")
    # Lists storing various properties of the final catalogue. Unfortunately, 
    # we can't pre-allocate these as arrays because we don't know the size of
    # the final catalogue before we do the matching, so these are 
    # continuously updated lists:
    twoWayMatchLists = [[] for k in range(0,numCats)] # Stores a list
        # of which matches are two-way matches
    finalCat = [] # Will contain the final catalogue, a list of voids
        # for which we have candidate anti-halos in each mcmc sample
    finalCandidates = [] # Stores the candidate voids in each mcmc sample for
        # each row in the final cadalogue (not just the best)
    finalRatios = [] # Stores the radius (mass) ratios of all the pairs 
        # in the final cataloge
    finalDistances = [] # Stores the distances for all pairs in the final
        # catalogue
    finalCombinatoricFrac = [] # Stores the combinatoric fraction for each void
        # in the final catalogue
    finalCatFrac = [] # Stores the catalogue fraction for each void in the final
        # catalogue
    candidateCounts = [np.zeros((numCats,ahCounts[l]),dtype=int) \
        for l in range(0,numCats)] # Number of candidates
        # that each void could match to.
    # To avoid adding duplicates, we need to remember which voids we have
    # already added to the catalogue somehow. This is achieved using a 
    # boolean array - every time we find a void, we flag it here so that
    # we can later check if it has already been added:
    alreadyMatched = np.zeros((numCats,max_index),dtype=bool)
    # List of the other catalogues, for each catalogue:
    diffMap = [np.setdiff1d(np.arange(0,numCats),[k]) \
        for k in range(0,numCats)]
    # Loop over all catalogues:
    for k in range(0,numCats):
        # Matches for catalogue k to all other catalogues:
        oneWayMatches = oneWayMatchesAllCatalogues[k]
        # Columns corresponding to the other catalogues:
        otherColumns = diffMap[k]
        # One way matches to other catalogues only:
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        # Loop over all voids in this catalogue:
        for l in range(0,np.min([ahCounts[k],max_index])):
            twoWayMatch = getTwoWayMatches(l,k,otherColumns,numCats,\
                oneWayMatchesAllCatalogues,alreadyMatched,\
                oneWayMatchesOther=oneWayMatchesOther,\
                enforceExclusive=enforceExclusive)
            twoWayMatchLists[k].append(twoWayMatch)
            # Skip if the void has already beeen included, or just
            # doesn't have any two way matches:
            for m in range(0,numCats):
                candidateCounts[k][m,l] = len(allCandidates[k][m][l])
            if not checkIfVoidIsNeeded(l,k,alreadyMatched,twoWayMatch,\
                    otherColumns,candidateCounts,oneWayMatches,\
                    twoWayOnly=twoWayOnly,blockDuplicates=blockDuplicates):
                continue
            matchVoidToOtherCatalogues(l,k,numCats,otherColumns,\
                oneWayMatchesOther,oneWayMatchesAllCatalogues,twoWayMatch,\
                allCandidates,alreadyMatched,candidateCounts,NWayMatch,\
                allRatios,allDistances,diffMap,finalCandidates,\
                finalCat,finalRatios,finalDistances,finalCombinatoricFrac,\
                finalCatFrac,refineCentres,centresListShort,quantityListRad,\
                boxsize,sortQuantity,sortMethod,crossMatchThreshold,distMax,\
                mode,treeList = treeList)
    return [np.array(finalCat),shortHaloList,twoWayMatchLists,\
        finalCandidates,finalRatios,finalDistances,allCandidates,\
        candidateCounts,allRatios,np.array(finalCombinatoricFrac),\
        np.array(finalCatFrac),alreadyMatched]


# Do we even use this function any more?
def getMatchRatios(matchList,quantityList):
    ratioArray = np.zeros(matchList.shape)
    for k in range(0,len(matchList)):
        values = np.zeros(len(matchList[k]))
        for l in range(0,len(matchList[k])):
            if matchList[k,l] < 0.0:
                values[l] = -1
            else:
                values[l] = quantityList[:,l][matchList[k,l]-1]
        maxVal = np.max(values)
        haveValue = np.where(values > 0)
        ratioArray[k][haveValue] = values[haveValue]/maxVal
    return ratioArray



def getPoissonSamples(lam,nSamples,seed = None):
    if seed is not None:
        np.random.seed(seed)
    if np.isscalar(lam):
        samples = np.random.poisson(lam,nSamples)
    else:
        samples = np.zeros((len(lam),nSamples),dtype=int)
        for k in range(0,len(lam)):
            samples[k,:] = np.random.poisson(lam[k],nSamples)
    return samples

def estimatePoissonRatioErrorbarMonteCarlo(n1,n2,errorVal = 0.67,seed = None,\
        nSamples=1000,returnSamples=False):
    interval1 = scipy.stats.poisson(n1).interval(errorVal)
    interval2 = scipy.stats.poisson(n2).interval(errorVal)
    np.random.seed(seed)
    # Generate random samples
    n1Samples = getPoissonSamples(n1,nSamples)
    n2Samples = getPoissonSamples(n2,nSamples)
    nonZero = np.where(n2Samples != 0)
    ratioValues = np.zeros(n1Samples.shape)
    ratioValues[nonZero] = n1Samples[nonZero]/n2Samples[nonZero]
    if np.isscalar(n1):
        lower = np.percentile(ratioValues,100*(0.5 - errorVal/2))
        upper = np.percentile(ratioValues,100*(0.5 + errorVal/2))
    else:
        lower = np.zeros(n1.shape)
        upper = np.zeros(n1.shape)
        for k in range(0,len(n1)):
            lower[k] = np.percentile(ratioValues[k,:],0.5 - errorVal/2)
            upper[k] = np.percentile(ratioValues[k,:],100*(0.5 + errorVal/2))
    returnList = [n1/n2,lower,upper]
    if returnSamples:
        returnList.append(ratioValues)
    return returnList


def getMeanCentresFromCombinedCatalogue(combinedCat,centresList,\
        returnError=False,boxsize=None):
    meanCentresArray = np.zeros((len(combinedCat),3))
    numCats = combinedCat.shape[1]
    if returnError:
        stdCentresArray = np.zeros((len(combinedCat),3))
    for nV in range(0,len(combinedCat)):
        meanCentresArray[nV,:] = getMeanCentreFromVoidMatches(\
            combinedCat[nV],numCats,centresList)
        if returnError:
            stdCentresArray[nV,:] = getStdCentreFromVoidMatches(\
                combinedCat[nV],numCats,centresList)
    if returnError:
        return [meanCentresArray,stdCentresArray]
    else:
        return meanCentresArray


def getMeanCentreDistance(combinedCat,centresList,returnError=False,\
        boxsize=None):
    meanDistanceArray = np.zeros(len(combinedCat))
    if returnError:
        stdDistanceArray = np.zeros(len(combinedCat))
    for nV in range(0,len(combinedCat)):
        centresArray = []
        for l in range(0,combinedCat.shape[1]):
            if combinedCat[nV][l] > -1:
                centresArray.append(centresList[l][combinedCat[nV,l] - 1])
        meanCentre = context.computePeriodicCentreWeighted(\
            np.array(centresArray),periodicity=boxsize)
        distances = np.array([tools.getPeriodicDistance(\
            meanCentre,centre,boxsize=boxsize) \
            for centre in centresArray])
        meanDistanceArray[nV] = np.mean(distances)
        if returnError:
            stdDistanceArray[nV,:] = np.std(distances)/\
                np.sqrt(len(centresArray))
    if returnError:
        return [meanDistanceArray,stdDistanceArray]
    else:
        return meanDistanceArray

# Search radii around centre that have some threshold density relative
# to the mean density.
def getThresholdRadius(centres,snap,thresh=0.2,rSearchMax = 100,\
        rSearchMin=5.0):
    tree = tools.getKDTree(snap)
    radii = np.array(len(centres))
    Om = snap.properties['omegaM0']
    N = np.cbrt(len(snap))
    rhoM = 2.7754e11*Om
    mUnit = rhoM*(boxsize/N)**3
    for k in range(0,len(centres)):
        func = lambda r: mUnit*tree.query_ball_point(centres[k,:],r,\
            return_length=True,workers=-1)/(4*np.pi*rhoM*r**3/3) - thresh
        valMin = func(rSearchMin)
        valMax = func(rSearchMax)
        if valMax*valMin > 0:
            radii[k] = -1
        else:
            radii[k] = scipy.optimize.brentq(func,rSearchMin,rSearchMax)[0]
    return radii




def getDistanceMatrix(ahIndices,centresList,lowerTriangleOnly=False):
    centres = [centresList[k][ahIndices[k]-1] \
        for k in range(0,len(ahIndices))]
    matrix = np.zeros((len(ahIndices),len(ahIndices)))
    for k in range(0,len(ahIndices)):
        for l in range(0,len(ahIndices)):
            if l > k and lowerTriangleOnly:
                continue
            if l == k:
                continue
            if ahIndices[k] > 0 and ahIndices[l] > 0:
                matrix[k,l] = np.sqrt(np.sum((centres[k] - centres[l])**2))
            else:
                matrix[k,l] = np.nan
    return matrix

def getBestCandidateMeanDistance(catalogue,centresList):
    meanDistances = np.zeros(len(catalogue))
    if len(catalogue) == 0:
        return np.array([])
    lowerTriInd = np.tril_indices(len(catalogue[0]),k=-1)
    for k in range(0,len(catalogue)):
        matrix = getDistanceMatrix(catalogue[k],centresList,\
            lowerTriangleOnly=True)
        meanDistances[k] = np.nanmean(matrix[lowerTriInd])
    return meanDistances

# Ratio of two radii, smallest/biggest
def getRadiusRatio(radii1,radii2):
    if np.isscalar(radii1):
        if np.isscalar(radii2):
            if radii2 > radii1:
                return radii1/radii2
            else:
                return radii2/radii1
        else:
            returnArr = radii2/radii1
            bigger = np.where(radii2 > radii1)[0]
            returnArr[bigger] = radii1/radii2[bigger]
            return returnArr
    else:
        if np.isscalar(radii2):
            returnArr = radii1/radii2
            smaller = np.where(radii2 < radii1)[0]
            returnArr[smaller] = radii2/radii1[smaller]
            return returnArr
        else:
            returnArr = radii2/radii1
            bigger = np.where(radii2 > radii1)[0]
            returnArr[bigger] = radii1[bigger]/radii2[bigger]
            return returnArr

def getRatioMatrix(ahIndices,radiiList,lowerTriangleOnly = False):
    radii = [radiiList[:,k][ahIndices[k]-1] \
        for k in range(0,len(ahIndices))]
    matrix = np.ones((len(ahIndices),len(ahIndices)))
    for k in range(0,len(ahIndices)):
        for l in range(0,len(ahIndices)):
            if l > k and lowerTriangleOnly:
                continue
            if l == k:
                continue
            if ahIndices[k] > 0 and ahIndices[l] > 0:
                matrix[k,l] = getRadiusRatio(radii[k],radii[l])
            else:
                matrix[k,l] = np.nan
    return matrix

def getBestCandidateMeanRatio(catalogue,radiiList):
    meanRatios = np.zeros(len(catalogue))
    if len(catalogue) == 0:
        return np.array([])
    lowerTriInd = np.tril_indices(len(catalogue[0]),k=-1)
    for k in range(0,len(catalogue)):
        matrix = getRatioMatrix(catalogue[k],radiiList,lowerTriangleOnly=True)
        # Exclude the diagonals:
        meanRatios[k] = np.nanmean(matrix[lowerTriInd])
    return meanRatios

# Get statistics on performance for different parameters:
def specialNanMin(x):
    if len(x) == 0:
        return np.nan
    else:
        return np.nanmin(x)

def specialNanMax(x):
    if len(x) == 0:
        return np.nan
    else:
        return np.nanmax(x)


def getPropertyFromCat(catList,propertyList):
    propertyListOut = -np.ones(catList.shape,dtype=float)
    for k in range(0,len(catList)):
        for l in range(0,len(catList[0])):
            if catList[k,l] > 0:
                propertyListOut[k,l] = propertyList[l][catList[k,l]-1]
    return propertyListOut

def getCentresFromCat(catList,centresList,ns):
    centresListOut = np.zeros((len(catList),3),dtype=float)
    for k in range(0,len(catList)):
        if catList[k,ns] > 0:
            centresListOut[k,:] = centresList[ns][catList[k,ns]-1]
        else:
            centresListOut[k,:] = np.nan
    return centresListOut

def getMeanProperty(propertyList,stripNaN=True,lowerLimit=0,stdError=True):
    meanProperty = np.zeros(len(propertyList))
    sigmaProperty = np.zeros(len(propertyList))
    for k in range(0,len(propertyList)):
        condition = propertyList[k,:] > lowerLimit
        if stripNaN:
            condition = condition & (np.isfinite(propertyList[k,:] > 0))
        haveProperty = np.where(condition)[0]
        meanProperty[k] = np.mean(propertyList[k,haveProperty])
        sigmaProperty[k] = np.std(propertyList[k,haveProperty])
        if stdError:
            sigmaProperty[k] /= np.sqrt(len(haveProperty))
    return [meanProperty,sigmaProperty]

# Apply cuts to the catalogue based on combinatoric or catalogue fraction:
def applyCatalogueCuts(finalCatFracOpt,finalCombinatoricFracOpt,\
        percentilesCat,percentilesComb,scaleFilter,\
        snrList,snrThresh,catFracCut,combFracCut,snrCut):
    # Catalogue and combinatoric fraction filters:
    catFracFilter = [finalCatFracOpt > thresh for thresh in percentilesCat]
    combFracFilter = [finalCombinatoricFracOpt > thresh \
        for thresh in percentilesComb]
    nBins = len(scaleFilter)
    # Length of the pre-cut catalogue:
    nCatLength = len(finalCatFracOpt)
    # Combined filter:
    combinedFilter = np.zeros(nCatLength,dtype=bool)
    meanCatFrac = np.zeros(nBins)
    stdErrCatFrac = np.zeros(nBins)
    meanCombFrac = np.zeros(nBins)
    stdErrCombFrac = np.zeros(nBins)
    for k in range(0,nBins):
        individualFilter = scaleFilter[k] # In radius or mass bin k
        if snrCut:
            individualFilter = individualFilter & (snrList > snrThresh)
        # Include cuts on catalogue and combinatoric fractions:
        meanCatFrac[k] = np.mean(finalCatFracOpt[individualFilter])
        stdErrCatFrac[k] = np.std(finalCatFracOpt[individualFilter])/\
            np.sqrt(np.sum(individualFilter))
        meanCombFrac[k] = np.mean(finalCombinatoricFracOpt[individualFilter])
        stdErrCombFrac[k] = np.std(finalCombinatoricFracOpt[individualFilter])/\
            np.sqrt(np.sum(individualFilter))
        if catFracCut:
            individualFilter = individualFilter & catFracFilter[k]
        if combFracCut:
            individualFilter = individualFilter & combFracFilter[k]
        combinedFilter = combinedFilter | individualFilter
    return [combinedFilter, meanCatFrac, stdErrCatFrac, \
        meanCombFrac, stdErrCombFrac]

# Generate thresholds for the combined catalogue, using random catalogues:
def getThresholdsInBins(nBins,cutScale,massListMeanUn,radiiListMeanUn,\
        finalCombinatoricFracUn,finalCatFracUn,\
        rLower,rUpper,mLower,mUpper,percThresh,massBins=None,radBins=None):
    if massBins is None:
        massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),nBins+1))
    [inMassBins,noInMassBins] = plot.binValues(massListMeanUn,massBins)
    if radBins is None:
        radBins = np.linspace(rLower,rUpper,nBins+1)
    [inRadBins,noInRadBins] = plot.binValues(radiiListMeanUn,radBins)
    percentilesComb = []
    percentilesCat = []
    for k in range(0,nBins):
        if cutScale == "mass":
            selection = inMassBins[k]
        elif cutScale == "radius":
            selection = inRadBins[k]
        else:
            raise Exception("Unrecognised 'cutScale' value ")
        if len(selection) > 0:
            percentilesComb.append(np.percentile(\
                finalCombinatoricFracUn[selection],percThresh))
            percentilesCat.append(np.percentile(\
                finalCatFracUn[selection],percThresh))
        else:
            percentilesComb.append(0.0)
            percentilesCat.append(0.0)
    return [percentilesCat,percentilesComb]

def getFinalCatalogue(snapNumList,snapNumListUncon,snrThresh = 10,\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        samplesFolder="new_chain/",snapList = None,snapListRev = None,\
        snapListUnconstrained = None,snapListUnconstrainedRev=None,\
        mLower = "auto",mUpper = 2e15,nBinEdges = 8,muOpt = 0.9,rSearchOpt = 1,\
        rSphere = 300,rSphereInner = 135,NWayMatch = True,rMin=5,rMax=30,\
        mMin=1e11,mMax = 1e16,percThresh=99,chainFile="chain_properties.p",\
        Nden=256,recomputeUnconstrained = False,data_folder="./",\
        unconstrainedFolderNew = "new_chain/unconstrained_samples/",\
        recomputeData=True,verbose=True,catFracCut=True,combFracCut=False,\
        cutScale='mass',rLower=5,rUpper=20,snrCut=True,sortBy="radius"):
    # Load snapshots:
    if verbose:
        print("Catalogue construction. \nLoading snapshots..")
        sys.stdout.flush()
    if snapList is None:
        snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapname) for snapNum in snapNumList]
    if snapListRev is None:
        snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapnameRev) for snapNum in snapNumList]
    # Parameters:
    Om = snapList[0].properties['omegaM0']
    rhoc = 2.7754e11
    boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
    N = int(np.cbrt(len(snapList[0])))
    mUnit = 8*Om*rhoc*(boxsize/N)**3
    if mLower == "auto":
        mLower = 100*mUnit
    if mUpper == "auto":
        mUpper = 10*mLower
    # Load properties of the anti-halo catalogues:
    hrList = [snap.halos() for snap in snapListRev]
    ahProps = [pickle.load(\
                open(snap.filename + ".AHproperties.p","rb")) \
                for snap in snapList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahProps]
    ahCentresUnmapped = [props[5] for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    vorVols = [props[4] for props in ahProps]
    antihaloRadii = [props[7] for props in ahProps]
    # SNR data:
    if verbose:
        print("Loading chain data...")
        sys.stdout.flush()
    [mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
        std_field,hmc_Elh,hmc_Eprior,hades_accept_count,\
        hades_attempt_count] = pickle.load(open(chainFile,"rb"))
    snrField = mean_field**2/std_field**2
    snrFieldLin = np.reshape(snrField,Nden**3)
    # Centres about which to compute SNR:
    if verbose:
        print("Computing SNR...")
        sys.stdout.flush()
    grid = snapedit.gridListPermutation(Nden,perm=(2,1,0))
    centroids = grid*boxsize/Nden + boxsize/(2*Nden)
    positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
    tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
        boxsize=boxsize)
    # SNR for each catalogue:
    nearestPointsList = [tree.query_ball_point(\
        snapedit.wrap(antihaloCentres[k] + boxsize/2,boxsize),\
        antihaloRadii[k],workers=-1) \
        for k in range(0,len(antihaloCentres))]
    snrAllCatsList = [np.array([np.mean(snrFieldLin[points]) \
        for points in nearestPointsList[k]]) for k in range(0,len(snapNumList))]
    snrFilter = [snr > snrThresh for snr in snrAllCatsList]
    # Central anti-halos with appropriate filteR:
    if verbose:
        print("Filtering voids...")
        sys.stdout.flush()
    centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
            (antihaloMasses[k] <= mMax) & snrFilter[k]) \
            for k in range(0,len(snapNumList))]
    centralAntihaloMasses = [\
            antihaloMasses[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
    centralAntihaloRadii = [\
            antihaloRadii[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
    if sortBy == "mass":
        sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
                for k in range(0,len(snapNumList))]
    elif sortBy == "radius":
        sortedList = [np.flip(np.argsort(centralAntihaloRadii[k])) \
                for k in range(0,len(snapNumList))]
    else:
        raise Exception("sortBy parameter'" + str(sortBy) + "' invalid.")
    # Gather these into shortened property lists for the anti-halos
    # in the central region, including only those that match the filter:
    ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
    max_index = np.max(ahCounts)
    massListShort = [np.array([antihaloMasses[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
    centresListShort = [np.array([antihaloCentres[l][\
            centralAntihalos[l][0][sortedList[l][k]],:] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
    radiiListShort = [np.array([antihaloRadii[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
    # Maps from each catalogue to all the other catalogues:
    diffMap = [np.setdiff1d(np.arange(0,len(snapNumList)),[k]) \
        for k in range(0,len(snapNumList))]
    # Catalogue construction:
    if verbose:
        print("Constructing catalogue...")
        sys.stdout.flush()
    [finalCatOpt,shortHaloListOpt,twoWayMatchListOpt,finalCandidatesOpt,\
        finalRatiosOpt,finalDistancesOpt,allCandidatesOpt,candidateCountsOpt,\
        allRatiosOpt,finalCombinatoricFracOpt,finalCatFracOpt,\
        alreadyMatched] = \
        tools.loadOrRecompute(data_folder + "catalogue_all_data.p",\
        constructAntihaloCatalogue,snapNumList,snapList=snapList,\
        snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=rSphere,\
        massRange = [mMin,mMax],NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
        additionalFilters = snrFilter)
    # Mean radius, centre, and mass:
    if verbose:
        print("Computing catalogue properties...")
        sys.stdout.flush()
    radiiListOpt = getPropertyFromCat(finalCatOpt,radiiListShort)
    massListOpt = getPropertyFromCat(finalCatOpt,massListShort)
    [radiiMeanOpt, radiiSigmaOpt]  = getMeanProperty(radiiListOpt)
    [massMeanOpt, massSigmaOpt]  = getMeanProperty(massListOpt)
    finalCentresOptList = np.array([getCentresFromCat(\
        finalCatOpt,centresListShort,ns) for ns in range(0,len(snapNumList))])
    meanCentreOpt = np.nanmean(finalCentresOptList,0)
    [meanCentresArray,stdCentresArray] = getMeanCentresFromCombinedCatalogue(\
        finalCatOpt,centresListShort,returnError=True)
    # SNR about void centres:
    nearestPoints = tree.query_ball_point(\
        snapedit.wrap(meanCentreOpt + boxsize/2,boxsize),radiiMeanOpt,\
        workers=-1)
    snrList = np.array([np.mean(snrFieldLin[points]) \
        for points in nearestPoints])
    # Save catalogue details:
    np.savez(data_folder + "catalogue_data.npz",catalogue=finalCatOpt,\
        radii=radiiMeanOpt,sigma_radii=radiiSigmaOpt,mass=massMeanOpt,\
        sigma_mass=massSigmaOpt,centres=meanCentresArray,\
        sigma_centres=stdCentresArray,snr=snrList,\
        cat_frac=finalCatFracOpt,comb_frac=finalCombinatoricFracOpt)
    # Generate an unconstrained catalogue to determine threshold for
    # spurious voids:
    # Unconstrained catalogue:
    if verbose:
        print("Getting random catalogue...")
        sys.stdout.flush()
    unconstrainedCatFile = data_folder + "unconstrained_catalogue.p"
    if (not os.path.isfile(unconstrainedCatFile)) or recomputeUnconstrained:
        snapListUn = [pynbody.load("new_chain/unconstrained_samples/sample" + \
            str(num) + "/gadget_full_forward_512/snapshot_001") \
            for num in snapNumListUncon]
        snapListRevUn = [pynbody.load(\
            "new_chain/unconstrained_samples/sample" + \
            str(num) + "/gadget_full_reverse_512/snapshot_001") \
            for num in snapNumListUncon]
        hrListUn = [snap.halos() for snap in snapListRevUn]
        ahPropsUn = [pickle.load(\
                    open(snap.filename + ".AHproperties.p","rb")) \
                    for snap in snapListUn]
        [finalCatUn,shortHaloListUn,twoWayMatchListUn,finalCandidatesOUn,\
            finalRatiosUn,finalDistancesUn,allCandidatesUn,candidateCountsUn,\
            allRatiosUn,finalCombinatoricFracUn,finalCatFracUn,\
            alreadyMatched] = \
            constructAntihaloCatalogue(snapNumListUncon,snapList=snapListUn,\
            snapListRev=snapListRevUn,ahProps=ahPropsUn,hrList=hrListUn,\
            max_index=None,twoWayOnly=True,blockDuplicates=True,\
            crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=rSphere,\
            massRange = [mMin,mMax],NWayMatch = NWayMatch,rMin=rMin,rMax=rMax)
        antihaloCentresUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in ahPropsUn]
        antihaloMassesUn = [props[3] for props in ahPropsUn]
        antihaloRadiiUn = [props[7] for props in ahPropsUn]
        centralAntihalosUn = [tools.getAntiHalosInSphere(antihaloCentresUn[k],\
            rSphere,filterCondition = (antihaloRadiiUn[k] > rMin) & \
            (antihaloRadiiUn[k] <= rMax) & (antihaloMassesUn[k] > mMin) & \
            (antihaloMassesUn[k] <= mMax)) \
            for k in range(0,len(snapNumListUncon))]
        centralAntihaloMassesUn = [\
            antihaloMassesUn[k][centralAntihalosUn[k][0]] \
            for k in range(0,len(centralAntihalosUn))]
        centralAntihaloRadiiUn = [\
            antihaloRadiiUn[k][centralAntihalosUn[k][0]] \
            for k in range(0,len(centralAntihalosUn))]
        if sortBy == "mass":
            sortedListUn = [np.flip(np.argsort(centralAntihaloMassesUn[k])) \
                    for k in range(0,len(snapNumListUncon))]
        elif sortBy == "radius":
            sortedListUn = [np.flip(np.argsort(centralAntihaloRadiiUn[k])) \
                    for k in range(0,len(snapNumListUncon))]
        else:
            raise Exception("sortBy parameter'" + str(sortBy) + "' invalid.")
        ahCountsUn = np.array([len(cahs[0]) for cahs in centralAntihalosUn])
        max_indexUn = np.max(ahCountsUn)
        radiiListShortUn = [np.array([antihaloRadiiUn[l][\
            centralAntihalosUn[l][0][sortedListUn[l][k]]] \
            for k in range(0,np.min([ahCountsUn[l],max_indexUn]))]) \
            for l in range(0,len(snapNumListUncon))]
        massListShortUn = [np.array([antihaloMassesUn[l][\
                centralAntihalosUn[l][0][sortedListUn[l][k]]] \
                for k in range(0,np.min([ahCountsUn[l],max_indexUn]))]) \
                for l in range(0,len(snapNumListUncon))]
        pickle.dump([finalCatUn,shortHaloListUn,twoWayMatchListUn,\
            finalCandidatesOUn,\
            finalRatiosUn,finalDistancesUn,allCandidatesUn,candidateCountsUn,\
            allRatiosUn,finalCombinatoricFracUn,finalCatFracUn,\
            antihaloCentresUn,antihaloMassesUn,antihaloRadiiUn,\
            centralAntihalosUn,centralAntihaloMassesUn,sortedListUn,\
            ahCountsUn,radiiListShortUn,massListShortUn],\
            open(unconstrainedCatFile,"wb"))
    else:
        [finalCatUn,shortHaloListUn,twoWayMatchListUn,\
            finalCandidatesOUn,\
            finalRatiosUn,finalDistancesUn,allCandidatesUn,candidateCountsUn,\
            allRatiosUn,finalCombinatoricFracUn,finalCatFracUn,\
            antihaloCentresUn,antihaloMassesUn,antihaloRadiiUn,\
            centralAntihalosUn,centralAntihaloMassesUn,sortedListUn,\
            ahCountsUn,radiiListShortUn,massListShortUn] = pickle.load(open(\
                unconstrainedCatFile,"rb"))
    # Unconstrained radii and masses:
    radiiListCombUn = getPropertyFromCat(finalCatUn,radiiListShortUn)
    massListCombUn = getPropertyFromCat(finalCatUn,massListShortUn)
    [radiiListMeanUn,radiiListSigmaUn] = getMeanProperty(radiiListCombUn)
    [massListMeanUn,massListSigmaUn] = getMeanProperty(massListCombUn)
    # Determine percentiles from the unconstrained catalogues:
    massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),nBinEdges))
    [inMassBins,noInMassBins] = plot.binValues(massListMeanUn,massBins)
    radBins = np.linspace(rLower,rUpper,nBinEdges)
    [inRadBins,noInRadBins] = plot.binValues(radiiListMeanUn,radBins)
    [percentilesCat, percentilesComb] = getThresholdsInBins(\
        nBinEdges-1,cutScale,massListMeanUn,radiiListMeanUn,\
        finalCombinatoricFracUn,finalCatFracUn,\
        rLower,rUpper,mLower,mUpper,percThresh,massBins=massBins,\
        radBins=radBins)
    # Construct the filter by mass bin or radius bin:
    if cutScale == "mass":
        scaleBins = massBins
        if verbose:
            print("Filtering voids by mass bin...")
            sys.stdout.flush()
        scaleFilter = [(massMeanOpt > massBins[k]) & \
            (massMeanOpt <= massBins[k+1]) \
            for k in range(0,len(massBins) - 1)]
    if cutScale == "radius":
        scaleBins = radBins
        if verbose:
            print("Filtering voids by radius bin...")
            sys.stdout.flush()
        scaleFilter = [(radiiMeanOpt > radBins[k]) & \
            (radiiMeanOpt <= radBins[k+1]) \
            for k in range(0,len(radBins) - 1)]
    # Apply catalogue/combinatoric/snr cuts to construct the filtered final
    # catalouge:
    [combinedFilter, meanCatFrac, stdErrCatFrac, \
        meanCombFrac, stdErrCombFrac] = applyCatalogueCuts(finalCatFracOpt,\
        finalCombinatoricFracOpt,percentilesCat,percentilesComb,scaleFilter,\
        snrList,snrThresh,catFracCut,combFracCut,snrCut)
    # Save data on the scale cut for future use:
    tools.savePickle([scaleBins,percentilesCat,percentilesComb,\
        meanCatFrac,stdErrCatFrac,meanCombFrac,stdErrCombFrac,\
        radiiMeanOpt,massMeanOpt,massSigmaOpt,radiiSigmaOpt,\
        massBins,radBins,scaleFilter],\
        data_folder + "catalogue_scale_cut_data.p")
    # Cut on distance:
    distanceArray = np.sqrt(np.sum(meanCentresArray**2,1))
    combinedFilter135 = combinedFilter & \
        (distanceArray < rSphereInner)
    # Conditions to supply to the void profile code:
    filteredCatalogues = [\
        np.array(centralAntihalos[k][0])[sortedList[k][finalCatOpt[\
        (finalCatOpt[:,k] >= 0) & combinedFilter135,k] - 1]] \
        for k in range(0,len(snapList))]
    additionalConditions = [np.isin(np.arange(0,len(antihaloMasses[k])),\
        filteredCatalogues[k]) for k in range(0,len(snapList))]
    tools.savePickle(filteredCatalogues,data_folder + "filtered_catalogue.p")
    # Data for unconstrained sims:
    if verbose:
        print("Getting unconstrained regions with similar density...")
        sys.stdout.flush()
    snapListUnconstrained = [pynbody.load(unconstrainedFolderNew + "sample" \
            + str(snapNum) + "/" + snapname) for snapNum in snapNumListUncon]
    snapListUnconstrainedRev = [pynbody.load(unconstrainedFolderNew + \
            "sample" + str(snapNum) + "/" + snapnameRev) \
            for snapNum in snapNumListUncon]
    ahPropsUnconstrained = [tools.loadPickle(snap.filename + ".AHproperties.p")\
            for snap in snapListUnconstrained]
    ahCentresList = [props[5] for props in ahProps]
    ahCentresListUn = [props[5] for props in ahPropsUnconstrained]
    antihaloRadiiUn = [props[7] for props in ahPropsUnconstrained]
    rEffMin = 0.0
    rEffMax = 10.0
    nRadiusBins = 101
    rBinStack = np.linspace(rEffMin,rEffMax,nRadiusBins)
    vorVols = [props[4] for props in ahProps]
    vorVolsUn = [props[4] for props in ahPropsUnconstrained]
    pairCountsList = []
    volumesList = []
    pairCountsListUn = []
    volumesListUn = []
    # Centres of regions with similar density contrast to the BORG region:
    [centreListUn,densitiesInCentres,denListUn] = tools.loadOrRecompute(\
        data_folder + "centre_list_unconstrained_data.p",\
        getCentreListUnconstrained,\
        snapListUnconstrained,
        randomSeed = 1000,numDenSamples = 1000,rSphere = 135,\
        densityRange = [-0.051,-0.049],_recomputeData=recomputeData)
    gc.collect()
    if verbose:
        print("Computing combined catalogue profiles...")
        sys.stdout.flush()
    [rBinStackCentresCombined,nbarjSepStackCombined,\
            sigmaSepStackCombined,nbarjSepStackUnCombined,\
            sigmaSepStackUnCombined,\
            nbarjAllStackedCombined,sigmaAllStackedCombined,\
            nbarjAllStackedUnCombined,sigmaAllStackedUnCombined,\
            nbar,rMin2,mMin2,mMax2] = tools.loadOrRecompute(\
                data_folder + "void_profiles_data_combined.p",\
                getVoidProfilesData,\
                snapNumList,snapNumListUncon,\
                snapList = snapList,snapListRev = snapListRev,\
                samplesFolder="new_chain/",\
                unconstrainedFolder="new_chain/unconstrained_samples/",\
                snapname = "gadget_full_forward_512/snapshot_001",\
                snapnameRev = "gadget_full_reverse_512/snapshot_001",\
                reCentreSnaps = False,N=512,boxsize=677.7,mMin = mLower,\
                mMax = mUpper,rMin=5,rMax=30,verbose=True,combineSims=False,\
                method="poisson",errorType = "Weighted",\
                unconstrainedCentreList = centreListUn,\
                additionalConditions = additionalConditions,\
                redoPairCounts=True,rEffMax=10.0,rEffMin=0.0,nBins=nRadiusBins,\
                pairCountsListUn=None,\
                volumesListUn=None,pairCountsList=None,\
                volumesList=None,\
                ahPropsConstrained = ahProps,\
                ahPropsUnconstrained = ahPropsUnconstrained,\
                snapListUnconstrained=snapListUnconstrained,\
                snapListUnconstrainedRev=snapListUnconstrainedRev,\
                _recomputeData=recomputeData,data_folder=data_folder)
    gc.collect()
    if verbose:
        print("Computing all voids profiles...")
    [rBinStackCentres,nbarjSepStack,\
            sigmaSepStack,nbarjSepStackUn,sigmaSepStackUn,\
            nbarjAllStacked,sigmaAllStacked,nbarjAllStackedUn,\
            sigmaAllStackedUn,\
            nbar,rMin2,mMin2,mMax2] = tools.loadOrRecompute(\
                data_folder + "void_profiles_data_all.p",\
                getVoidProfilesData,\
                snapNumList,snapNumListUncon,\
                snapList = snapList,snapListRev = snapListRev,\
                samplesFolder="new_chain/",\
                unconstrainedFolder="new_chain/unconstrained_samples/",\
                snapname = "gadget_full_forward_512/snapshot_001",\
                snapnameRev = "gadget_full_reverse_512/snapshot_001",\
                reCentreSnaps = False,N=512,boxsize=677.7,mMin = mLower,\
                mMax = mUpper,rMin=5,rMax=30,verbose=True,combineSims=False,\
                method="poisson",errorType = "Weighted",\
                unconstrainedCentreList = centreListUn,\
                additionalConditions = None,\
                redoPairCounts=True,rEffMax=10.0,rEffMin=0.0,nBins=nRadiusBins,\
                pairCountsListUn=None,\
                volumesListUn=None,pairCountsList=None,\
                volumesList=None,\
                ahPropsConstrained = ahProps,\
                ahPropsUnconstrained = ahPropsUnconstrained,\
                snapListUnconstrained=snapListUnconstrained,\
                snapListUnconstrainedRev=snapListUnconstrainedRev,\
                _recomputeData=recomputeData,data_folder=data_folder)
    gc.collect()
    return [massMeanOpt,combinedFilter135,combinedFilter,rBinStackCentresCombined,\
    nbarjSepStackCombined,sigmaSepStackCombined,\
    nbarjAllStackedUnCombined,sigmaAllStackedUnCombined,nbar,rMin2,\
    mMin2,mMax2,nbarjSepStackUn,sigmaSepStackUn,\
    rBinStackCentres,nbarjSepStack,\
    sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,\
    nbarjSepStackUn,sigmaSepStackUn]


# Estimate the masses of clusters in the vicinity of the supplies points,
# clusterLoc, using halos nearby.
def getClusterMassEstimatesFromSnapshot(clusterLoc,snap,equatorialXYZ2MPP,\
        reductions=4,iterations=20,gatherRadius=5,neighbourRadius=10,\
        ahProps=None,recomputeData=True,recentre=True,hncentres=None,\
        hnmasses=None):
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if recentre:
        # Make sure that we re-centre the simulation about the local
        # centre of mass, to account for drift between BORG realisations:
        recentredLocs = snapedit.unwrap(simulation_tools.getClusterCentres(\
                    clusterLoc,snap=snap,snapPath = snap.filename,\
                    fileSuffix = "clusters1",\
                    recompute=recomputeData,method="snapshot",\
                    reductions=reductions,\
                    iterations=iterations) + boxsize/2,boxsize)
    else:
        recentredLocs = clusterLoc
    # Load halo properties:
    if ahProps is None and ((hncentres is None) or (hnmasses is None)):
        ahProps = tools.loadPickle(snap.filename + ".AHproperties.p")
    # Adjust positions to ICRS co-ordinates (simulation co-ordinates
    # are different):
    if hncentres is None:
        hncentres = -snapedit.unwrap((ahProps[0] - boxsize/2),boxsize/2)
    if hnmasses is None:
        hnmasses = ahProps[1] # Halo masses
    # Find the counterpart halos corresponding to the supplied clusters:
    [counterpartClusters,counterpartHalos] = \
        simulation_tools.matchClustersAndHalos(recentredLocs,hncentres,\
        hnmasses,boxsize,equatorialXYZ2MPP,\
        gatherRadius=gatherRadius,neighbourRadius=neighbourRadius)
    if np.all(counterpartHalos >= 0):
        clusterMasses = hnmasses[counterpartHalos]
        clusterCentres = hncentres[counterpartHalos,:]
    else:
        # Handle the case when we couldn't find a candidate:
        clusterMasses = -np.ones(len(clusterLoc))
        clusterCentres = -np.ones(clusterLoc.shape)
        for k in range(0,len(clusterLoc)):
            if counterpartHalos[k] >= 0:
                clusterMasses[k] = hnmasses[counterpartHalos[k]]
                clusterCentres[k,:] = hncentres[counterpartHalos[k],:]
    return [clusterMasses,clusterCentres,counterpartHalos]

# Helper function to compute mass estimates from all samples:
def getBORGClusterMassEstimates(snapList,clusterLoc,equatorialXYZ2MPP,\
        reductions=4,iterations=20,gatherRadius=5,neighbourRadius=10,\
        allAhProps = None,recomputeData=True,recentre=True):
    clusterMasses = -np.ones((len(snapList),len(clusterLoc)))
    clusterCentres = -np.ones((len(snapList),len(clusterLoc),3))
    clusterCounterparts = -np.ones((len(snapList),len(clusterLoc)),dtype=int)
    for ns in range(0,len(snapList)):
        snap = tools.getPynbodySnap(snapList[ns])
        gc.collect()
        # Transform the snapshot appropriately:
        tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
        # Use existing ahProps list if available. Otherwise we will
        # just load it manually:
        if allAhProps is None:
            ahProps = None
        else:
            ahProps = allAhProps[ns]
        # Computation of masses for a single snapshot:
        [masses,centres,counterparts] = getClusterMassEstimatesFromSnapshot(\
            clusterLoc,snap,equatorialXYZ2MPP,\
            reductions=reductions,iterations=iterations,\
            gatherRadius=gatherRadius,neighbourRadius=neighbourRadius,\
            ahProps=ahProps,recomputeData=recomputeData,recentre=recentre)
        clusterMasses[ns,:] = masses
        clusterCentres[ns,:,:] = centres
        clusterCounterparts[ns,:] = counterparts
    # Compute means, accounting for any missing data:
    meanMasses = -np.ones(len(clusterLoc))
    meanCentres = -np.ones(clusterLoc.shape)
    sigmaMasses = -np.ones(len(clusterLoc))
    sigmaCentres = -np.ones(clusterLoc.shape)
    for nc in range(0,len(clusterLoc)):
        haveData = np.where(clusterMasses[:,nc] >= 0.0)[0]
        meanMasses[nc] = np.mean(clusterMasses[haveData,nc])
        sigmaMasses[nc] = np.std(clusterMasses[haveData,nc])/\
            np.sqrt(len(haveData))
        meanCentres[nc,:] = np.mean(clusterCentres[haveData,nc,:],0)
        sigmaCentres[nc,:] = np.std(clusterCentres[haveData,nc,:],0)/\
            np.sqrt(len(haveData))
    return [meanMasses,meanCentres,sigmaMasses,sigmaCentres,\
        clusterMasses,clusterCentres,clusterCounterparts]

def getShortPairCounts(snapNameList,centresList,radiiList,rBins,vorVols,\
        method="poisson"):
    pairsListVar = []
    volsListVar = []
    for ns in range(0,len(snapNameList)):
        snap = tools.getPynbodySnap(snapNameList[ns])
        gc.collect()
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
        gc.collect()
        [pairs,vols] = stacking.getPairCounts(\
                centresList[ns],\
                radiiList[ns],snap,rBins,\
                nThreads=-1,tree=tree,\
                method=method,vorVolumes=vorVols[ns])
        pairsListVar.append(pairs)
        volsListVar.append(vols)
    return [pairsListVar,volsListVar]

# Get void profile plot:
def getVoidProfilesForPaper(finalCatOpt,combinedFilter,\
        snapNameList,snapNumListUncon,centreListUn,rSphere=300,\
        rSphereInner=135,data_folder="./",recomputeData=True,catData=None,\
        rEffMin = 0.0,rEffMax = 10.0,nBins = 101,rMin=5,rMax=30,mMin=1e11,\
        mMax = 1e16,mUpper=2e15,mLower="auto",ahRBinsMin=10,ahRBinsMax=20,\
        ahRBins=6,unconstrainedFolderNew="new_chain/unconstrained_samples/",\
        snapname="gadget_full_forward_512/snapshot_001",\
        chainFile="chain_properties.p",Nden=256):
    if catData is None:
        catData = np.load(data_folder + "catalogue_data.npz")
    # reference snap:
    refSnap = pynbody.load(snapNameList[0])
    boxsize = refSnap.properties['boxsize'].ratio("Mpc a h**-1")
    rBins = np.linspace(rEffMin,rEffMax,nBins)
    rBinStackCentres = plot.binCentres(rBins)
    # Get centres and radii for catalogues in each sample:
    allProps = [tools.loadPickle(name + ".AHproperties.p") \
        for name in snapNameList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in allProps]
    antihaloCentresUnmapped = [props[5] for props in allProps]
    antihaloMasses = [props[3] for props in allProps]
    antihaloRadii = [props[7] for props in allProps]
    vorVols = [props[4] for props in allProps]
    # SNR data:
    [mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
        std_field,hmc_Elh,hmc_Eprior,hades_accept_count,\
        hades_attempt_count] = pickle.load(open(chainFile,"rb"))
    snrField = mean_field**2/std_field**2
    snrFieldLin = np.reshape(snrField,Nden**3)
    # Centres about which to compute SNR:
    grid = snapedit.gridListPermutation(Nden,perm=(2,1,0))
    centroids = grid*boxsize/Nden + boxsize/(2*Nden)
    positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
    tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
        boxsize=boxsize)
    nearestPointsList = [tree.query_ball_point(\
            snapedit.wrap(antihaloCentres[k] + boxsize/2,boxsize),\
            antihaloRadii[k],workers=-1) \
            for k in range(0,len(antihaloCentres))]
    snrAllCatsList = [np.array([np.mean(snrFieldLin[points]) \
            for points in nearestPointsList[k]]) \
            for k in range(0,len(snapNameList))]
    snrFilter = [snr > snrThresh for snr in snrAllCatsList]
    # More AH properties:
    centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
                filterCondition = (antihaloRadii[k] > rMin) & \
                (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
                (antihaloMasses[k] <= mMax) & snrFilter[k]) \
                for k in range(0,len(snapNameList))]
    centralAntihaloMasses = [\
                antihaloMasses[k][centralAntihalos[k][0]] \
                for k in range(0,len(centralAntihalos))]
    sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
            for k in range(0,len(snapNameList))]
    ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
    max_index = np.max(ahCounts)
    centresListShortUnmapped = [np.array([antihaloCentresUnmapped[l][\
        centralAntihalos[l][0][sortedList[l][k]],:] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNameList))]
    radiiListShort = [np.array([antihaloRadii[l][\
        centralAntihalos[l][0][sortedList[l][k]]] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNameList))]
    # Compute the trimmed catalogue:
    inTrimmedCatalogue = np.any((finalCatOpt >= 0) & combinedFilter[:,None],1)
    trimmedCatalogue = finalCatOpt[inTrimmedCatalogue]
    selectedVoids = np.where(inTrimmedCatalogue)[0]
    rBins = np.linspace(rEffMin,rEffMax,nBins)
    # Data for unconstrained simulations:
    snapListUnconstrained = [pynbody.load(unconstrainedFolderNew + "sample" \
        + str(snapNum) + "/" + snapname) for snapNum in snapNumListUncon]
    unmappedCentres = snapedit.wrap(\
        np.fliplr(catData['centres']) + boxsize/2,boxsize)
    Om = refSnap.properties['omegaM0']
    N = int(np.cbrt(len(refSnap)))
    nbar = len(refSnap)/boxsize**3
    mUnit = 8*Om*2.7754e11*(boxsize/N)**3
    if mLower == "auto":
        mLower = 100*mUnit
    ahPropsUnconstrained = [tools.loadPickle(snap.filename + ".AHproperties.p")\
        for snap in snapListUnconstrained]
    ahCentresListUn = [props[5] for props in ahPropsUnconstrained]
    ahCentresListRemapUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahPropsUnconstrained]
    antihaloRadiiUn = [props[7] for props in ahPropsUnconstrained]
    antihaloMassesListUn = [props[3] for props in ahPropsUnconstrained]
    deltaCentralListUn = [props[11] for props in ahPropsUnconstrained]
    centralAntihalosUn = [[tools.getAntiHalosInSphere(ahCentresListRemapUn[ns],\
            rSphereInner,origin=centre) for centre in centreListUn[ns]] \
            for ns in range(0,len(snapListUnconstrained))]
    conditionListMrangeUn = [[(deltaCentralListUn[ns] < 0) & \
        (centralAHs[1]) & (antihaloMassesListUn[ns] > mLower) & \
        (antihaloMassesListUn[ns] <= mUpper) & \
        (antihaloRadiiUn[ns] > rMin) & \
        (antihaloRadiiUn[ns] <= rMax)
        for centralAHs in centralAntihalosUn[ns]] \
        for ns in range(0,len(snapNumListUncon))]
    antihaloRadiusBins = np.linspace(ahRBinsMin,ahRBinsMax,ahRBins)
    [binListCon,noInBinsCon] = plot.binValues(\
        catData['radii'][combinedFilter],antihaloRadiusBins)
    allPairCountsUn = [[] for ns in range(0,len(snapNumListUncon))]
    allVolumesListsUn = [[] for ns in range(0,len(snapNumListUncon))]
    # Load the pair counts:
    for ns in range(0,len(snapNumListUncon)):
        for l in range(0,len(centreListUn[ns])):
            [newPairCountsUn,newVolumesListUn,filtersListUn] = \
                tools.loadPickle(data_folder + \
                    "pair_counts_data_unconstrained_sample_" + \
                    str(ns) + "_region_" + str(l) + ".p")
            allPairCountsUn[ns].append(newPairCountsUn[0])
            allVolumesListsUn[ns].append(newVolumesListUn[0])
    # Unconstrained profiles:
    [nbarjUnSameRadii,sigmaUnSameRadii] = \
    stacking.stackUnconstrainedWithConstrainedRadii(snapNumListUncon,\
        rBins,antihaloRadiusBins,noInBinsCon,conditionListMrangeUn,\
        antihaloRadiiUn,ahCentresListUn,allPairCountsUn,allVolumesListsUn)
    # Get pair counts about fixed void positions:
    [pairsListMean,volsListMean] = tools.loadOrRecompute(\
        data_folder + "fixed_centres.p",\
        stacking.pairCountsFixedPosition,snapNameList,\
        unmappedCentres,catData['radii'],rBins,_recomputeData=recomputeData)
    [pairsListVar,volsListVar] = tools.loadOrRecompute(\
        data_folder + "variable_centres.p",getShortPairCounts,snapNameList,\
        centresListShortUnmapped,radiiListShort,rBins,vorVols,\
        _recomputeData=recomputeData)
    # Compute averaged void profiles:
    [nbarj,sigmaj] = tools.loadOrRecompute(\
        data_folder + "mean_profiles.p",\
        stacking.computeAveragedIndividualProfiles,finalCatOpt,\
        pairsListMean,volsListMean,additionalFilter = combinedFilter,\
        _recomputeData=recomputeData)
    [nbarjVar,sigmajVar] = tools.loadOrRecompute(\
        data_folder + "var_profiles.p",\
        stacking.computeAveragedIndividualProfiles,finalCatOpt,\
        pairsListVar,volsListVar,additionalFilter = combinedFilter135,\
        existingOnly=True,_recomputeData=recomputeData)
    # Get the trimmed volumes list:
    radiiCombined = catData['radii'][selectedVoids]
    rBinsUp = rBins[1:]
    rBinsLow = rBins[0:-1]
    volumes = 4*np.pi*(rBinsUp**3 - rBinsLow**3)/3
    volumesListCombined = np.outer(radiiCombined**3,volumes)
    # Get mean profile:
    [nbarMean,sigmaMean] = tools.loadOrRecompute(\
        data_folder + "combined_profile_mean.p",\
        stacking.stackProfilesWithError,\
        nbarj,sigmaj,volumesListCombined,_recomputeData=recomputeData)
    [nbarVar,sigmaVar] = tools.loadOrRecompute(\
        data_folder + "combined_profile_var.p",\
        stacking.stackProfilesWithError,\
        nbarjVar,sigmajVar,volumesListCombined,_recomputeData=recomputeData)
    return [rBinStackCentres,nbarMean,sigmaMean,nbarVar,sigmaVar,nbar,\
        nbarjUnSameRadii,sigmaUnSameRadii]


