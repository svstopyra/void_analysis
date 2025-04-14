#-------------------------------------------------------------------------------
# CONFIGURATION
from void_analysis import plot, tools, snapedit, catalogue
from void_analysis.catalogue import *
from void_analysis.paper_plots_borg_antihalos_generate_data import *
from void_analysis.real_clusters import getClusterSkyPositions
from void_analysis import massconstraintsplot
from void_analysis.simulation_tools import ngPerLBin
from void_analysis.cosmology_inference import *
from matplotlib import transforms
import matplotlib.ticker
from matplotlib.ticker import NullFormatter
from matplotlib import cm
from matplotlib import patches
import matplotlib.lines as mlines
import matplotlib.colors as colors
import pickle
import numpy as np
import seaborn as sns
import pandas
seabornColormap = sns.color_palette("colorblind",as_cmap=True)
import pynbody
import astropy.units as u
from astropy.coordinates import SkyCoord
import scipy
import os
import sys

# Set global figure font:
import matplotlib.pyplot as plt
fontfamily = "serif"


# Data export options:
save_plot_data = True
load_plot_data = True

figuresFolder = "void_applications/"
recomputeData = False
testDataFolder = figuresFolder + "tests_data/"
runTests = False

# Filename data:
unconstrainedFolderNew = "new_chain/unconstrained_samples/"
unconstrainedFolderOld = "unconstrainedSamples/"
snapnameNew = "gadget_full_forward_512/snapshot_001"
snapnameNewRev = "gadget_full_reverse_512/snapshot_001"
samplesFolder="new_chain/"
samplesFolderOld = "./"
snapnameOld = "forward_output/snapshot_006"
snapnameOldRev = "reverse_output/snapshot_006"

data_folder = figuresFolder
data_folder2 = "borg-antihalos_paper_figures/all_samples/"

fontsize = 9
legendFontsize = 9

low_memory_mode = True

#-------------------------------------------------------------------------------
# LOAD SNAPSHOT DATA:



clusterNames = np.array([['Perseus-Pisces (A426)'],
       ['Hercules B (A2147)'],
       ['Coma (A1656)'],
       ['Norma (A3627)'],
       ['Shapley (A3571)'],
       ['A548'],
       ['Hercules A (A2199)'],
       ['Hercules C (A2063)'],
       ['Leo (A1367)']], dtype='<U21')


# HMF plots data:

# Snapshots to use:
snapNumList = [7300,7600,7900,8200,8500,8800,9100,9400,9700,10000,\
    10300,10600,10900,11200,11500,11800,12100,12400,12700,13000]


snapNumListUncon = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
boxsize = 677.7

# Get profiles for the constrained voids only:
snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" 
             + "gadget_full_forward_512/snapshot_001") 
             for snapNum in snapNumList]
snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" 
                + "gadget_full_reverse_512/snapshot_001") 
                for snapNum in snapNumList]
snapNameList = [samplesFolder + "sample" + str(snapNum) + "/" 
                + "gadget_full_forward_512/snapshot_001" 
                for snapNum in snapNumList]
snapNameListRev = [samplesFolder + "sample" + str(k) + "/" + snapnameNewRev \
    for k in snapNumList]

# Unconstrained simulations:
snapListUn = [pynbody.load("new_chain/unconstrained_samples/sample"
              + str(num) + "/gadget_full_forward_512/snapshot_001") 
              for num in snapNumListUncon]
snapListRevUn = [pynbody.load("new_chain/unconstrained_samples/sample"
                 + str(num) + "/gadget_full_reverse_512/snapshot_001") \
                 for num in snapNumListUncon]


def extract_antihalo_property(snap_list,prop):
    properties = []
    for snap in snap_list:
        props_list = tools.loadPickle(snap.filename + ".AHproperties.p")
        properties.append(props_list[prop])
        del props_list
        gc.collect()
    return properties


#-------------------------------------------------------------------------------
# SNAPSHOT GROUP CLASS

from void_analysis.cosmology_inference import SnapshotGroup

borg_snaps = SnapshotGroup(snapList,snapListRev,low_memory_mode=True,
                           swapXZ  = False,reverse = True,remap_centres=True)

lcdm_snaps = SnapshotGroup(snapListUn,snapListRevUn,
                                    low_memory_mode=True,swapXZ  = False,
                                    reverse = True,remap_centres=True)

#-------------------------------------------------------------------------------
# BUILDING VOID CATALOGUES FROM SCRATCH
recomputeCatalogues = False

# Parameters:
rSphere = 300 # Radius out to which to search for voids
muOpt = 0.75 # Optimal choice of \mu_R, radius ratio
rSearchOpt = 0.5 # Optimal choice of \mu_S, search radius ratio
NWayMatch = False # Whether to use N-way matching code.
refineCentres = True # Whether to refine centres iteratively using all samples
sortBy = "radius" # Quantity to sort catalogue by
enforceExclusive = True # Whether to apply the algorithm purging duplicate
    # voids from the catalogues.
mMin = 1e11 # Minimum mass halos to include (Note, this is effectively
    # over-ridden by the minimum radius threshold)
mMax = 1e16 # Maximum mass halos to include (Note, this is effectively 
    # over-ridden by the maximum radius threshold)
rMin = 5 # Minimum radius voids to use
rMax = 30 # Maximum radius voids to use
m_unit = borg_snaps.snaps[0]['mass'][0]*1e10

# Signal to noise information:
snrThresh=10
chainFile="chain_properties.p"
if not low_memory_mode:
    [snrFilter,snrAllCatsList] = getSNRFilterFromChainFile(
        chainFile,snrThresh,snapNameList,boxsize)

if recomputeCatalogues or (not os.path.isfile(data_folder2 + "cat300.p")):
    cat300 = catalogue.combinedCatalogue(
        borg_snaps.snap_filenames,borg_snaps.snap_reverse_filenames,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=borg_snaps.all_property_lists,hrList=borg_snaps["antihalos"],\
        max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [mMin,mMax],\
        NWayMatch = NWayMatch,r_min=rMin,r_max=rMax,\
        additionalFilters = snrFilter,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    cat300test = catalogue.combinedCatalogue(
        borg_snaps.snap_filenames,borg_snaps.snap_reverse_filenames,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=borg_snaps.all_property_lists,hrList=borg_snaps["antihalos"],\
        max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [m_unit*800,mMax],\
        NWayMatch = NWayMatch,r_min=10,r_max=30,\
        additionalFilters = snrFilter,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    finalCat300test = cat300test.constructAntihaloCatalogue()
    finalCat300 = cat300.constructAntihaloCatalogue()
    tools.savePickle(cat300,data_folder2 + "cat300.p")
else:
    cat300 = tools.loadPickle(data_folder2 + "cat300.p")

# Random catalogues:
if recomputeCatalogues or (not os.path.isfile(data_folder2 + "cat300Rand.p")):
    cat300Rand = catalogue.combinedCatalogue(
        lcdm_snaps.snap_filenames,lcdm_snaps.snap_reverse_filenames,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=lcdm_snaps.all_property_lists,hrList=lcdm_snaps["antihalos"],\
        max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [mMin,mMax],\
        NWayMatch = NWayMatch,r_min=rMin,r_max=rMax,\
        additionalFilters = None,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    cat300RandTest = catalogue.combinedCatalogue(
        lcdm_snaps.snap_filenames,lcdm_snaps.snap_reverse_filenames,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=lcdm_snaps.all_property_lists,hrList=lcdm_snaps["antihalos"],\
        max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [m_unit*800,mMax],\
        NWayMatch = NWayMatch,r_min=10,r_max=30,\
        additionalFilters = None,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    finalCat300RandTest = cat300RandTest.constructAntihaloCatalogue()
    finalCat300Rand = cat300Rand.constructAntihaloCatalogue()
    tools.savePickle(cat300Rand,data_folder2 + "cat300Rand.p")
else:
    cat300Rand = tools.loadPickle(data_folder2 + "cat300Rand.p")

# Apply Catalogue filter:
nBinEdges = 8
rLower = 10
rUpper = 20
radBins = np.linspace(rLower,rUpper,nBinEdges)
cat300.set_filter_from_random_catalogue(cat300Rand,radBins,r_min=10,r_max=25,
                                        r_sphere=135)



#-------------------------------------------------------------------------------
# CATALOGUE DATA:

ns = 0
doSky=True
snapToShow = pynbody.load(samplesFolder + "sample7000" + \
    "/gadget_full_forward_512/snapshot_001")
tools.remapBORGSimulation(snapToShow,swapXZ=False,reverse=True)

rCut = 135
ha = ['right','left','left','left','left','center','right',\
        'right','right']
va = ['center','left','bottom','bottom','top',\
        'top','center','center','center']
annotationPos = [[-1.1,0.9],\
        [1.3,0.1],[1.5,0.6],[1.3,-1.2],[1.3,-0.7],[-1,0.2],[0.8,0.6],\
        [1.0,0.1],[-1.7,0.5]]
nameList = [name[0] for name in clusterNames]
textwidth=7.1014
textheight=9.0971
scale = 1.26
width = textwidth
height = 0.55*textwidth
cropPoint = ((scale -1)/2)*np.array([width,height]) + np.array([0,0.09])
bound_box = transforms.Bbox([[cropPoint[0], cropPoint[1]],
    [cropPoint[0] + width, cropPoint[1] + height]])

# Cluster locations:
# Galaxy positions:
if doSky:
    [combinedAbellN,combinedAbellPos,abell_nums] = \
        real_clusters.getCombinedAbellCatalogue()
    abell_nums = [426,2147,1656,3627,3571,548,2197,2052,1367]
    [abell_l,abell_b,abell_n,abell_z,\
            abell_d,p_abell,coordAbell] = getClusterSkyPositions("./")
    clusterInd = [np.where(combinedAbellN == n)[0] for n in abell_nums]
    clusterIndMain = [ind[0] for ind in clusterInd]
    coordCombinedAbellCart = SkyCoord(x=combinedAbellPos[:,0]*u.Mpc,\
            y = combinedAbellPos[:,1]*u.Mpc,z = combinedAbellPos[:,2]*u.Mpc,\
            frame='icrs',representation_type='cartesian')
    equatorialRThetaPhi = np.vstack(\
        [coordCombinedAbellCart.icrs.spherical.distance.value,\
        coordCombinedAbellCart.icrs.spherical.lat.value*np.pi/180.0,\
        coordCombinedAbellCart.icrs.spherical.lon.value*np.pi/180]).transpose()
    coordCombinedAbellSphere = SkyCoord(distance=\
        coordCombinedAbellCart.icrs.spherical.distance.value*u.Mpc,\
        ra = coordCombinedAbellCart.icrs.spherical.lon.value*u.deg,\
        dec = coordCombinedAbellCart.icrs.spherical.lat.value*u.deg,\
        frame='icrs')

clusterLoc = np.array([np.array([\
    coordCombinedAbellCart[ind].x.value,\
    coordCombinedAbellCart[ind].y.value,\
    coordCombinedAbellCart[ind].z.value]) for ind in clusterIndMain])

referenceSnap = snapToShow
Om0 = referenceSnap.properties['omegaM0']
Ode0 = referenceSnap.properties['omegaL0']
H0 = referenceSnap.properties['h']*100
h = referenceSnap.properties['h']
boxsize = referenceSnap.properties['boxsize'].ratio("Mpc a h**-1")
cosmo = astropy.cosmology.LambdaCDM(H0,Om0,Ode0)
nbar = len(referenceSnap)/boxsize**3

# 2M++ Data:
catFile = "./2mpp_data/2m++.txt"
catalogueData = np.loadtxt(catFile,delimiter='|',skiprows=31,
    usecols=(1,2,3,4,5,6,7,8,10,11,12,13,14,15,16))
# Filter useable galaxies:
useGalaxy = (catalogueData[:,10] == 0.0) & (catalogueData[:,5] > 0)
c = 299792.458 # Speed of light in km/s
z = catalogueData[:,5]/c # Redshift
# Cosmological parameters:

# Comoving distance to all galaxies, in Mpc/h:
dcz = cosmo.comoving_distance(z[useGalaxy]).value*cosmo.h
# Co-ordinates of the galaxies (in Mpc/h):
coord = astropy.coordinates.SkyCoord(\
    ra = catalogueData[useGalaxy,0]*astropy.units.degree,\
    dec=catalogueData[useGalaxy,1]*astropy.units.degree,\
    distance=dcz*astropy.units.Mpc)
# Cartesian positions of galaxies in equatorial, comoving co-ordinates (Mpc/h):
equatorialXYZ = np.vstack((coord.cartesian.x.value,\
    coord.cartesian.y.value,coord.cartesian.z.value)).T
# In spherical polar co-ordinates:
equatorialRThetaPhi = np.vstack((coord.icrs.spherical.distance.value,\
    coord.icrs.spherical.lon.value,\
    coord.icrs.spherical.lat.value)).T



#-------------------------------------------------------------------------------
# STACKED VOIDS

# Stack all voids relative to their barycentre:

# Get points within an ellipse of radius R, and ellipticity eps, given a z
# position and displacement from the axis
# Get and save the line of sight positions:

from void_analysis.simulation_tools import redshift_space_positions
from void_analysis.simulation_tools import get_los_pos_for_snapshot
from void_analysis.simulation_tools import get_los_positions_for_all_catalogues
from void_analysis.plot import draw_ellipse, plot_los_void_stack



#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
# COSMOLOGY CONNECTION






#-------------------------------------------------------------------------------
# END TO END FUNCTION

# Bins in LOS co-ords:
upper_dist_reff = 2
spar_bins = np.linspace(0,upper_dist_reff,21)
sperp_bins = np.linspace(0,upper_dist_reff,21)
bin_z_centres = plot.binCentres(spar_bins)
bin_d_centres = plot.binCentres(sperp_bins)

# LCDM LOS positions
# Get densities for MCMC simulations, so we can find comparable LCDM regions:
densities_file = data_folder2 + "delta_list.p"
deltaMAPBootstrap, deltaMAPInterval = get_borg_density_estimate(
    borg_snaps,densities_file = densities_file)
# Creat voids list for lcdm:
centres_file = data_folder2 + "random_centres_and_densities.p"
voids_used_lcdm = get_lcdm_void_catalogue(lcdm_snaps,deltaMAPBootstrap,
                                          centres_file = centres_file)
voids_used_lcdm_all = get_lcdm_void_catalogue(lcdm_snaps,deltaMAPBootstrap,
                                              centres_file = centres_file,
                                              flattened=False)

los_lcdm_zspace = get_los_positions_for_all_catalogues(
    lcdm_snaps["snaps"],lcdm_snaps["snaps_reverse"],lcdm_snaps["void_centres"],
    lcdm_snaps["void_radii"],all_particles=True,
    filter_list=voids_used_lcdm,dist_max=3,
    rmin=10,rmax=20,recompute=False,zspace=True,recompute_zspace=False,
    suffix=".lospos_all_zspace_selected.p",flatten_filters=True)

los_lcdm_real = get_los_positions_for_all_catalogues(
    lcdm_snaps["snaps"],lcdm_snaps["snaps_reverse"],
    lcdm_snaps["void_centres"],lcdm_snaps["void_radii"],all_particles=True,
    filter_list=voids_used_lcdm,dist_max=10,rmin=10,rmax=20,
    recompute=True,zspace=False,suffix=".lospos_all.p")

# BORG LOS positions:
halo_indices = cat300.get_final_catalogue(void_filter=True,short_list=False)
filter_list_borg = [halo_indices[ns] >= 0 for ns in range(0,borg_snaps.N)]
zcentres = tools.loadOrRecompute(data_folder + "zspace_centres.p",
                                 get_zspace_centres,halo_indices,
                                 borg_snaps["snaps"],
                                 borg_snaps["snaps_reverse"],hrlist=None,
                                 _recomputeData=False)

los_borg_zspace = get_los_positions_for_all_catalogues(
    borg_snaps["snaps"],borg_snaps["snaps_reverse"],zcentres,
    cat300.getAllProperties("radii",void_filter=True).T,
    all_particles=True,void_indices = halo_indices,
    filter_list=filter_list_borg,dist_max=3,rmin=10,rmax=20,recompute=False,
    zspace=True,recompute_zspace=False,suffix=".lospos_all_zspace2.p")


los_borg_real = get_los_positions_for_all_catalogues(
    borg_snaps["snaps"],borg_snaps["snaps_reverse"],
    cat300.getAllCentres(void_filter=True),
    cat300.getAllProperties("radii",void_filter=True).T,all_particles=True,
    filter_list=filter_list_borg,dist_max=10,rmin=10,rmax=20,
    void_indices = halo_indices,recompute=True,zspace=False,
    suffix=".lospos_all.p")

# Trimmed LOS lists:
void_radii_borg = cat300.getMeanProperty("radii",void_filter=True)[0]
void_individual_radii_borg = cat300.getAllProperties("radii",void_filter=True).T

los_list_trimmed_borg, voids_used_borg = trim_los_list(
    los_borg_zspace,spar_bins,sperp_bins,void_individual_radii_borg)

los_list_trimmed_lcdm, voids_used_lcdm = trim_los_list(
    los_lcdm_zspace,spar_bins,sperp_bins,lcdm_snaps["void_radii"])


# Additional weights for BORG, based on reproducibility score:
additional_weights_unfiltered_borg = get_additional_weights_borg(cat300)
additional_weights_borg = get_additional_weights_borg(
    cat300,voids_used = voids_used_borg)

rbins = np.linspace(0,10,101)

# BORG density field:
field_borg_test = get_stacked_void_density_field(
    borg_snaps,cat300.getAllProperties("radii",void_filter=True).T,zcentres,
    spar_bins,sperp_bins,
    additional_weights = additional_weights_unfiltered_borg,
    los_pos = los_borg_zspace,filter_list = filter_list_borg)

field_borg_1d, field_borg_1d_sigma = get_1d_real_space_field(
    borg_snaps,cat300.getAllProperties("radii",void_filter=True).T,
    cat300.getAllCentres(void_filter=True),rbins,
    filter_list=filter_list_borg,
    additional_weights=additional_weights_unfiltered_borg,dist_max=3,
    rmin=10,rmax=20,suffix=".lospos_all.p",
    recompute=False,nbar=nbar,los_pos=los_borg_real)


# LCDM density field:
field_lcdm_test = get_stacked_void_density_field(
    lcdm_snaps,lcdm_snaps["void_radii"],lcdm_snaps["void_centres"],
    spar_bins,sperp_bins,filter_list=voids_used_lcdm,recompute=False,
    los_pos = los_lcdm_zspace)

field_lcdm_1d, field_lcdm_1d_sigma = get_1d_real_space_field(
    lcdm_snaps,lcdm_snaps["void_radii"],
    lcdm_snaps["void_centres"],rbins,
    filter_list=voids_used_lcdm,dist_max=3,rmin=10,rmax=20,
    suffix=".lospos_all.p",recompute=False,nbar=nbar,
    los_pos = los_lcdm_real)


#-------------------------------------------------------------------------------
# TEST PLOTS FOR REAL SPACE FIELD

# Test Plot:
fig, ax = plt.subplots()
ax.plot(rvals,delta_func(rvals),label='$\\delta(r)$')
ax.plot(rvals,Delta_func(rvals),label='$\\Delta(r)$')
ax.set_xlabel('r [$\\mathrm{Mpc}h^{-1}$]')
ax.set_ylabel('$\\rho(r)$')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "rho_real_plot.pdf")
plt.show()

start_values = np.linspace(0,0.5,101)

x = r_bin_centres
y = rho_r_mean

fit_large = np.polyfit(np.log(x[x > 1.5]),np.log(y[x > 1.5]),1)
fit_small = np.polyfit(np.log(x[x < 0.5]),np.log(y[x < 0.5]),1)

plt.clf()
fig, ax = plt.subplots()
ax.fill_between(r_bin_centres,rho_r_range_2sigma[0,:],rho_r_range_2sigma[1,:],
    alpha=0.25,color=seabornColormap[0],ec=None,
    label='$\\rho_{\\Lambda\\mathrm{CDM}}(r), 95\\%$')
ax.fill_between(r_bin_centres,rho_r_range_1sigma[0,:],rho_r_range_1sigma[1,:],
    alpha=0.5,color=seabornColormap[0],ec=None,
    label='$\\rho_{\\Lambda\\mathrm{CDM}}(r), 68\\%$')
ax.errorbar(r_bin_centres,rho_r_borg_mean,
    yerr=np.abs(rho_r_borg_range_1sigma - rho_r_borg_mean),
    label='$\\rho_{\\mathrm{borg}}(r)$',color=seabornColormap[1])
#ax.plot(r_bin_centres,field_borg[0]/np.mean(rho_func_borg_z0(start_values)),
#    label='$\\rho_{\\mathrm{2d}}(0,d)$')
ax.plot(rvals,np.exp(fit_large[0]*np.log(rvals) + fit_large[1]),
        linestyle=':',color='k',
        label="$\\rho = " + ("%.2g" % (np.exp(fit_large[1]))) + "\\cdot " + 
        "r^{" + ("%.2g" % (fit_large[0])) + "}$")
ax.plot(rvals,np.exp(fit_small[0]*np.log(rvals) + fit_small[1]),
        linestyle='--',color='k',
        label="$\\rho = " + ("%.2g" % (np.exp(fit_small[1]))) + "\\cdot " + 
        "r^{" + ("%.2g" % (fit_small[0])) + "}$")
ax.set_xlabel('$r/r_{\\mathrm{eff}}$')
ax.set_ylabel('$\\rho(r)$')
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_ylim([1e-3,0.1])
ax.set_xlim([0,2])
plt.legend(frameon=False)
plt.savefig(figuresFolder + "rho_real_plot_void_only.pdf")
plt.show()

# 2D profile function test (zspace):
z = 0.0225
profile_2d = np.zeros((len(spar_bins)-1,len(sperp_bins)-1))

Om = 0.3111
f = f_lcdm(z,Om)
A = 0.013

#-------------------------------------------------------------------------------
# COVARIANCE MATRIX ESTIMATION:



v_weights_all_borg = get_void_weights(
    los_list_trimmed_borg,voids_used_borg,void_individual_radii_borg,
    additional_weights = additional_weights_borg)

v_weights_all_lcdm = get_void_weights(los_list_trimmed_lcdm,voids_used_lcdm,
                                       lcdm_snaps["void_radii"])



rho_cov_borg, rhomean_borg = get_covariance_matrix(
    los_borg_zspace,cat300.getAllProperties("radii",void_filter=True).T,
    spar_bins,sperp_bins,nbar,
    additional_weights=additional_weights_unfiltered_borg,return_mean=True,
    cholesky=False)
cholesky_cov_borg = scipy.linalg.cholesky(rho_cov_borg,lower=True)

logrho_cov_borg, logrho_mean_borg = get_covariance_matrix(
    los_borg_zspace,cat300.getAllProperties("radii",void_filter=True).T,
    spar_bins,sperp_bins,nbar,
    additional_weights=additional_weights_unfiltered_borg,return_mean=True,
    cholesky=False,log_field=True)
log_cholesky_cov_borg = scipy.linalg.cholesky(logrho_cov_borg,lower=True)


rho_cov_lcdm, rhomean_lcdm = get_covariance_matrix(
    los_lcdm_zspace,lcdm_snaps["void_radii"],
    spar_bins,sperp_bins,nbar,additional_weights=None,
    return_mean=True,cholesky=False)
cholesky_cov_lcdm = scipy.linalg.cholesky(rho_cov_lcdm,lower=True)

logrho_cov_lcdm, logrho_mean_lcdm = get_covariance_matrix(
    los_lcdm_zspace,lcdm_snaps["void_radii"],
    spar_bins,sperp_bins,nbar,additional_weights=None,
    return_mean=True,cholesky=False,log_field=True)
log_cholesky_cov_lcdm = scipy.linalg.cholesky(logrho_cov_lcdm,lower=True)


#-------------------------------------------------------------------------------
#
#A,r0,c1,f1,B = sol2.x


# Inference for LCDM mock (end-to-end test):





# Inference for Borg catalogue:
tau, sampler = run_inference_pipeline(
    field_lcdm_test,rho_cov_lcdm,sperp_bins,spar_bins,ri,field_lcdm_1d-1.0,
    field_lcdm_1d_sigma,log_field=False,infer_profile_args=True,
    tabulate_inverse=True,cholesky=True,sample_epsilon=True,filter_data=False,
    z = 0.0225,lambda_cut=1e-23,lambda_ref=1e-27,
    profile_param_ranges = [[0,np.inf],[0,np.inf],[0,np.inf],[-1,0],[-1,1],
    [0,2]],om_ranges = [[0.1,0.5]],eps_ranges = [[0.9,1.1]],f_ranges = [[0,1]],
    Om_fid = 0.3111,filename = "inference_weighted.h5",
    autocorr_filename = "autocorr.npy",disp=1e-2,nwalkers=64,n_mcmc=10000,
    max_n=1000000,batch_size=100,nbatch=100,redo_chain=False,backup_start=True)


# Profiles test:

ri = plot_utilities.binCentres(rbins)

params_lcdm = get_profile_parameters_fixed(ri,field_lcdm_1d-1,
                                           field_lcdm_1d_sigma)
params_borg = get_profile_parameters_fixed(ri,field_borg_1d-1,
                                           field_borg_1d_sigma)


plt.clf()
rrange = np.linspace(0,2,1000)
plt.plot(rrange,profile_modified_hamaus(rrange,*params_lcdm),
         color=seabornColormap[0],label="LCDM model")
plt.fill_between(ri,field_lcdm_1d - 1 - field_lcdm_1d_sigma,
                 field_lcdm_1d - 1 + field_lcdm_1d_sigma,
                 color=seabornColormap[0],label="LCDM simulated",
                 alpha=0.5)
plt.plot(rrange,profile_modified_hamaus(rrange,*params_borg),
         color=seabornColormap[1],label="MCMC model")
plt.fill_between(ri,field_borg_1d - 1 - field_borg_1d_sigma,
                 field_borg_1d - 1 + field_borg_1d_sigma,
                 color=seabornColormap[1],label="MCMC simulated",
                 alpha=0.5)
plt.legend(frameon=False)
plt.xlabel("$r/r_{\\mathrm{eff}}$")
plt.ylabel("\\delta(r)")
plt.savefig(figuresFolder + "profile_fit_test.pdf")
plt.show()







# Eigenvalue quality filter:

#Umap, good_eig = get_nonsingular_subspace(jackknife_cov,1e-27,lambda_cut=1e-23,
#                                          normalised_cov = False,
#                                          mu=jackknife_mean)


#covariance = jackknife_cov
#mean = jackknife_mean
#norm_cov = reg_norm_cov
#cholesky = cholesky_cov

covariance = logrho_cov
mean = logrho_mean
#norm_cov = reg_norm_logcov
cholesky_matrix = log_cholesky_cov

Umap, good_eig = get_nonsingular_subspace(covariance,1e-27,lambda_cut=1e-23,
                                          normalised_cov = False,
                                          mu=mean)
singular = True


spar = np.hstack([s*np.ones(field_borg.shape[1]) 
    for s in plot.binCentres(spar_bins)])
sperp = np.hstack([plot.binCentres(sperp_bins) 
    for s in plot.binCentres(spar_bins)])
scoords = np.vstack([spar,sperp]).T

data_filter = np.where((1.0/np.sqrt(np.diag(norm_cov)) > 5) & \
        (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]


# Data to compare to:
#data_field = field_borg.flatten()
data_field = np.log(field_borg.flatten())
spar = np.hstack([s*np.ones(field_borg.shape[1]) 
    for s in plot.binCentres(spar_bins)])
sperp = np.hstack([plot.binCentres(sperp_bins) 
    for s in plot.binCentres(spar_bins)])
scoords = np.vstack([spar,sperp]).T
z = 0.0225

# Covariances plot:
plt.clf()
plt.imshow(norm_cov,vmin=-1e-3,vmax=1e-3,cmap='PuOr_r')
plt.savefig(figuresFolder + "covariance_plot.pdf")
plt.show()

# 

theta_initial_guess = np.array([0.3,f_lcdm(z,0.3),0.01])
test = False
import time

if test:
    t0 = time.time()
    for k in tools.progressbar(range(0,100)):
        logp = log_probability_aptest(theta_initial_guess,data_field,scoords,
                                      cholesky_matrix,z,Delta_func,delta_func,
                                      rho_real,Om_fid = 0.3111,cholesky=True,
                                      tabulate_inverse=True)
    t1 = time.time()
    average_time = (t1 - t0)/100


#log_likelihood_aptest(theta,data_field,scoords,inv_cov,
#                          z,Delta,delta,rho_real,data_filter=None,
#                          cholesky=False,normalised=False,tabulate_inverse=True,
#                          ntab = 10,sample_epsilon=False,Om_fid=None,
#                          singular=False,Umap=None,good_eig=None,
#                          F_inv=None,log_density=False,**kwargs)

# Start with a MLE guess of the 1d profile parameters:






# Inference run:


# Tabulated inverse:
ntab = 100
svals = np.linspace(0,3,ntab)
rperp_vals = np.linspace(0,3,ntab)
ntab_f = 100
f_vals = np.linspace(0,1,ntab_f)
F_inv_vals = np.zeros((ntab,ntab,ntab_f))
for i in tools.progressbar(range(0,ntab)):
    for j in range(0,ntab):
        for k in range(0,ntab_f):
            F = (lambda r: r - r*(f_vals[k]/3)*\
                Delta_func(np.sqrt(r**2 + rperp_vals[i]**2)) \
                - svals[j])
            F_inv_vals[i,j,k] = scipy.optimize.fsolve(F,svals[j])

F_inv = lambda x, y, z: scipy.interpolate.interpn((rperp_vals,svals,f_vals),
                                               F_inv_vals,
                                               np.vstack((x,y,z)).T,
                                               method='cubic')



# Run the inference:
#profile_param_ranges = [[-np.inf,np.inf],[0,2],[-np.inf,0],[0,np.inf],[-1,1]]
profile_param_ranges = [[0,np.inf],[0,np.inf],[0,np.inf],[-1,0],[-1,1],[0,2]]
om_ranges = [[0.1,0.5]]
eps_ranges = [[0.9,1.1]]
f_ranges = [[0,1]]
z = 0.0225
Om_fid = 0.3111
eps_initial_guess = np.array([1.0,f_lcdm(z,Om_fid)])
theta_initial_guess = np.array([0.3,f_lcdm(z,0.3)])

initial_guess_eps = np.hstack([eps_initial_guess,sol2.x])
initial_guess_theta = np.hstack([theta_initial_guess,sol2.x])

theta_ranges=om_ranges + f_ranges + profile_param_ranges
theta_ranges_epsilon = eps_ranges + f_ranges + profile_param_ranges

args = (data_field[data_filter],scoords[data_filter,:],
        cholesky_matrix[data_filter,:][:,data_filter],z,Delta_func,
        delta_func,rho_real)
kwargs = {'cholesky':True,'tabulate_inverse':True,
          'sample_epsilon':True,'theta_ranges':theta_ranges_epsilon,
          'singular':False,'Umap':Umap,'good_eig':good_eig,'F_inv':F_inv,
          'log_density':True}

# Wrapper allowing us to pass arbitrary arguments that won't be sampled over:
def posterior_wrapper(theta,additional_args,*args,**kwargs):
    theta_comb = np.hstack([theta,additional_args])
    return log_probability_aptest(theta_comb,*args,**kwargs)

nll = lambda theta: -log_likelihood_aptest(theta,*args,**kwargs)
mle_estimate = scipy.optimize.minimize(nll,initial_guess_eps,
                                       bounds=theta_ranges_epsilon)


plt.clf()
fig, ax = plt.subplots(figsize=(textwidth,0.45*textwidth))
plt.errorbar(rBinStackCentres,delta_borg,yerr=sigma_delta_borg,linestyle='-',
             color=seabornColormap[1],label="BORG profile")
plt.plot(rBinStackCentres,
         profile_modified_hamaus(rBinStackCentres,*mle_estimate.x[2:]),
         label="MLE Profile (joint inference)")
plt.plot(rBinStackCentres,
         profile_modified_hamaus(rBinStackCentres,*sol2.x),
         label="MLE Profile (Separate inference)")

plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$\\rho(r)$')
plt.xlim([0,10])
plt.ylim([0,1.1])
#plt.yscale('log')
#plt.xscale('log')
plt.legend(frameon=False,loc="lower right")
plt.tight_layout()
plt.savefig(figuresFolder + "profile_fit_test3.pdf")
plt.show()







import emcee

tau, sampler = run_inference(data_field,theta_ranges_epsilon,mle_estimate.x,
                             data_folder + "inference_weighted.h5",
                             log_probability_aptest,*args,redo_chain=False,
                             backup_start=True,nwalkers=64,sample="all",
                             n_mcmc=10000,disp=1e-1,
                             max_n=1000000,z=0.0225,parallel=False,Om_fid=0.3111,
                             batch_size=100,n_batches=100,data_filter=None,
                             autocorr_file = data_folder + "autocorr.npy",
                             **kwargs)

sampler = emcee.backends.HDFBackend(
    data_folder + "inference_weighted_2025_04_03.h5")
tau = sampler.get_autocorr_time(tol=0)
chain = sampler.get_chain(discard=2000)

jump = 100
intervals = np.arange(0,chain.shape[0],jump)
taus = np.zeros((len(intervals)-1,chain.shape[1],chain.shape[2]))
for k in tools.progressbar(range(0,chain.shape[1])):
    for l in range(0,len(intervals)-1):
        taus[l,k,:] = emcee.autocorr.integrated_time(chain[0:intervals[l+1],
                                                     [k],:],tol=0)

# Taus plot:
plt.clf()
plt.plot(intervals[1:],taus[:,5,:])
plt.savefig(figuresFolder + "taus.pdf")
plt.show()

std_chain = np.std(chain,0)

mean_chain = np.mean(chain,1)
tau_mean = emcee.autocorr.integrated_time(
    mean_chain.reshape(chain.shape[0],1,chain.shape[2]),tol=0)

# Chains plot:
plt.clf()
#plt.plot(np.sign(mean_chain)*np.log10(1 + np.abs(mean_chain)))
plt.plot(chain[:,5,:])
plt.ylim([-5,5])
plt.savefig(figuresFolder + "all_chains.pdf")
plt.show()



all_samples = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])
#flat_samples = sampler.get_chain(discard = 300,thin=50,flat=True)

redo_chain = False
continue_chain = True
backup_start = True
import emcee
import h5py
nwalkers = 64
ndims = 2
n_mcmc = 10000
disp = 1e-4
max_n = 1000000


batch_size = 100

#data_filter = np.where(np.sqrt(np.sum(scoords**2,1)) < 1.5)[0]

data_filter = np.where((1.0/np.sqrt(np.diag(norm_cov)) > 5) & \
         (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]


# Filter the MCMC samples to account for correlation:
tau = sampler.get_autocorr_time(tol=0)
tau_max = np.max(tau)
flat_samples = sampler.get_chain(discard=int(3*tau_max), 
                                 thin=int(tau_max/2), flat=True)

flat_mean = np.mean(flat_samples,0)
flat_range = np.percentile(flat_samples,[16,84],axis=0)


import corner

plt.clf()
#fig = corner.corner(flat_samples, labels=["$\\Omega_{m}$","$f$","A"])
#fig = corner.corner(flat_samples, labels=["$\\epsilon$","$f$","$A$","$r_0$",
#    "$c_1$","$f_1$","$B$"])
#param_ranges = [[0.9,1.1],[0,1.0]]
param_ranges = None
fig = corner.corner(flat_samples,labels=["$\\epsilon$","$f$"],
                    range=param_ranges)
fig.suptitle("$\\Lambda$-CDM Inference from Void Catalogue")
plt.savefig(figuresFolder + "corner_plot_cosmo_inference_cosmo_only.pdf")
plt.show()


# Tests of the inverse function:
#X = scoords[:,0].reshape((20,20))
#Y = scoords[:,1].reshape((20,20))
X = np.hstack([np.array([x for k in range(0,100)]) 
    for x in np.linspace(0,2,100)])
Y = np.hstack([np.linspace(0,2,100) for k in range(0,100)])

inv_field = F_inv(X,Y,
                  0.25*np.ones(len(X))).reshape((100,100))


s_par_range = np.linspace(0,2,40)
s_perp_range = np.linspace(0,2,40)
f_range = np.linspace(0,1,40)
X, Y, Z = [x.flatten() for x in 
    np.meshgrid(s_par_range,s_perp_range,f_range)]

inv_field_3d = F_inv(X,Y,Z).reshape((40,40,40))

plt.clf()
im = plt.imshow(inv_field,cmap='Blues',vmin=0,vmax=2,extent=(0,2,0,2))
plt.xlabel('$s_{\\perp}/R_{\\mathrm{eff}}$')
plt.ylabel('$s_{\\parallel}/R_{\\mathrm{eff}}$')
plt.colorbar()
plt.savefig(figuresFolder + "inverse_field.pdf")
plt.show()

plt.clf()
fig, ax = plt.subplots(1,3)
slices = [inv_field_3d[:,10,20],inv_field_3d[27,:,13],inv_field_3d[20,39,:]]
ranges = [s_par_range,s_perp_range,f_range]
xnames = ['$s_{\\parallel}/R_{\\mathrm{eff}}$',
          '$s_{\\perp}/R_{\\mathrm{eff}}$','$f$']
data_points = []
for axi, y, xvals, name in zip(ax, slices,ranges,xnames):
    axi.plot(xvals,y)
    axi.set_xlabel('$s_{\\parallel}/R_{\\mathrm{eff}}$')
    axi.set_ylabel('$F^{-1}$')

plt.tight_layout()
plt.savefig(figuresFolder + "inverse_field_slice.pdf")
plt.show()


# Optimal A:
args2 = (data_field,scoords,np.identity(1600),z,Delta_func,delta_func,rho_real)
kwargs2={'Om_fid':Om_fid,'cholesky':True,'tabulate_inverse':True,
    'data_filter':data_filter,'sample_epsilon':False,'normalised':True}
sol_A = scipy.optimize.minimize_scalar(
    lambda A: -log_likelihood_aptest(np.array([Om_fid,f_lcdm(z,Om_fid),A]),
    *args2,**kwargs2))
A_opt = sol_A.x

# Fix the amplitude, and plot the likelihood:
Om_fid = 0.3111
data_filter = np.where((1.0/np.sqrt(np.diag(norm_cov)) > 5) & \
        (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]
reg_cov_filtered = reg_cov[data_filter,:][:,data_filter]
cholesky_cov_filtered = scipy.linalg.cholesky(reg_cov_filtered,lower=True)

profile_param_ranges_hamaus = [(0,np.inf),(0,np.inf),(0,np.inf),(-1,0),
                               (-1,1),(0,2)]

theta_ranges_hamaus = eps_ranges + f_ranges + profile_param_ranges_hamaus

args=(data_field[data_filter],scoords[data_filter,:],
                  cholesky_matrix[data_filter,:][:,data_filter],z,Delta_func,
                  delta_func,rho_real)
kwargs={'Om_fid':Om_fid,'cholesky':True,'tabulate_inverse':True,
        'sample_epsilon':True,'theta_ranges':theta_ranges_hamaus,
        'singular':False,'Umap':Umap,'good_eig':good_eig,
        'F_inv':F_inv,'log_density':True}
#data_filter = np.where(np.sqrt(np.sum(scoords**2,1)) < 2.0)[0]
#kwargs={'Om_fid':0.3111,'data_filter':data_filter}
#Om_fid = 0.3111
#kwargs={'Om_fid':Om_fid,'cholesky':True,'tabulate_inverse':True,
#    'sample_epsilon':True}
#params = combine_parameters(np.mean(flat_samples,0),fixed)
#params = np.mean(flat_samples[:,2:],0)
params = mle_estimate.x[2:]
eps_initial_guess = np.hstack([np.array([1.0,f_lcdm(z,Om_fid)]),params])
nll = lambda *theta: -log_likelihood_aptest(np.hstack([*theta,params]),*args,
                                            **kwargs)
nlp = lambda *theta: -log_probability_aptest(np.hstack([*theta,params]),
                                                 *args,**kwargs)
#nll = lambda *theta: -log_likelihood_aptest(np.hstack([*theta,A_opt]),
#                                            *args,**kwargs)
soln = scipy.optimize.minimize(nll, np.array([1.0,f_lcdm(z,Om_fid)]),
    bounds = [(0.9,1.1),(0,1.0)])
solnp = scipy.optimize.minimize(nlp, np.array([1.0,f_lcdm(z,Om_fid)]),
    bounds = [(0.9,1.1),(0,1.0)])
    #bounds=[(0.9,1.1),(0,1.0),(None,None),(0,2),(None,0),(0,None),(-1,1)])
#soln = scipy.optimize.minimize(nll, theta_initial_guess,
#    bounds=[(0.1,0.5),(0,1.0),(None,None)])
theta_initial_guess2 = np.array([Om_fid,f_lcdm(z,Om_fid)])
#soln = scipy.optimize.minimize(nll, theta_initial_guess2,
#    bounds=[(0.1,0.5),(0,1.0)])
#soln = scipy.optimize.minimize(nll, eps_initial_guess,
#    bounds=[(0.9,1.1),(0,1.0)])


Om_range = np.linspace(0.1,0.5,41)
f_range = np.linspace(0,1,41)
f_range_centres = plot.binCentres(f_range)
Om_range_centres = plot.binCentres(Om_range)
eps_range = np.linspace(0.9,1.1,41)
eps_centres = plot.binCentres(eps_range)

log_like_ap = np.zeros((40,40))
#A,r0,c1,f1,B = sol2.x
alpha,beta,rs,delta_c,delta_large, rv = sol2.x
#params = combine_parameters(np.mean(flat_samples1,0),fixed)
#A = A_opt
params = mle_estimate.x[2:]
#params = sol2.x
for i in tools.progressbar(range(0,len(Om_range_centres))):
    for j in range(0,len(f_range_centres)):
        if kwargs['sample_epsilon']:
            theta = np.array([eps_range[i],f_range_centres[j],*params])
        else:
            theta = np.array([Om_range_centres[i],f_range_centres[j],*params])
        #log_like_ap[i,j] = log_likelihood_aptest(theta,*args,**kwargs)
        log_like_ap[i,j] = log_probability_aptest(theta,*args,**kwargs)


if kwargs['sample_epsilon']:
    plt.clf()
    #im = plt.imshow(-log_like_ap.T,
    #           extent=(eps_range[0],eps_range[-1],f_range[0],f_range[-1]),
    #           norm=colors.LogNorm(vmin=1e9,vmax=1e10),cmap='Blues',
    #           aspect='auto',origin='lower')
    im = plt.imshow(-log_like_ap.T,
               extent=(eps_range[0],eps_range[-1],f_range[0],f_range[-1]),
               norm=colors.LogNorm(vmin=100,vmax=2e3),cmap='Blues',
               aspect='auto',origin='lower')
    X, Y = np.meshgrid(eps_centres,f_range_centres)
    CS = plt.contour(X,Y,-log_like_ap.T,levels = 10)
    plt.clabel(CS, inline=True, fontsize=10)
    plt.axvline(1.0,linestyle=':',color='k',label='Fiducial $\\Lambda$CDM')
    plt.axhline(f_lcdm(z,Om_fid),linestyle=':',color='k')
    plt.scatter(*soln.x,marker='x',color='k',label='MLE')
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$f$')
    #plt.colorbar(im,label='Negative Log Likelihood')
    plt.colorbar(im,label='Negative Log Posterior')
    plt.legend(frameon=False,loc="upper left")
    plt.savefig(figuresFolder + "likelihood_test_plot_eps.pdf")
    plt.show()
else:
    plt.clf()
    plt.imshow(-log_like_ap.T,
               extent=(Om_range[0],Om_range[-1],f_range[0],f_range[-1]),
               norm=colors.LogNorm(vmin=1e21,vmax=3e22),cmap='Blues',
               aspect='auto',origin='lower')
    plt.xlabel('$\Omega_m$')
    plt.ylabel('$f$')
    plt.colorbar(label='Negative Log Likelihood')
    plt.savefig(figuresFolder + "likelihood_test_plot.pdf")
    plt.show()

plt.clf()
plt.plot(f_range_centres,log_like_ap[20,:])
plt.xlabel('f')
plt.ylabel('Log Likelihood')
plt.title("Likelihood at $\\epsilon = " + ("%.2g" % eps_range[20]) + "$")
plt.savefig(figuresFolder + "nll_plot_f.pdf")
plt.show()


#likelihood at mean value:
log_like_mean = np.zeros(eps_range[0:-1].shape)
for i in tools.progressbar(range(0,len(Om_range_centres))):
    if kwargs['sample_epsilon']:
            theta = np.array([eps_range[i],flat_mean[1],A,r0,c1,f1,B])
    else:
        theta = np.array([Om_range_centres[i],flat_mean[1],
                          A,r0,c1,f1,B])
    log_like_mean[i] = log_likelihood_aptest(theta,*args,**kwargs)

plt.clf()
plt.plot(eps_range[0:-1],log_like_mean)
#plt.axvline(ap_parameter(z,0.0,Om_fid),linestyle=':',color='k',
#    label='$\\Omega_m=0$')
#plt.axvline(ap_parameter(z,1.0,Om_fid),linestyle='--',color='k',
#    label='$\\Omega_m=1.0$')
plt.axvspan(ap_parameter(z,0.0,Om_fid),ap_parameter(z,1.0,Om_fid),alpha=0.5,
            color='k',label='$0 < \\Omega_m(\\epsilon) < 1$',ec=None)
plt.xlabel('$\\epsilon$')
plt.ylabel('Log Likelihood')
plt.title("Likelihood at $f = " + ("%.3g" % flat_mean[1]) + "$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(figuresFolder + "nll_plot_eps.pdf")
plt.show()



# Range of A values:
Om = 0.3111
f = f_lcdm(z,Om)
Arange = np.logspace(-5,1,101)
log_like_ap = np.zeros(Arange.shape)
log_like_ap2 = np.zeros(Arange.shape)

for i in tools.progressbar(range(0,len(Arange))):
    theta = np.array([Om,f,Arange[i]])
    log_like_ap[i] = log_likelihood_aptest(theta,*args,**kwargs)
    #log_like_ap2[i] = log_likelihood_aptest(theta,*args2,**kwargs2)

plt.clf()
plt.loglog(Arange,-log_like_ap,label='With Covariance')
#plt.loglog(Arange,-log_like_ap2,label='Least Squares')
plt.xlabel('A')
plt.ylabel('Negative log likelihood')
plt.title("Likelihood with $\\Omega_m = 0.3111, f = f_{\\mathrm{lcdm}}$")
plt.legend(frameon=False)
plt.savefig(figuresFolder + "A_likelihood_plot.pdf")
plt.show()



Om = soln.x[0]
eps = soln.x[0]
f = soln.x[1]
A = soln.x[2]

f_fid = f_lcdm(0.0225,0.3111)

for i in range(0,len(spar_bins)-1):
    for j in range(0,len(sperp_bins)-1):
        spar = bin_z_centres[i]
        sperp = bin_d_centres[j]
        profile_2d[i,j] = z_space_profile(spar,sperp,
                                          lambda r: rho_real(r,*sol2.x),
                                          z,Om,Delta_func,delta_func,
                                          epsilon=1.0,
                                          f=f_fid,apply_geometry=True)

plot_los_void_stack(\
        profile_2d,bin_d_centres,bin_z_centres,
        cmap='Blues',vmin=0,vmax=0.06,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")

plot_los_void_stack(\
        field_borg,bin_d_centres,bin_z_centres,
        cmap='Blues',vmin=0,vmax=0.06,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_field_borg.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")

plot_los_void_stack(\
        (profile_2d - field_borg)/field_borg,bin_d_centres,bin_z_centres,
        cmap='PuOr_r',
        vmin=0,vmax=200,fontsize=10,norm=colors.Normalize(vmin=-1,vmax=1),
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test_diff.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\Delta\\rho(s_{\\parallel},s_{\\perp})" + \
        "/\\tilde{\\rho}(s_{\\parallel},s_{\\perp})$")


plot_los_void_stack(\
        profile_2d - field_borg,bin_d_centres,bin_z_centres,
        cmap='PuOr_r',
        vmin=0,vmax=200,fontsize=10,norm=colors.Normalize(vmin=-3e-3,vmax=3e-3),
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test_diff_abs.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})" + \
        " - \\tilde{\\rho}(s_{\\parallel},s_{\\perp})$")



plot_los_void_stack(\
        field_lcdm,bin_d_centres,bin_z_centres,
        cmap='Blues',
        vmin=0,vmax=0.06,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_theory_borg.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")




# Fit a multi-parameter model to the profile:



def profile_fit(r,rho0,p,C,rb):
    #return A*(1.0 + k*r + (k*r)**2/(2.0 - k*r0))*np.exp(-k*r)
    return np.exp(np.log(rho0) + (p*C/2)*rb*r - C*r**p)

def log_profile_fit(r,rho0,p,C,rb):
    return np.log(rho0) + (p*C/2)*rb*r - C*r**p

def combine_parameters(theta,fixed):
    free_params = np.isnan(fixed)
    fixed_params = np.logical_not(free_params)
    if np.sum(free_params) != len(theta):
        raise Exception("Parameter count not consistent with free parameters.")
    else:
        theta_new = np.zeros(len(fixed))
        theta_new[free_params] = theta
        theta_new[fixed_params] = fixed[fixed_params]
        return theta_new

# MLE of the profile fit:
def log_likelihood(theta, x, y, yerr,profile_model,fixed=None):
    #rho0,p,C,rb = theta
    #A,r0,c1,f1,B = theta
    #model = profile_broken_power_log(x, A,r0,c1,f1,B)
    if fixed is None:
        fixed = np.array([np.nan for x in theta])
    model = profile_model(x,*combine_parameters(theta,fixed))
    sigma2 = yerr**2
    return -0.5 * np.sum( (y - model)**2/sigma2 + np.log(sigma2) )

# Priors:
def log_prior(theta):
    #A,r0,c1,f1,B = theta
    #alpha,beta,rs,delta_c,delta_large, rv = theta
    beta,delta_c,delta_large, rv = theta
    if (0 <= beta < np.inf) \
        and (-1 < delta_c <= 0) and (-1 <= delta_large <= 1) and (0 <= rv <= 2):
        # Jeffries priors:
        return 0
    else:
        return -np.inf

def log_probability(theta,x,y,yerr,*args,**kwargs):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr,*args,**kwargs)

nll1 = lambda *theta: -log_likelihood(*theta)
initial = np.array([0.1,0.9,-9.5,3.5,0.01])
sol1 = scipy.optimize.minimize(nll1, initial, 
    bounds = [(0,None),(0,2),(None,0),(0,None),(-1,1)],
    args=(r_bin_centres, np.log(rho_r_mean),
    0.5*np.log((rho_r_mean + rho_r_std)/(rho_r_mean - rho_r_std)),
    profile_broken_power_log) )

sol2 = scipy.optimize.minimize(nll1, initial, 
    bounds = [(0,None),(0,2),(None,0),(0,None),(-1,1)],
    args=(r_bin_centres, np.log(rho_r_borg_mean),
    0.5*np.log((rho_r_borg_mean + rho_r_borg_std)/\
    (rho_r_borg_mean - rho_r_borg_std)),
    profile_broken_power_log) )

#A1,r01,c11,f11,B1 = sol1.x
#A2,r02,c12,f12,B2 = sol2.x

#initial = np.array([1.0,1.0,1.0,-0.2,0.0,1.0])
#bounds = [(0,None),(0,None),(0,None),(-1,0),(-1,1),(0,2)]

initial_free = np.array([1.0,-0.85,0.0,1.0])
bounds_free =  [(0,None),(-1,0),(-1,1),(0,2)]
fixed = np.array([np.nan for x in range(0,6)])
fixed[0] = 1
fixed[2] = np.inf

sol1 = scipy.optimize.minimize(nll1, initial_free, 
    bounds = bounds_free,
    args=(rBinStackCentres, delta_lcdm, sigma_delta_lcdm,
    profile_modified_hamaus,fixed) )

sol2 = scipy.optimize.minimize(nll1, initial_free, 
    bounds = bounds_free,
    args=(rBinStackCentres, delta_borg, sigma_delta_borg,
          profile_modified_hamaus,fixed) )

beta,delta_c,delta_large,rv = sol1.x

alpha=1
rs = np.inf

import emcee
pos = sol1.x + 1e-4*np.random.randn(32,4)
nwalkers, ndim = pos.shape

sampler1 = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(rBinStackCentres, delta_lcdm, 
    sigma_delta_lcdm, profile_modified_hamaus,fixed)
)
sampler1.run_mcmc(pos, 10000, progress=True)

tau1 = sampler1.get_autocorr_time(tol=0)

flat_samples1 = sampler1.get_chain(discard=int(3*np.max(tau1)), 
                                 thin=int(np.max(tau1)), flat=True)

#A1, r01, c11,f11,B1 = np.mean(flat_samples1,0)
params = combine_parameters(np.mean(flat_samples1,0),fixed)

sampler2 = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(rBinStackCentres, delta_borg, 
    sigma_delta_borg, profile_modified_hamaus,fixed)
)
sampler2.run_mcmc(pos, 10000, progress=True)
tau2 = sampler2.get_autocorr_time(tol=0)

flat_samples2 = sampler2.get_chain(discard=int(3*np.max(tau2)), 
                                 thin=int(np.max(tau2)), flat=True)


# Test effect on the inference:

nll_cosmo1 = lambda *theta: -log_likelihood_aptest(
    np.hstack([*theta,np.mean(flat_samples1,0)]),*args,**kwargs)
nll_cosmo2 = lambda *theta: -log_likelihood_aptest(
    np.hstack([*theta,np.mean(flat_samples2,0)]),*args,**kwargs)
#nll = lambda *theta: -log_likelihood_aptest(np.hstack([*theta,A_opt]),
#                                            *args,**kwargs)
soln_cosmo1 = scipy.optimize.minimize(nll_cosmo1, 
    np.array([1.0,f_lcdm(z,Om_fid)]),bounds = [(0.9,1.1),(0,1.0)])
soln_cosmo2 = scipy.optimize.minimize(nll_cosmo2, 
    np.array([1.0,f_lcdm(z,Om_fid)]),bounds = [(0.9,1.1),(0,1.0)])
soln_mle2 = scipy.optimize.minimize(nll, 
    np.array([1.0,f_lcdm(z,Om_fid)]),bounds = [(0.9,1.1),(0,1.0)])


# Corner plot:
import corner

plt.clf()
#fig = corner.corner(flat_samples1,labels=["$A$","$r_0$","$c_1$","$f_1$","$B$"])
fig = corner.corner(flat_samples1, labels=["$\\alpha$","$\\beta$","$r_s$",
    "$\\delta_c$","$\\delta_{LS}$","$r_v$"])
fig.suptitle("$\\Lambda$-CDM Simulations Contour Ellipticity")
plt.savefig(figuresFolder + "corner_plot_profile_fit.pdf")
plt.show()



#A = 0.1
#c1 = -9.5
#f1 = 3.5
#r0 = 0.9
#r1 = 1

#rho_r_fit_samples = np.vstack([
#    profile_broken_power(r_bin_centres,A,r0,c1,f1,B) 
#    for A,r0,c1,f1,B in flat_samples1])

#r_values = r_bin_centres
r_values = rBinStackCentres

rho_r_fit_samples = np.vstack([
    profile_modified_hamaus(r_values,*combine_parameters(theta,fixed)) 
    for theta in flat_samples1])

rho_r_fit_samples_mean = np.mean(rho_r_fit_samples,0)
rho_r_fit_samples_std = np.std(rho_r_fit_samples,0)


rho_r_borg_fit_samples = np.vstack([
    profile_modified_hamaus(r_values,*combine_parameters(theta,fixed))
    for theta in flat_samples2])

rho_r_borg_fit_samples_mean = np.mean(rho_r_borg_fit_samples,0)
rho_r_borg_fit_samples_std = np.std(rho_r_borg_fit_samples,0)

#field_borg_fit_samples = rho_r_borg_fit_samples = np.vstack([
#    profile_broken_power(r_values,A,r0,c1,f1,B) 
#    for A,r0,c1,f1,B in flat_samples[:,2:]])

field_borg_fit_samples = rho_r_borg_fit_samples = np.vstack([
    profile_modified_hamaus(r_values,combine_parameters(theta,fixed)) 
    for theta in flat_samples[:,2:]])

field_borg_fit_mean = np.mean(field_borg_fit_samples,0)
field_borg_fit_std = np.std(field_borg_fit_samples,0)

#rho_r_fit_vals = profile_broken_power(r_values,A1,r01,c11,f11,B1)
rho_r_fit_vals = profile_broken_power(r_values,A1,r01,c11,f11,B1)
rho_r_fit_vals_borg = profile_broken_power(r_values,A2,r02,c12,f12,B2)


plt.clf()
fig, ax = plt.subplots(figsize=(textwidth,0.45*textwidth))
plt.errorbar(rBinStackCentres,delta_lcdm,yerr=sigma_delta_lcdm,linestyle='-',
             color=seabornColormap[0],label="$\\Lambda$-CDM profile")
plt.fill_between(r_values,rho_r_fit_samples_mean-rho_r_fit_samples_std,
                 rho_r_fit_samples_mean + rho_r_fit_samples_std,alpha=0.5,
                 color=seabornColormap[0],
                 label="Fitting-formula, $\\Lambda$CDM-profile fit")
plt.errorbar(rBinStackCentres,delta_borg,yerr=sigma_delta_borg,linestyle='-',
             color=seabornColormap[1],label="BORG profile")
plt.fill_between(r_values,
                 rho_r_borg_fit_samples_mean-rho_r_borg_fit_samples_std,
                 rho_r_borg_fit_samples_mean + rho_r_borg_fit_samples_std,
                 alpha=0.5,color=seabornColormap[1],
                 label="Fitting-formula, BORG-profile fit")
#plt.plot(r_bin_centres,rho_r_fit_vals,
#         linestyle=':',color=seabornColormap[0])
#plt.plot(r_bin_centres,rho_r_fit_vals_borg,
#         linestyle=':',color=seabornColormap[1])
#plt.plot(bin_d_centres,field_borg[0],linestyle='-',color='k',
#    label="$\\rho_{\\mathrm{BORG}}(s_{\\perp},0)$")
#plt.fill_between(r_bin_centres,
#                 field_borg_fit_mean - field_borg_fit_std,
#                 field_borg_fit_mean + field_borg_fit_std,
#                 alpha=0.5,color='k',
#                 label="Fitting-formula, $\\rho(s_{\\parallel},s_{\\perp})$ " + 
#                 " fit")
#plt.plot(bin_d_centres,field_lcdm[0],linestyle='--',color='k',
#    label="$\\rho_{\\Lambda\\mathrm{CDM}}(s_{\\perp},0)$")

plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$\\rho(r)$')
plt.xlim([0,10])
plt.ylim([0,1.1])
#plt.yscale('log')
#plt.xscale('log')
plt.legend(frameon=False,loc="lower right")
plt.tight_layout()
plt.savefig(figuresFolder + "profile_fit_test2.pdf")
plt.show()


plt.clf()
fig, ax = plt.subplots(figsize=(textwidth,0.45*textwidth))
plt.errorbar(r_values,(delta_lcdm - rho_r_fit_samples_mean)/\
             rho_r_fit_samples_mean,
             yerr=sigma_delta_lcdm/rho_r_fit_samples_mean,linestyle='-',
             color=seabornColormap[0],
             label="$\\Lambda$-CDM profile")
plt.errorbar(r_values,
             (delta_borg - rho_r_borg_fit_samples_mean)/\
             rho_r_borg_fit_samples_mean,
             yerr=sigma_delta_borg/rho_r_borg_fit_samples_mean,linestyle='-',
             color=seabornColormap[1],label="BORG profile")
plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$\\Delta\\rho(r)/\\rho(r)$')
plt.axhline(0.0,linestyle=':',color='k')
plt.axhspan(-0.01,0.01,color='grey',alpha=0.5,label='1% error')
#plt.yscale('log')
#plt.xscale('log')
plt.ylim([-0.1,0.1])
plt.legend(frameon=False,loc="upper center")
plt.tight_layout()
plt.savefig(figuresFolder + "profile_fit_test_diff.pdf")
plt.show()








