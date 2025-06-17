#-------------------------------------------------------------------------------
# CONFIGURATION
from void_analysis import plot, tools, snapedit, catalogue
from void_analysis.catalogue import *
from void_analysis.paper_plots_borg_antihalos_generate_data import *
from void_analysis.real_clusters import getClusterSkyPositions
from void_analysis import massconstraintsplot
from void_analysis.simulation_tools import (
    ngPerLBin,
    get_borg_density_estimate
)
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

from void_analysis.simulation_tools import SnapshotGroup

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
                                        r_sphere=150)

# Save catalogue information:
all_cents = cat300.getAllCentres(void_filter=True)
mean_cent = np.nanmean(all_cents,0)
std_cent = np.nanstd(all_cents,0)
all_rad = cat300.getAllProperties("radii",void_filter=True)
cat_rad = np.nanmean(all_rad,1)
cat_rad_err = np.nanstd(all_rad,1)

np.savez("cat300_properties.npz",mean_cent,std_cent,cat_rad,cat_rad_err)

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
# END TO END FUNCTION

# Bins in LOS co-ords:
upper_dist_reff = 2
spar_bins = np.linspace(0,upper_dist_reff,21)
sperp_bins = np.linspace(0,upper_dist_reff,21)
bin_spar_centres = plot.binCentres(spar_bins)
bin_sperp_centres = plot.binCentres(sperp_bins)

# LCDM LOS positions
# Get densities for MCMC simulations, so we can find comparable LCDM regions:
densities_file = data_folder2 + "delta_list.p"
deltaMAPBootstrap, deltaMAPInterval = get_borg_density_estimate(
    borg_snaps,densities_file = densities_file)
# Creat voids list for lcdm:
centres_file = data_folder2 + "random_centres_and_densities.p"
voids_used_lcdm = get_lcdm_void_catalogue(
    lcdm_snaps,deltaMAPInterval,
    centres_file = centres_file)
# Sample of lcdm voids without applying density constraints:
voids_used_lcdm_unconstrained = get_lcdm_void_catalogue(
    lcdm_snaps,centres_file = centres_file)

los_lcdm_zspace = get_los_positions_for_all_catalogues(
    lcdm_snaps["snaps"],lcdm_snaps["snaps_reverse"],lcdm_snaps["void_centres"],
    lcdm_snaps["void_radii"],all_particles=True,
    filter_list=voids_used_lcdm,dist_max=3,
    rmin=10,rmax=20,recompute=False,zspace=True,recompute_zspace=False,
    suffix=".lospos_all_zspace_selected.p",flatten_filters=True)

los_lcdm_zspace_unconstrained = get_los_positions_for_all_catalogues(
    lcdm_snaps["snaps"],lcdm_snaps["snaps_reverse"],lcdm_snaps["void_centres"],
    lcdm_snaps["void_radii"],all_particles=True,
    filter_list=voids_used_lcdm_unconstrained,dist_max=3,
    rmin=10,rmax=20,recompute=False,zspace=True,recompute_zspace=False,
    suffix=".lospos_all_zspace.p",flatten_filters=True)

#los_lcdm_real = get_los_positions_for_all_catalogues(
#    lcdm_snaps["snaps"],lcdm_snaps["snaps_reverse"],
#    lcdm_snaps["void_centres"],lcdm_snaps["void_radii"],all_particles=True,
#    filter_list=voids_used_lcdm,dist_max=3,rmin=10,rmax=20,
#    recompute=False,zspace=False,suffix=".lospos_all.p")

los_lcdm_real_unconstrained = get_los_positions_for_all_catalogues(
    lcdm_snaps["snaps"],lcdm_snaps["snaps_reverse"],
    lcdm_snaps["void_centres"],lcdm_snaps["void_radii"],all_particles=True,
    filter_list=voids_used_lcdm_unconstrained,dist_max=3,rmin=10,rmax=20,
    recompute=False,zspace=False,suffix=".lospos_all.p")


# BORG LOS positions:
halo_indices = cat300.get_final_catalogue(void_filter=True,short_list=False).T
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


#los_borg_real = get_los_positions_for_all_catalogues(
#    borg_snaps["snaps"],borg_snaps["snaps_reverse"],
#    cat300.getAllCentres(void_filter=True),
#    cat300.getAllProperties("radii",void_filter=True).T,all_particles=True,
#    filter_list=filter_list_borg,dist_max=3,rmin=10,rmax=20,
#    void_indices = halo_indices,recompute=False,zspace=False,
#    suffix=".lospos_all.p")

# Trimmed LOS lists:
void_radii_borg = cat300.getMeanProperty("radii",void_filter=True)[0]
void_individual_radii_borg = cat300.getAllProperties("radii",void_filter=True).T

los_list_trimmed_borg, voids_used_borg = trim_los_list(
    los_borg_zspace,spar_bins,sperp_bins,void_individual_radii_borg)

los_list_trimmed_lcdm, voids_used_lcdm = trim_los_list(
    los_lcdm_zspace,spar_bins,sperp_bins,lcdm_snaps["void_radii"])

los_list_trimmed_lcdm_unconstrainted, voids_used_lcdm_unconstrained = \
    trim_los_list(los_lcdm_zspace_unconstrained,spar_bins,sperp_bins,
                  lcdm_snaps["void_radii"])

# Additional weights for BORG, based on reproducibility score:
additional_weights_unfiltered_borg = get_additional_weights_borg(cat300)
additional_weights_borg = get_additional_weights_borg(
    cat300,voids_used = voids_used_borg)

rbins = np.linspace(0,3,31)
recompute_fields = False

# BORG density field:
field_borg_test = tools.loadOrRecompute(data_folder + "borg_field_2d.p",
    get_stacked_void_density_field,
    borg_snaps,cat300.getAllProperties("radii",void_filter=True).T,zcentres,
    spar_bins,sperp_bins,
    additional_weights = additional_weights_unfiltered_borg,
    los_pos = los_borg_zspace,filter_list = filter_list_borg,
    _recomputeData = recompute_fields)

field_borg_1d, field_borg_1d_sigma = \
    tools.loadOrRecompute(data_folder + "borg_field_1d.p",
    get_1d_real_space_field,
    borg_snaps,filter_list=filter_list_borg,
    additional_weights=additional_weights_unfiltered_borg,
    _recomputeData = recompute_fields)


# LCDM density field (with density constraint):
field_lcdm_test = tools.loadOrRecompute(data_folder + "lcdm_field_2d.p",
    get_stacked_void_density_field,
    lcdm_snaps,lcdm_snaps["void_radii"],lcdm_snaps["void_centres"],
    spar_bins,sperp_bins,filter_list=voids_used_lcdm,recompute=False,
    los_pos = los_lcdm_zspace,_recomputeData = recompute_fields)

field_lcdm_1d, field_lcdm_1d_sigma = \
    tools.loadOrRecompute(data_folder + "lcdm_field_1d.p",
    get_1d_real_space_field,lcdm_snaps,filter_list=voids_used_lcdm,
    _recomputeData=recompute_fields)

# LCDM density field (without density constraint):
field_lcdm_uncon = \
    tools.loadOrRecompute(data_folder + "lcdm_field_uncon_2d.p",
    get_stacked_void_density_field,
    lcdm_snaps,lcdm_snaps["void_radii"],lcdm_snaps["void_centres"],
    spar_bins,sperp_bins,filter_list=voids_used_lcdm_unconstrained,
    recompute=False,los_pos = los_lcdm_zspace_unconstrained,
    _recomputeData = recompute_fields)

field_lcdm_uncon_real = \
    tools.loadOrRecompute(data_folder + "lcdm_field_uncon_2d_real.p",
    get_stacked_void_density_field,
    lcdm_snaps,lcdm_snaps["void_radii"],lcdm_snaps["void_centres"],
    spar_bins,sperp_bins,filter_list=voids_used_lcdm_unconstrained,
    recompute=False,los_pos = los_lcdm_real_unconstrained,
    _recomputeData = recompute_fields)

field_lcdm_1d_uncon, field_lcdm_1d_sigma_uncon = \
    tools.loadOrRecompute(data_folder + "lcdm_field_uncon_1d.p",
    get_1d_real_space_field,
    lcdm_snaps,filter_list=voids_used_lcdm_unconstrained,
    _recomputeData = recompute_fields)

# Individual sample fields:
field_lcdm_uncon_all = [
    get_stacked_void_density_field(
        lcdm_snaps,[lcdm_snaps["void_radii"][k]],
        [lcdm_snaps["void_centres"][k]],
        spar_bins,sperp_bins,filter_list=[voids_used_lcdm_unconstrained[k]],
        recompute=False,los_pos = [los_lcdm_zspace_unconstrained[k]],
        _recomputeData = recompute_fields
    )
    for k in range(0,20)
]

field_lcdm_uncon_real_all = [
    get_stacked_void_density_field(
        lcdm_snaps,[lcdm_snaps["void_radii"][k]],
        [lcdm_snaps["void_centres"][k]],
        spar_bins,sperp_bins,filter_list=[voids_used_lcdm_unconstrained[k]],
        recompute=False,los_pos = [los_lcdm_real_unconstrained[k]],
        _recomputeData = recompute_fields
    )
    for k in range(0,20)
]



#-------------------------------------------------------------------------------
# VELOCITY DISTRIBUTION TEST
from void_analysis.tools import get_weighted_samples, weighted_boostrap_error
from void_analysis.tools import get_unit_vector
from void_analysis.cosmology import comoving_distance_to_redshift
from void_analysis.cosmology_inference import get_los_velocities_for_void

# Test plot:
filt = voids_used_lcdm_unconstrained[0]
cell_vols = lcdm_snaps["cell_volumes"][0]
snap = lcdm_snaps["snaps"][0]
radii = lcdm_snaps["void_radii"][0]
centres = lcdm_snaps["void_centres"][0]
n_threads= -1

box_centre = np.array([boxsize/2]*3)
void_radius = radii[filt][0]
void_centre = snapedit.wrap(box_centre - centres[filt,:],boxsize)[0]
tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
r_par, u_par, disp, u = get_los_velocities_for_void(
    void_centre,void_radius,snap,rbins,cell_vols=cell_vols,tree=tree
)
r = np.array(np.sqrt(np.sum(disp**2,1)))
[indices,counts] = plot.binValues(r,rbins*void_radius)
Delta_r = np.cumsum(counts)/(4*np.pi*nbar*rbins[1:]**3*void_radius**3/3) - 1.0
indices_list = np.array(tree.query_ball_point(void_centre,\
            rbins[-1]*void_radius,workers=n_threads),dtype=np.int64)

ur_norm = np.sum(np.array(u * disp),1)/r**2

ur_profile = np.array([
    np.average(ur_norm[ind],weights=cell_vols[indices_list][ind]) 
    for ind in indices
])

ur_profile_error = np.array([
    np.sqrt(
        stacking.weightedVariance(ur_norm[ind],cell_vols[indices_list][ind]) *
        np.sum(cell_vols[indices_list][ind]**2)) 
    for ind in indices
])

from void_analysis.cosmology_inference import plot_all_Deltaf, plot_all_pn
from void_analysis.cosmology_inference import plot_pn_inverses, plot_inverse_r

plot_all_Deltaf(savename=figuresFolder + "Deltaf_plot_all.pdf")
plot_all_pn(savename=figuresFolder + "pn_plot_all.pdf")
plot_pn_inverses(savename=figuresFolder + "pn_inverse_plot.pdf")


# Average over multiple voids:
from void_analysis.cosmology_inference import spherical_lpt_velocity
#trees = [scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize) 
#    for snap in lcdm_snaps["snaps"]
#]

# Use all 20 simulations:
all_ur_and_Delta_profiles = [
    tools.loadOrRecompute(
        snap.filename + ".void_ur_profiles.p",
        get_all_ur_profiles,
        snapedit.wrap(box_centre - centres[filt,:],boxsize),
        radii[filt],rbins,snap,tree=None,cell_vols=cell_vols,
        _recomputeData=False
    )
    for centres, radii, snap, filt, cell_vols in zip(
        lcdm_snaps["void_centres"],lcdm_snaps["void_radii"], 
        lcdm_snaps["snaps"], voids_used_lcdm_unconstrained,
        lcdm_snaps["cell_volumes"]
    )
]

all_ur_profiles = np.vstack([prof[0] for prof in all_ur_and_Delta_profiles])
all_Delta_profiles = np.vstack([prof[1] for prof in all_ur_and_Delta_profiles])

all_void_radii = np.hstack(
    [
        radii[filt] for radii, filt in zip(
            lcdm_snaps["void_radii"],voids_used_lcdm_unconstrained
        )
    ]
)

# Velocities:
rbin_centres = plot.binCentres(rbins)
rbins_physical = np.outer(all_void_radii,rbin_centres)
all_u_profiles = all_ur_profiles*rbins_physical
u_mean = np.mean(all_u_profiles,0)
u_error = np.std(all_u_profiles,0)/np.sqrt(all_u_profiles.shape[0])
Delta_r_profiles = all_Delta_profiles*rbins_physical
Delta_r_mean = np.mean(Delta_r_profiles,0)


ur_mean = np.mean(all_ur_profiles,0)
ur_error = np.std(all_ur_profiles,0)/np.sqrt(all_Delta_profiles.shape[0])
ur_mean_samples = tools.loadOrRecompute(
    data_folder + "ur_samples.p",get_weighted_samples,all_ur_profiles,axis=0,
    _recomputeData = False
)
ur_range = np.percentile(ur_mean_samples,[16,84],axis=1)

Delta_mean = np.mean(all_Delta_profiles,0)
Delta_std = np.std(all_Delta_profiles,0)
Delta_mean_samples = tools.loadOrRecompute(
    data_folder + "Delta_mean_samples.p",get_weighted_samples,
    all_Delta_profiles,axis=0,_recomputeData=False
)
Delta_range = np.percentile(Delta_mean_samples,[16,84],axis=1)

plot_inverse_r(rbin_centres,Delta_mean,
               savename=figuresFolder + "pn_inverse_r_plot.pdf")

plt.clf()
plot_velocity_profiles(rbin_centres,u_mean,Delta_mean,
                       filename = figuresFolder + "u_profiles_average.pdf",
                       ur_range=np.vstack([u_mean-u_error,u_mean + u_error]),
                       ylabel="$u_r [\\mathrm{kms}^{-1}]$",velocity=True,
                       Delta_r = None,normalised=True,
                       fixed_delta=True,perturbative_ics=False,
                       use_linear_on_fail=False,treat_2lpt_separately=False,
                       show_error_estimates=True)
plt.show()


plt.clf()
plot_velocity_profiles(rbin_centres,ur_mean,Delta_mean,
                       filename = figuresFolder + "ur_profiles_average.pdf",
                       ur_range=ur_range,normalised=True,
                       fixed_delta=True,perturbative_ics=False,
                       use_linear_on_fail=False,treat_2lpt_separately=False,
                       show_error_estimates=True,eulerian_radius=True,
                       taylor_expand=True,force_linear_ics=False,
                       expand_denom_only=False,ylim=[0,20])
plt.show()


plt.clf()
plot_velocity_profiles(rbin_centres,ur_mean,Delta_mean,
                       filename = figuresFolder + "ur_difference_plot.pdf",
                       ur_range=ur_range,normalised=True,
                       fixed_delta=True,perturbative_ics=False,
                       use_linear_on_fail=False,treat_2lpt_separately=False,
                       show_error_estimates=False,eulerian_radius=True,
                       taylor_expand=False,force_linear_ics=False,
                       expand_denom_only=False,ylim=[-0.3,0.3],
                       difference_plot=True,
                       ylabel="Fractional Difference in $v_{r}/r$")
plt.show()

plot_psi_true(Delta_mean,savename=figuresFolder + "psi_true_plot.pdf")


#-------------------------------------------------------------------------------
# SEMI-ANALYTIC VELOCITY MODEL

N = 5
u = 1 - np.cbrt(1 + Delta_mean)
z = 0
u_filter = range(2,len(u))
Om = 0.3111

alphas_guesses = [np.ones(len(range(2,N+1))) for N in range(2,10)]
sols = [
    scipy.optimize.least_squares(
        lambda alphas: semi_analytic_model(
            u[u_filter],alphas,z=z,Om=Om,
        ) - ur_mean[u_filter],alphas_guess
    )
    for alphas_guess in alphas_guesses
]
alphas_list = [sol.x for sol in sols]

# Test plot:
plot_velocity_model(
    rbin_centres,Delta_mean,ur_range,ur_mean,alphas_list,u_filter = u_filter,
    savename=figuresFolder + "velocity_model.pdf"
)


alpha_vec = [
    np.array([alphas[n] for alphas in alphas_list if len(alphas) > n]) 
    for n in range(len(alphas_list))
]

plot_alphas(alpha_vec,savename=figuresFolder + "alphas_plot.pdf")

# What would these be in LPT?


#-------------------------------------------------------------------------------
# COVARIANCE MATRIX ESTIMATION:

recompute_covariances = False

v_weights_all_borg = get_void_weights(
    los_list_trimmed_borg,voids_used_borg,void_individual_radii_borg,
    additional_weights = additional_weights_borg)

v_weights_all_lcdm = get_void_weights(los_list_trimmed_lcdm,voids_used_lcdm,
                                       lcdm_snaps["void_radii"])



rho_cov_borg, rhomean_borg = tools.loadOrRecompute(
    data_folder + "borg_cov.p",get_covariance_matrix,
    los_borg_zspace,cat300.getAllProperties("radii",void_filter=True).T,
    spar_bins,sperp_bins,nbar,
    additional_weights=additional_weights_unfiltered_borg,return_mean=True,
    cholesky=False,_recomputeData = recompute_covariances)
cholesky_cov_borg = scipy.linalg.cholesky(rho_cov_borg,lower=True)

logrho_cov_borg, logrho_mean_borg = tools.loadOrRecompute(
    data_folder + "borg_logcov.p",get_covariance_matrix,
    los_borg_zspace,cat300.getAllProperties("radii",void_filter=True).T,
    spar_bins,sperp_bins,nbar,
    additional_weights=additional_weights_unfiltered_borg,return_mean=True,
    cholesky=False,log_field=True,_recomputeData = recompute_covariances)
log_cholesky_cov_borg = scipy.linalg.cholesky(logrho_cov_borg,lower=True)


rho_cov_lcdm, rhomean_lcdm = tools.loadOrRecompute(
    data_folder + "lcdm_cov.p",get_covariance_matrix,
    los_lcdm_zspace_unconstrained,lcdm_snaps["void_radii"],
    spar_bins,sperp_bins,nbar,additional_weights=None,
    return_mean=True,cholesky=False,_recomputeData = recompute_covariances)
cholesky_cov_lcdm = scipy.linalg.cholesky(rho_cov_lcdm,lower=True)

logrho_cov_lcdm, logrho_mean_lcdm = tools.loadOrRecompute(
    data_folder + "lcdm_logcov.p",get_covariance_matrix,
    los_lcdm_zspace_unconstrained,lcdm_snaps["void_radii"],
    spar_bins,sperp_bins,nbar,additional_weights=None,
    return_mean=True,cholesky=False,log_field=True,
    _recomputeData = recompute_covariances)
log_cholesky_cov_lcdm = scipy.linalg.cholesky(logrho_cov_lcdm,lower=True)


#-------------------------------------------------------------------------------
#
#A,r0,c1,f1,B = sol2.x





# Profiles test:

ri = plot_utilities.binCentres(rbins)
ri_lcdm = plot_utilities.binCentres(lcdm_snaps["radius_bins"][0])
ri_borg = plot_utilities.binCentres(borg_snaps["radius_bins"][0])
params_lcdm = get_profile_parameters_fixed(ri_lcdm,field_lcdm_1d_uncon-1,
                                           field_lcdm_1d_sigma_uncon)
params_borg = get_profile_parameters_fixed(ri_borg,field_borg_1d-1,
                                           field_borg_1d_sigma)

ri_upper = lcdm_snaps["radius_bins"][0][1:]

Delta_lcdm = (3/ri_upper**3)*np.cumsum(ri_upper**2*(field_lcdm_1d_uncon - 1)*0.1)


plt.clf()
rrange_lcdm = np.linspace(0,10,1000)
rrange_borg = np.linspace(0,3,1000)
plt.plot(rrange_lcdm,profile_modified_hamaus(rrange_lcdm,*params_lcdm),
         color=seabornColormap[0],label="LCDM model")
plt.fill_between(ri_lcdm,field_lcdm_1d_uncon - 1 - field_lcdm_1d_sigma_uncon,
                 field_lcdm_1d_uncon - 1 + field_lcdm_1d_sigma_uncon,
                 color=seabornColormap[0],label="LCDM simulated",
                 alpha=0.5)
plt.plot(rrange_borg,profile_modified_hamaus(rrange_borg,*params_borg),
         color=seabornColormap[1],label="MCMC model")
plt.fill_between(ri_borg,field_borg_1d - 1 - field_borg_1d_sigma,
                 field_borg_1d - 1 + field_borg_1d_sigma,
                 color=seabornColormap[1],label="MCMC simulated",
                 alpha=0.5)
plt.legend(frameon=False)
plt.xlabel("$r/r_{\\mathrm{eff}}$")
plt.ylabel("\\delta(r)")
plt.savefig(figuresFolder + "profile_fit_test.pdf")
plt.show()


plt.clf()
model_profile = profile_modified_hamaus(ri_lcdm,*params_lcdm)
prof_ratio = (field_lcdm_1d_uncon - 1 - model_profile)/model_profile
prof_low = (field_lcdm_1d_uncon - 1 - field_lcdm_1d_sigma_uncon - model_profile)/model_profile
prof_high = (field_lcdm_1d_uncon - 1 + field_lcdm_1d_sigma_uncon - model_profile)/model_profile
plt.plot(
    ri_lcdm,prof_ratio,color=seabornColormap[0]
)
plt.fill_between(ri_lcdm,prof_low,prof_high,alpha=0.5,color=seabornColormap[0])
plt.xlim([0,np.sqrt(8)])
plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$(\\rho - \\rho_{\mathrm{model}})/\\rho_{\mathrm{model}}$')
plt.savefig(figuresFolder + "1d_profile_ratio.pdf")
plt.show()


# Inference for LCDM mock (end-to-end test):

# Inference for Borg catalogue:
tau, sampler = run_inference_pipeline(
    field_lcdm_uncon,rho_cov_lcdm,rhomean_lcdm,sperp_bins,spar_bins,ri_lcdm,
    field_lcdm_1d_uncon-1.0,field_lcdm_1d_sigma_uncon,log_field=False,
    infer_profile_args=True,tabulate_inverse=True,cholesky=True,
    sample_epsilon=True,filter_data=False,z = 0.0225,lambda_cut=1e-23,
    lambda_ref=1e-27,
    profile_param_ranges = [[0,np.inf],[0,np.inf],[0,np.inf],[-1,0],[-1,1],
    [0,2]],om_ranges = [[0.1,0.5]],eps_ranges = [[0.9,1.1]],f_ranges = [[0,1]],
    Om_fid = 0.3111,filename = "inference_weighted.h5",
    autocorr_filename = "autocorr.npy",disp=1e-2,nwalkers=64,n_mcmc=10000,
    max_n=1000000,batch_size=100,nbatch=100,redo_chain=True,backup_start=True)

sampler = emcee.backends.HDFBackend("inference_weighted.h5")
tau = sampler.get_autocorr_time(tol=0)


# MLE Test:

profile_param_ranges = [[0,np.inf],[0,np.inf],[0,np.inf],[-1,0],[-1,1],[0,2]]
om_ranges = [[0.1,0.5]]
eps_ranges = [[0.9,1.1]]
f_ranges = [[0,1]]
z = 0
Om_fid = 0.3111
eps_initial_guess = np.array([1.0,f_lcdm(z,Om_fid)])
theta_initial_guess = np.array([0.3,f_lcdm(z,0.3)])

vel_param_guess = alphas_list[1]
vel_param_ranges = [[-np.inf,np.inf] for _ in vel_param_guess]
N_vel = len(vel_param_guess)

profile_params = get_profile_parameters_fixed(ri_lcdm, field_lcdm_1d_uncon-1.0, 
                                              field_lcdm_1d_sigma_uncon)


initial_guess_eps = np.hstack([eps_initial_guess,profile_params])
initial_guess_theta = np.hstack([theta_initial_guess,profile_params])
if N_vel > 0:
    initial_guess_eps = np.hstack([initial_guess_eps,vel_param_guess])
    initial_guess_theta = np.hstack([initial_guess_theta,vel_param_guess])

theta_ranges=om_ranges + f_ranges + profile_param_ranges
theta_ranges_epsilon = eps_ranges + f_ranges + profile_param_ranges
if N_vel > 0:
    theta_ranges += vel_param_ranges
    theta_ranges_epsilon += vel_param_ranges


Umap, good_eig = get_nonsingular_subspace(
        rho_cov_lcdm, lambda_reg=1e-27,
        lambda_cut=1e-23, normalised_cov=False,
        mu=rhomean_lcdm)

F_inv = None

data_field = field_lcdm_uncon.flatten()
data_field_real = field_lcdm_uncon_real.flatten()
data_filter = np.ones(data_field.flatten().shape, dtype=bool)
delta = profile_modified_hamaus
Delta = integrated_profile_modified_hamaus
rho_real = lambda *args: delta(*args) + 1.0
cholesky_matrix = scipy.linalg.cholesky(rho_cov_lcdm, lower=True)

scoords = generate_scoord_grid(sperp_bins, spar_bins)

args = (data_field[data_filter],scoords[data_filter,:],
        cholesky_matrix[data_filter,:][:,data_filter],z,Delta,
        delta,rho_real)
kwargs = {'cholesky':True,'tabulate_inverse':True,
          'sample_epsilon':True,'theta_ranges':theta_ranges_epsilon,
          'singular':False,'Umap':Umap,'good_eig':good_eig,'F_inv':None,
          'log_density':False,'infer_profile_args':True,
          'linearise_jacobian':False,
          'vel_model':void_los_velocity_ratio_semi_analytic,
          'dvel_dlogr_model':void_los_velocity_ratio_derivative_semi_analytic,
          'N_vel':N_vel,'data_filter':None,'normalised':False,'ntab':10,
          'Om_fid':0.3,'N_prof':6}
names = ['data_filter','cholesky','normalised','tabulate_inverse',
         'ntab','sample_epsilon','Om_fid','singular','Umap','good_eig',
         'F_inv','log_density','infer_profile_args','N_prof','N_vel']

kwargs_left = {key:kwargs[key] for key in kwargs if key not in names}


# For rapid testing:
theta = initial_guess_eps
data_field, scoords, inv_cov, z, Delta, delta, rho_real = args
[cholesky, tabulate_inverse, sample_epsilon, theta_ranges, singular, Umap, 
 good_eig, F_inv, log_density, infer_profile_args, linearise_jacobian,
 vel_model, dvel_dlogr_model, N_vel,data_filter,
 normalised,ntab,Om_fid,N_prof] = [kwargs[key] for key in kwargs]

vel_params = vel_param_guess
epsilon = 1.0
s_par, s_perp = scoords[:, 0], scoords[:, 1]
s_par_new, s_perp_new = geometry_correction(s_par,s_perp,epsilon)
Delta_func = lambda r: Delta(r, *profile_params)
delta_func = lambda r: delta(r, *profile_params)
rho_func = lambda r: rho_real(r, *profile_params)

z = 0
Om_fid = 0.3111
f1 = f_lcdm(z,Om_fid)

F_inv = get_tabulated_inverse(
    s_par_new,s_perp_new,ntab,Delta_func,f1,vel_params=vel_params,
    **kwargs_left
)

model_field = z_space_profile(
    s_par_new, s_perp_new, rho_func, Delta_func, delta_func,f1=f1,
    z=z, Om=Om, epsilon=epsilon,apply_geometry=False,F_inv=F_inv,
    vel_params=vel_params,**kwargs_left
)

diff_field = (data_field - model_field).reshape(20,20)
diff_field_snr = np.abs(diff_field)/np.sqrt(np.diag(rho_cov_lcdm).reshape(20,20))
diff_field_ratio = diff_field/model_field.reshape(20,20)

# Wrapper allowing us to pass arbitrary arguments that won't be sampled over:
def posterior_wrapper(theta,additional_args,*args,**kwargs):
    theta_comb = np.hstack([theta,additional_args])
    return log_probability_aptest(theta_comb,*args,**kwargs)

nll = lambda theta: -log_likelihood_aptest(theta,*args,**kwargs)
mle_estimate = scipy.optimize.minimize(nll,initial_guess_eps,
                                       bounds=theta_ranges_epsilon)



# Test at fixed epsilon:
f1_range = np.linspace(0,1,101)
log_like_f1 = np.array([nll(np.hstack([[1.0],[f1],profile_params,vel_param_guess]))
                        for f1 in f1_range])

plt.clf()
plt.plot(f1_range,log_like_f1,color=seabornColormap[0],label="log likelihood")
plt.axvline(f_lcdm(0,0.3111),linestyle=":",color='grey',label="$\\Lambda$-CDM value")
plt.xlabel('$f_1$')
plt.ylabel('Log Likelihood')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "f1_plot.pdf")
plt.show()

# Plot difference field:
plt.clf()
im = plt.imshow(
    diff_field_ratio,origin='lower',cmap='PuOr_r',vmin=-0.5,vmax=0.5,
    extent=(0,2,0,2),aspect='auto'
)
plt.xlabel('$s_{\\perp}/R_{\\mathrm{eff}}$')
plt.ylabel('$s_{\\parallel}/R_{\\mathrm{eff}}$')
plt.colorbar(im,label="Fractional Difference from Model")
plt.savefig(figuresFolder + "diff_ratio.pdf")
plt.show()

# Comparison real vs z_space:
spar_centres = plot.binCentres(spar_bins)
sperp_centres = plot.binCentres(sperp_bins)
plot.plot_los_void_stack(\
    field_lcdm_uncon_real,
    sperp_centres,spar_centres,
    cmap='Blues',vmin=0,vmax=1.0,
    xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
    ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$',fontfamily='serif',
    title="Simulated field",colorbar=True,
    colorbar_title = "Fractional Difference",
    savename=figuresFolder + "real_field.pdf"
)


# Difference map:
def plot_difference_map(field,theory_field,sperp_centres,spar_centres,
                        vmin=-0.1,vmax=0.1,filename=None,
                        title="Simulation - Theory Stack"):
    plt.clf()
    diff = field - theory_field.reshape(field.shape)
    im = plot.plot_los_void_stack(\
                diff/theory_field.reshape(field.shape),
                sperp_centres,spar_centres,
                cmap='PuOr_r',vmin=vmin,vmax=vmax,
                xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
                ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$',fontfamily='serif',
                colorbar=True,colorbar_title = "Fractional Difference")
    ax = plt.gca()
    ax.set_title(title)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

rcoords = np.sqrt(np.sum(scoords**2,1))
real_field_estimate = rho_func(rcoords.reshape(20,20))

r_par, r_perp = to_real_space(
    s_par_new, s_perp_new, f1=f1, z=z, Om=Om,Delta=Delta_func,vel_params=vel_params,
    **kwargs_left
)

jacobian = z_space_jacobian(
    Delta_func, delta_func, r_par, r_perp, Om=Om,z=z,f1=f1,vel_params=vel_params,
    **kwargs_left
)

dudr_hz_o1pz = get_dudr_hz_o1pz(
    Delta_func,delta_func,r_par,r_perp,f1,vel_params=vel_params,**kwargs_left
)

jacobian_sims = field_lcdm_uncon/field_lcdm_uncon_real

all_jacobians_sims = [
    field_z/field_r 
    for field_z, field_r in zip(field_lcdm_uncon_all,field_lcdm_uncon_real_all)
]


plt.clf()
plt.plot(r_par.reshape(20,20)[:,0],s_par_new.reshape(20,20)[:,0])
plt.xlabel("$r_{\\parallel}$")
plt.ylabel("$s_{\\parallel}$")
plt.savefig(figuresFolder + "r_par_vs_s_par.pdf")
plt.show()

plt.clf()
plt.plot(
    spar_centres,f1*((2/3)*Delta_func(spar_centres) - delta_func(spar_centres)),
    label='Linear'
)
plt.plot(
    spar_centres,dudr_hz_o1pz.reshape(20,20)[:,0],
    label='Calculated'
)
plt.xlabel("$r_{\\parallel}$")
plt.ylabel("$du_{\\parallel}/dr_{\\parallel}$")
plt.legend(frameon=False)
plt.savefig(figuresFolder + "derivative_factor.pdf")
plt.show()


plt.clf()
plt.plot(
    spar_centres,1/(1 + f1*((2/3)*Delta_func(spar_centres) - delta_func(spar_centres))) - 1,
    label='Linear',linestyle='-',color=seabornColormap[0]
)
plt.plot(
    spar_centres,1/(1 + dudr_hz_o1pz.reshape(20,20)[:,0]) - 1,
    label='Calculated',linestyle='-',color=seabornColormap[1]
)
for k in range(0,20):
    plt.plot(
        spar_centres,all_jacobians_sims[k][:,0] - 1,
        linestyle=':',color=seabornColormap[2]
    )

plt.plot(
    spar_centres,field_lcdm_uncon[:,0]/field_lcdm_uncon_real[:,0] - 1,
    label="Simulations",linestyle='-',color=seabornColormap[2]
)
plt.xlabel("$r_{\\parallel}$")
plt.ylabel("Fractional Difference")
plt.legend(frameon=False)
plt.savefig(figuresFolder + "jacobian_mapping_density.pdf")
plt.show()


# Comparison real vs z_space:
spar_centres = plot.binCentres(spar_bins)
sperp_centres = plot.binCentres(sperp_bins)
plot.plot_los_void_stack(\
    field_lcdm_uncon_real,
    sperp_centres,spar_centres,
    cmap='Blues',vmin=0,vmax=1.0,
    xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
    ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$',fontfamily='serif',
    title="Simulated field",colorbar=True,
    colorbar_title = "Fractional Difference",
    savename=figuresFolder + "real_field.pdf"
)
plot.plot_los_void_stack(\
    real_field_estimate,
    sperp_centres,spar_centres,
    cmap='Blues',vmin=0,vmax=1.0,
    xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
    ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$',fontfamily='serif',
    title="Simulated field",colorbar=True,
    colorbar_title = "Fractional Difference",
    savename=figuresFolder + "real_field_estimate.pdf"
)

plot_difference_map(
    field_lcdm_uncon_real,real_field_estimate,sperp_centres,spar_centres,
    vmin=-0.1,vmax=0.1,filename=figuresFolder + "real_field_difference.pdf",
    title="Stacked - Expected"
)

plot_difference_map(
    field_lcdm_uncon,field_lcdm_uncon_real,sperp_centres,spar_centres,
    vmin=-0.4,vmax=0.4,filename=figuresFolder + "zspace_vs_real_field.pdf",
    title="Redshift-space - Real-space"
)

diff_ratio_sims = (field_lcdm_uncon - field_lcdm_uncon_real)/field_lcdm_uncon_real
diff_ratio_theory = (model_field.reshape(20,20) - real_field_estimate)/real_field_estimate

plot_difference_map(
    model_field.reshape(20,20),real_field_estimate,sperp_centres,spar_centres,
    vmin=-0.4,vmax=0.4,filename=figuresFolder + "zspace_vs_real_field_theory.pdf",
    title="Redshift-space - Real-space (theory)"
)

plot.plot_los_void_stack(\
    jacobian.reshape(20,20),
    sperp_centres,spar_centres,
    cmap='Blues',vmin=0.8,vmax=1.2,
    xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
    ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$',fontfamily='serif',
    title="Simulated field",colorbar=True,
    colorbar_title = "Jacobian Factor",
    savename=figuresFolder + "jacobian.pdf"
)
plot.plot_los_void_stack(\
    jacobian_sims,
    sperp_centres,spar_centres,
    cmap='Blues',vmin=0.8,vmax=1.2,
    xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
    ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$',fontfamily='serif',
    title="Simulated field",colorbar=True,
    colorbar_title = "Jacobian Factor",
    savename=figuresFolder + "jacobian_sims.pdf"
)

# All jacobians:
plt.clf()
nrows = 4
ncols = 5
fig, ax = plt.subplots(nrows,ncols)
for k in range(0,20):
    i = int(np.floor(k/ncols))
    j = k - ncols*i
    axij = ax[i,j]
    im = plot.plot_los_void_stack(
        all_jacobians_sims[k],
        sperp_centres,spar_centres,ax=axij,
        cmap='Blues',vmin=0.8,vmax=1.2,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$',fontfamily='serif',
        title=None,colorbar=False,
        colorbar_title = "Jacobian Factor",
    )
    if j > 0:
        axij.yaxis.label.set_visible(False)
        axij.yaxis.set_major_formatter(NullFormatter())
        axij.yaxis.set_minor_formatter(NullFormatter())
    if i < nrows - 1:
        axij.xaxis.label.set_visible(False)
        axij.xaxis.set_major_formatter(NullFormatter())
        axij.xaxis.set_minor_formatter(NullFormatter())

plt.subplots_adjust(wspace=0.0,hspace=0.025,top=0.97,bottom=0.1)
fig.colorbar(
    im, ax=ax.ravel().tolist(),shrink=0.9,
    label='Jacobian factor'
)
plt.suptitle("Jacobians for 20 LCDM simulations")
plt.savefig(figuresFolder + "all_jacobians.pdf")
plt.show()

plt.clf()
rrange_lcdm = np.linspace(0,10,1000)
smax = np.sqrt(np.max(spar_bins)**2 + np.max(sperp_bins)**2)
fig, ax = plt.subplots(figsize=(textwidth,0.45*textwidth))
plt.plot(rrange_lcdm,
         profile_modified_hamaus(rrange_lcdm,*mle_estimate.x[2:]),
         label="MLE Profile (joint inference)")
plt.plot(rrange_lcdm,
         profile_modified_hamaus(rrange_lcdm,*profile_params),
         label="MLE Profile (Separate inference)")
plt.fill_between(ri_lcdm,field_lcdm_1d_uncon - 1 - field_lcdm_1d_sigma_uncon,
                 field_lcdm_1d_uncon - 1 + field_lcdm_1d_sigma_uncon,
                 color=seabornColormap[0],label="LCDM simulated",
                 alpha=0.5)
plt.axvline(smax,label="2D Inference Max r",linestyle=':',color='grey')
plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$\\delta(r)$')
plt.xlim([0,10])
plt.ylim([-1,0.1])
#plt.yscale('log')
#plt.xscale('log')
plt.legend(frameon=False,loc="lower right")
plt.tight_layout()
plt.savefig(figuresFolder + "profile_fit_test_mle.pdf")
plt.show()


# Relative error:
plt.clf()
rrange_lcdm = np.linspace(0,10,1000)
smax = np.sqrt(np.max(spar_bins)**2 + np.max(sperp_bins)**2)
fig, ax = plt.subplots(figsize=(textwidth,0.45*textwidth))
plt.plot(ri_lcdm,
         profile_modified_hamaus(ri_lcdm,*mle_estimate.x[2:]) - 
         field_lcdm_1d_uncon + 1,
         label="MLE Profile (joint inference)")
plt.plot(ri_lcdm,
         profile_modified_hamaus(ri_lcdm,*profile_params) - 
         field_lcdm_1d_uncon + 1,
         label="MLE Profile (Separate inference)")
plt.fill_between(ri_lcdm,-field_lcdm_1d_sigma_uncon,field_lcdm_1d_sigma_uncon,
                 color=seabornColormap[0],label="LCDM simulation uncertainty",
                 alpha=0.5)
plt.axvline(smax,label="2D Inference Max r",linestyle=':',color='grey')
plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$\\delta_{\\mathrm{model}}(r) - \\delta_{\\mathrm{data}}$')
plt.xlim([0,10])
plt.ylim([-0.05,0.05])
plt.legend(frameon=False,loc="upper right")
plt.tight_layout()
plt.savefig(figuresFolder + "profile_fit_test_mle_relative.pdf")
plt.show()


# Test profile function integration?

r_range = np.logspace(-3,1,1001)
del_func = lambda r: profile_modified_hamaus(r,*profile_params)
Del_func = lambda r: integrated_profile_modified_hamaus(r,*profile_params)
Del_val_true = np.array([scipy.integrate.quad(
                            lambda r: del_func(r)*r**2,0,r)[0] * (3/(r**3))
                         for r in r_range])
Del_val_test = Del_func(r_range)

plt.clf()
plt.plot(r_range,Del_val_true,label="Numerical Integration")
plt.plot(r_range,Del_val_test,label="Analytic Integration")
plt.legend(frameon=False)
plt.savefig(figuresFolder + "integrated_profile_test.pdf")
plt.show()


plt.clf()
plt.plot(r_range,Del_val_test - Del_val_true,label="Analytic - Numerical")
plt.legend(frameon=False)
plt.savefig(figuresFolder + "integrated_profile_test_diff.pdf")
plt.show()



# Likelihood plot:


f_range = np.linspace(0,1,41)
f_range_centres = plot.binCentres(f_range)
eps_range = np.linspace(0.9,1.1,41)
eps_centres = plot.binCentres(eps_range)




#log_like_ap_joint = np.zeros((40,40))
log_like_ap_sep = np.zeros((40,40))
#params_joint = mle_estimate.x[2:]
params_sep = np.hstack([profile_params,vel_param_guess])
for i in tools.progressbar(range(0,len(eps_centres))):
    for j in range(0,len(f_range_centres)):
        #theta_joint = np.array([eps_range[i],f_range_centres[j],*params_joint])
        theta_sep = np.array([eps_range[i],f_range_centres[j],*params_sep])
        #log_like_ap_joint[i,j] = log_likelihood_aptest(theta_joint,*args,**kwargs)
        log_like_ap_sep[i,j] = log_likelihood_aptest(theta_sep,*args,**kwargs)
        #log_like_ap[i,j] = log_probability_aptest(theta,*args,**kwargs)

filename = figuresFolder + "likelihood_test_plot_eps.pdf"

def plot_likelihood_distribution(
        log_like,eps_range,f_range,vmin=100,vmax=25000,
        levels = [100,500,1000,2000,5000,10000,15000,20000],
        f_fid=None,eps_fid=1.0,mle=None,filename=None,Om_fid = 0.3111,
        z = 0.0225):
    plt.clf()
    #im = plt.imshow(-log_like_ap.T,
    #           extent=(eps_range[0],eps_range[-1],f_range[0],f_range[-1]),
    #           norm=colors.LogNorm(vmin=1e9,vmax=1e10),cmap='Blues',
    #           aspect='auto',origin='lower')
    im = plt.imshow(-log_like.T,
               extent=(eps_range[0],eps_range[-1],f_range[0],f_range[-1]),
               norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap='Blues',
               aspect='auto',origin='lower')
    f_range_centres = plot.binCentres(f_range)
    eps_centres = plot.binCentres(eps_range)
    X, Y = np.meshgrid(eps_centres,f_range_centres)
    CS = plt.contour(X,Y,-log_like.T,levels = levels)
    plt.clabel(CS, inline=True, fontsize=10)
    plt.axvline(eps_fid,linestyle=':',color='k',label='Fiducial $\\Lambda$CDM')
    if f_fid is None:
        f_fid = f_lcdm(z,Om_fid)
    plt.axhline(f_fid,linestyle=':',color='k')
    if mle is not None:
        plt.scatter(mle[0],mle[1],marker='x',color='k',label='MLE')
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$f$')
    plt.colorbar(im,label='Negative Log Likelihood')
    #plt.colorbar(im,label='Negative Log Posterior')
    plt.legend(frameon=False,loc="upper left")
    if filename is not None:
        plt.savefig(filename)
    plt.show()

plot_likelihood_distribution(log_like_ap_joint,eps_range,f_range,
                             filename = figuresFolder + "likelihood_joint.pdf",
                             mle = mle_estimate.x)
plot_likelihood_distribution(log_like_ap_sep,eps_range,f_range,
                             filename = figuresFolder + "likelihood_sep.pdf",
                             mle = None)

plt.clf()
plt.plot(f_range_centres,log_like_ap[20,:])
plt.xlabel('f')
plt.ylabel('Log Likelihood')
plt.title("Likelihood at $\\epsilon = " + ("%.2g" % eps_range[20]) + "$")
plt.savefig(figuresFolder + "nll_plot_f.pdf")
plt.show()

def get_theory_field(theta,spar_bins,sperp_bins,
                     Delta=integrated_profile_modified_hamaus,
                     delta=profile_modified_hamaus,ntab=10,
                     z=0,Om = 0.3111,N_prof = 6,**kwargs):
    # Parse arguments:
    epsilon, f1 = theta[0:2]
    profile_params = theta[2:(2 + N_prof)]
    N_vel = len(theta) - N_prof - 2
    vel_params = None if N_vel <= 0 else theta[(2 + N_prof):(2 + N_prof + N_vel)]
    # Co-ordinate grid:
    scoords = generate_scoord_grid(sperp_bins, spar_bins)
    s_par = scoords[:,0]
    s_perp = scoords[:,1]
    # Geometry correction:
    s_par_new, s_perp_new = geometry_correction(s_par,s_perp,epsilon)
    # Density profile functions:
    Delta_func = lambda r: Delta(r,*profile_args)
    delta_func = lambda r: delta(r,*profile_args)
    rho_real = lambda r: delta_func(r) + 1.0
    # Precomputed inverse function:
    F_inv = get_tabulated_inverse(
        s_par_new,s_perp_new,ntab,Delta_func,f1,vel_params=vel_params,
        **kwargs
    )
    theory_val = z_space_profile(
        s_par_new, s_perp_new, rho_real, Delta_func, delta_func,f1=f1,
        z=z, Om=Om, epsilon=epsilon,apply_geometry=False,F_inv=F_inv,
        vel_params=vel_params,**kwargs
    )
    return theory_val

def get_diff_vector(N,i):
    base = np.zeros(N)
    base[i] = 1
    return base

def get_diffs(theta,spar_bins,sperp_bins,step=0.01,**kwargs):
    N = len(theta)
    diff_vectors = np.array([get_diff_vector(N,i)*step for i in range(N)])
    upp_fields = [get_theory_field(theta + vector,spar_bins,sperp_bins,
                                   **kwargs) for vector in diff_vectors]
    low_fields = [get_theory_field(theta - vector,spar_bins,sperp_bins,
                                   **kwargs) for vector in diff_vectors]
    derivatives = [(upp - low) / ( 2 * step ) 
                   for upp, low in zip(upp_fields, low_fields)]
    return derivatives

derivs = get_diffs(mle_estimate.x,spar_bins,sperp_bins)
theory_field = get_theory_field(mle_estimate.x,spar_bins,sperp_bins)

# Plot of derivatives:
plt.clf()
nrows = 2
ncols = 4
param_labels = ["$\\epsilon$","$f$","$\\alpha$","$\\beta$","$r_s$",
                   "$\\delta_c$","$\\delta_{\\mathrm{large}}$","$r_v$"]
spar_centres = plot.binCentres(spar_bins)
sperp_centres = plot.binCentres(sperp_bins)
fig, ax = plt.subplots(nrows,ncols,figsize=(textwidth,0.5*textwidth))
for i in range(nrows):
    for j in range(ncols):
        k = i*ncols + j
        axij = ax[i,j]
        im = plot.plot_los_void_stack(\
            derivs[k].reshape((20,20))/theory_field.reshape((20,20)),
            sperp_centres,spar_centres,ax=axij,
            cmap='PuOr_r',vmin=-1,vmax=1,
            xlabel = '$d/R_{\\mathrm{eff}}$',
            ylabel = '$z/R_{\\mathrm{eff}}$',fontfamily='serif',
            title=param_labels[k])
        if j > 0:
            axij.yaxis.label.set_visible(False)
            axij.yaxis.set_major_formatter(NullFormatter())
            axij.yaxis.set_minor_formatter(NullFormatter())
        if i < nrows - 1:
            axij.xaxis.label.set_visible(False)
            axij.xaxis.set_major_formatter(NullFormatter())
            axij.xaxis.set_minor_formatter(NullFormatter())

plt.subplots_adjust(wspace=0.0,hspace=0.025,top=0.97,bottom=0.1)
fig.colorbar(im, ax=ax.ravel().tolist(),shrink=0.9,
    label='Fractional derivative, $(\\mathrm{d}\\rho/\\mathrm{d}\\theta)/\\rho$')
plt.savefig(figuresFolder + "derivatives_map.pdf")
plt.show()


theory_field_init = get_theory_field(initial_guess_eps,spar_bins,sperp_bins)
filename = figuresFolder + "theory_vs_simulation_difference_map.pdf"




plot_difference_map(
    field_lcdm_uncon,theory_field_init,sperp_centres,spar_centres,
    vmin=-1,vmax=1,
    filename = figuresFolder + "theory_vs_simulation_initial.pdf")

plot_difference_map(
    field_lcdm_uncon,theory_field,sperp_centres,spar_centres,
    vmin=-0.1,vmax=0.1,
    filename = figuresFolder + "theory_vs_simulation_mle.pdf")

plot.plot_los_void_stack(\
        theory_field_init.reshape((20,20)),
        sperp_centres,spar_centres,
        cmap='Blues',vmin=0,vmax=1.0,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$',fontfamily='serif',
        title="Theory Field, initial guess",colorbar=True,
        colorbar_title = "Fractional Difference",
        savename=figuresFolder + "initial_theory_field.pdf")

plot.plot_los_void_stack(\
        field_lcdm_uncon,
        sperp_centres,spar_centres,
        cmap='Blues',vmin=0,vmax=1.0,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$',fontfamily='serif',
        title="Simulated field",colorbar=True,
        colorbar_title = "Fractional Difference",
        savename=figuresFolder + "initial_simulation_field.pdf")

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
#chain = sampler.get_chain(discard=2000)
tau_max = np.max(tau)
chain = sampler.get_chain()

jump = 100
intervals = np.arange(0,chain.shape[0],jump)
taus = np.zeros((len(intervals)-1,chain.shape[1],chain.shape[2]))
for k in tools.progressbar(range(0,chain.shape[1])):
    for l in range(0,len(intervals)-1):
        taus[l,k,:] = emcee.autocorr.integrated_time(chain[0:intervals[l+1],
                                                     [k],:],tol=0)


param_labels = ["$\\epsilon$","$f$","$\\alpha$","$\\beta$","$r_s$",
                   "$\\delta_c$","$\\delta_{\\mathrm{large}}$","$r_v$"]

# Taus plot:
plt.clf()
plt.plot(intervals[1:],np.mean(taus,1),
         label = param_labels)
plt.xlabel("MCMC Steps")
plt.ylabel("Correlation Length")
plt.legend(frameon=False)
plt.savefig(figuresFolder + "taus.pdf")
plt.show()

std_chain = np.std(chain,0)

mean_chain = np.mean(chain,1)
tau_mean = emcee.autocorr.integrated_time(
    mean_chain.reshape(chain.shape[0],1,chain.shape[2]),tol=0)

# Chains plot:
plt.clf()
#plt.plot(np.sign(mean_chain)*np.log10(1 + np.abs(mean_chain)))
plt.plot(np.mean(chain,1),label=param_labels)
#plt.ylim([-5,5])
plt.xlabel("MCMC step")
plt.ylabel("Parameter")
plt.legend(frameon=False)
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
Om = lcdm_snaps["snaps"][0].properties["omegaM0"]
f_truth = f_lcdm(0,Om)
eps_truth = 1.0
#fig = corner.corner(flat_samples, labels=["$\\Omega_{m}$","$f$","A"])
#fig = corner.corner(flat_samples, labels=["$\\epsilon$","$f$","$A$","$r_0$",
#    "$c_1$","$f_1$","$B$"])
param_ranges = [[0.9,1.1],[0,1.0]]
#param_ranges = None
fig = corner.corner(flat_samples[:,0:2],labels=["$\\epsilon$","$f$"],
                    range=param_ranges,truths=[eps_truth,f_truth],
                    truth_color="grey",)
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
        spar = bin_spar_centres[i]
        sperp = bin_sperp_centres[j]
        profile_2d[i,j] = z_space_profile(spar,sperp,
                                          lambda r: rho_real(r,*sol2.x),
                                          z,Om,Delta_func,delta_func,
                                          epsilon=1.0,
                                          f=f_fid,apply_geometry=True)

plot_los_void_stack(\
        profile_2d,bin_sperp_centres,bin_spar_centres,
        cmap='Blues',vmin=0,vmax=0.06,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")

plot_los_void_stack(\
        field_borg,bin_sperp_centres,bin_spar_centres,
        cmap='Blues',vmin=0,vmax=0.06,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_field_borg.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")

plot_los_void_stack(\
        (profile_2d - field_borg)/field_borg,bin_sperp_centres,bin_spar_centres,
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
        profile_2d - field_borg,bin_sperp_centres,bin_spar_centres,
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
        field_lcdm,bin_sperp_centres,bin_spar_centres,
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
#plt.plot(bin_sperp_centres,field_borg[0],linestyle='-',color='k',
#    label="$\\rho_{\\mathrm{BORG}}(s_{\\perp},0)$")
#plt.fill_between(r_bin_centres,
#                 field_borg_fit_mean - field_borg_fit_std,
#                 field_borg_fit_mean + field_borg_fit_std,
#                 alpha=0.5,color='k',
#                 label="Fitting-formula, $\\rho(s_{\\parallel},s_{\\perp})$ " + 
#                 " fit")
#plt.plot(bin_sperp_centres,field_lcdm[0],linestyle='--',color='k',
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









