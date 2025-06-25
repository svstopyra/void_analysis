from void_analysis.simulation_tools import SnapshotGroup
from void_analysis import catalogue
import os
import pynbody

# Path to where the test snapshots are stored:
base_path = "test_snaps/"

# Forward simulations:
snapList = [
    pynbody.load(
        os.path.join(base_path,f"sample{k}/forward/snapshot_001")
    )
    for k in range(1,4)
]
# Reverse simulations:
snapListRev = [
    pynbody.load(
        os.path.join(base_path,f"sample{k}/reverse/snapshot_001")
    )
    for k in range(1,4)
]

# Class that handles groups of related snapshots cleanly:
snaps = SnapshotGroup(
    snapList,snapListRev,low_memory_mode=False,swapXZ  = False,
    reverse = True,remap_centres=True
)

# Construction of an anti-halo void catalogue:
# Parameters:
rSphere = 25 # Radius out to which to search for voids
muOpt = 0.2 # Optimal choice of \mu_R, radius ratio
rSearchOpt = 3 # Optimal choice of \mu_S, search radius ratio
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
m_unit = snaps.snaps[0]['mass'][0]*1e10

cat = catalogue.combinedCatalogue(
    snaps.snap_filenames,snaps.snap_reverse_filenames,\
    muOpt,rSearchOpt,rSphere,\
    ahProps=snaps.all_property_lists,hrList=snaps["antihalos"],\
    max_index=None,\
    twoWayOnly=True,blockDuplicates=True,\
    massRange = [mMin,mMax],\
    NWayMatch = NWayMatch,r_min=rMin,r_max=rMax,\
    additionalFilters = None,verbose=False,\
    refineCentres=refineCentres,sortBy=sortBy,\
    enforceExclusive=enforceExclusive
)
# Build the anti-halo catalogue
finalCat = cat.constructAntihaloCatalogue()

# Properties of the antihalo catalogue:
radii = cat.getMeanProperty("radii")
mass = cat.getMeanProperty("mass")
centres = cat.getMeanCentres()

# Properties of all the constituent halos:
all_radii = cat.getAllProperties("radii")
all_masses = cat.getAllProperties("mass")
all_centres = cat.getAllCentres()

# All halo indices:
halo_indices = cat.get_final_catalogue(short_list=False)
