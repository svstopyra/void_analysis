# Loads configuration options, including the locaitons of all the required
# executables, options etc...

# MPI executable (this may need to be changed from cluster to cluster):

# Total number of CPUs. Change for other cluster environments:
NTASKS=${SLURM_NTASKS}

# Paths to relevant void_analysis code:
scriptDir=/cfs/data/stst5617/borg-antihalos
codeDir=/cfs/home/stst5617/software
#postProcessingDir=${codeDir}/void_analysis/postprocessing/
#simulationDir=${codeDir}/void_analysis/simulation/
postProcessingDir=${scriptDir}
simulationDir=${scriptDir}

# Cosmological parameters:
Om=0.3111 # Matter density fraction
ns=0.9665 # Scalar spectral index
Ol=0.6889 # Dark energy density fraction
hubble=0.6766 # Hubble constant, in units of 100 km/s/Mpc
zin=50 # Initial redshift of the simulations
Ob=0.04897468161869667 # Baryonic matter density fraction
sigma8=0.8288 # value of sigma8
# Running options:
resolution=512

# Paths to external codes:
dtfe_exec=${codeDir}/DTFE_source_1.1.1/build/DTFE # DTFE executable
ahf_exec=${codeDir}/ahf/ahf-v1.0-084/bin/AHF-v1.0-084 # AHD executable
voz_dir=${codeDir}/voboz1.3.2/bin/ # Directory containing voz
genetic_dir=${codeDir}/genetIC/genetIC/ # Directory containing genetIC
python_exec=python3 # Python executable
mpiexec=mpirun # MPI executable (this may need to be changed from cluster to cluster):

# Analysis options:
dtfe_res=256
voz_div=4 # Divisions of the simulation box for Voronoi tesselation
voz_ncores=40
voz_buf=0.3
voz_frac=0.1
boxsize=677.7
vorCPUS=16 # Number of CPUs to use for the voronoi tesselation (reduce as very
# memory intensive)

# Name of the subfolders used for forward and reverse simulations:
declare -a subfolders=("gadget_full_forward_${resolution}" "gadget_full_reverse_${resolution}")
# Name of snapshots to be analysed (only change this is also changed in the
# gadget and AHF scripts):
snapname=snapshot_001

# If "constrained" then will try to import white noise from BORG. 
# If "unconstrained", will create unconstrained simulations with the same
# setup:
mode="constrained"





