import argparse
import borg
import os
import numpy as np

from void_analysis.tools import mcmcFileToWhiteNoise

# Process arguments:
parser = argparse.ArgumentParser(description = "Generates initial " + \
    "conditions from a white noise field.")
parser.add_argument("outname",help = "Output file name for initial conditions.")
parser.add_argument("--wn_file",help = "White noise file to use " + \
    "(Default: generate from seed).")
parser.add_argument("--transfer_file",help = "Transfer function file to use "+ \
    "(default: generates one from cosmological parameters)")
parser.add_argument("--Ob",help = "Baryon fraction today.",type=float,\
    default=0.04897468161869667)
parser.add_argument("--Om",help = "Matter fraction today.",type=float,\
    default=0.307)
parser.add_argument("--Ol",help = "Dark Energy fraction today.",type=float,\
    default=0.693)
parser.add_argument("--s8",help = "Sigma8.",type=float,\
    default=0.8288)
parser.add_argument("--ns",help = "Power spectrum spectral parameter.",\
    type=float,default=0.9611)
parser.add_argument("--hubble",help = "Hubble rate in 100 km/s/Mpc.",\
    type=float,default=0.705)
parser.add_argument("--zin",help = "Starting redshift.",\
    type=float,default=69.0)
parser.add_argument("--kmin",help = "Minimum wave-number (Mpc/h).",\
    type=float,default=0.001)
parser.add_argument("--kmax",help = "Maximum wave-number (Mpc/h).",\
    type=float,default=10)
parser.add_argument("--ksteps",help = "Number of steps in k-range.",\
    type=int,default=300)
parser.add_argument("--seed",help = "Seed to use in genetIC.",\
    type=int,default=0)
parser.add_argument("--boxsize",help = "Size of simulation box in Mpc/h",\
    type=float,default=677.7)
parser.add_argument("--Nres",help="Resolution along one " + \
    "side of the simulation box.",type=int,default=256)
parser.add_argument("--ssfactor",help="Super-sample factor " + \
    "(default: no supersampling).",type=int,default=1)
parser.add_argument("--dm_only",help = "If True (default) generates only " + \
    "dark matter, no baryons (default: True)",default=True,type=bool)
parser.add_argument("--renormalise_noise",type=bool,default=True,\
    help="If true, renormalise white noise to have unit variance" + \
    " (only applicable when extracting white noise from h5 file).")
parser.add_argument("--inverse_fourier",type=bool,default=False,\
    help="If true, obtain the white noise by inverting the real Fourier " + \
    "transform of the white noise field.")
parser.add_argument("--flip",type=bool,default=False,\
    help="If true, flip the Fourier sign convention of the white noise field.")
parser.add_argument("--reverse",type=bool,default=False,\
    help="If true, flip the sign of the white noise field.")
parser.add_argument("--generate_reversed",type=bool,default=False,\
    help="Generate a reversed initial condition alongside.")
parser.add_argument("--genetic_dir",type=str,default="",\
    help = "Directory in which the genetIC code binary is found.")
parser.add_argument("--seed_const",type=int,default=43940139,\
    help='Added to sample number to create predictable random seeds.')
parser.add_argument("--sample",type=int,default=0,\
    help="Used to seed the seed generator for white noise.")
parser.add_argument("--baseRes",type=int,default=256,\
    help="Base Resolution before super-sampling")



# Generate a transfer function file, if none exists:
args = parser.parse_args()
#SBATCH --node

# Seed:
if args.seed == 0:
    # Generate a random seed.
    np.random.seed(args.seed_const + args.sample) # Force predictable seed
    args.seed = np.random.randint(99999999)

# Calculate super-sampling:
if args.Nres != args.baseRes:
    if (args.Nres > args.baseRes) and (args.Nres % args.baseRes != 0):
        raise Exception("Supersampled resolution must be a multiple of " + \
            str(args.baseRes) + ".")
    elif (args.Nres < args.baseRes) and (args.baseRes % args.Nres != 0):
        raise Exception("Subsampled resolution must be a factor of " + \
            str(args.baseRes) + ".")
    if args.Nres > args.baseRes:
        args.ssfactor = int(args.Nres/args.baseRes)
    else:
        args.ssfactor = int(args.baseRes/args.Nres)


if args.transfer_file is None:
    # Generate one from existing cosmology:
    cosmo_par = borg.cosmo.CosmologicalParameters()
    cosmo_par.default()
    # choose cosmological parameters to match original reconstruction.
    cosmo_par.omega_m = args.Om
    cosmo_par.omega_k = 0.0
    cosmo_par.omega_r = 0.0
    cosmo_par.omega_b = args.Ob
    cosmo_par.omega_q = args.Ol
    cosmo_par.n_s = args.ns
    cosmo_par.sigma8 = args.s8
    cosmo_par.h = args.hubble
    krange = np.exp(np.linspace(np.log(args.kmin),np.log(args.kmax),\
        args.ksteps))
    #psBORGCalc = borg.cosmo.CosmoPower(cosmo_par).power(krange)
    psBORGCalc = borg.cosmo.CosmoPower(cosmo_par,z=args.zin).power(krange)
    transferBORG = np.sqrt(psBORGCalc/(krange**args.ns))
    transferBORG /= transferBORG[0] # Normalise
    # Write to file:
    tf = open(os.path.dirname(args.outname) + "/transfer_file.txt","w")
    for k in range(0,args.ksteps):
        tf.write(str(krange[k]) + "\t" + \
            '\t'.join([str(transferBORG[k]) for l in range(0,6)]) + '\n')
    tf.close()
    args.transfer_file = "transfer_file.txt"

# Generate the GenetIC Parameter file:
genetICParamFile = "# cosmology:\n"
if args.dm_only:
    genetICParamFile += "Ob\t0.0\n"
else:
    genetICParamFile += "Ob\t" + str(args.Ob) + "\n"
genetICParamFile += "Om\t" + str(args.Om) + "\n"
genetICParamFile += "Ol\t" + str(args.Ol) + "\n"
genetICParamFile += "s8\t" + str(args.s8) + "\n"
genetICParamFile += "ns\t" + str(args.ns) + "\n"
genetICParamFile += "hubble\t" + str(args.hubble) + "\n"
genetICParamFile += "zin\t" + str(args.zin) + "\n"
genetICParamFile += "camb\t" + str(args.transfer_file) + "\n"
genetICParamFile += "random_seed_real_space\t" + str(args.seed) + "\n"

# Body of script:
genetICParamFile += "\n\n# Output:\n"
genetICParamFile += "outname " + os.path.basename(args.outname) + "\n"
genetICParamFile += "outdir\t./\n"
genetICParamFile += "outformat 2\n"

# Grid size:
genetICParamFile += "base_grid " + str(args.boxsize) + " " + \
    str(args.baseRes) + "\n"
if args.wn_file is not None:
    # Check the wn file:
    fname, ext = os.path.splitext(args.wn_file)
    if ext == '.h5':
        # Extract the white noise from the h5py file:
        if os.path.dirname(args.wn_file) == '':
            wnFileName = "wn"
        else:
            wnFileName = os.path.dirname(args.wn_file) + "/wn"
        mcmcFileToWhiteNoise(args.wn_file,wnFileName,\
            normalise = args.renormalise_noise,\
            fromInverseFourier=args.inverse_fourier,flip=args.flip,\
            reverse=args.reverse)
        wnFileName += ".npy"
    elif ext == '.npy':
        wnFileName = args.wn_file
    else:
        raise Exception("Invalid White noise file.")
    genetICParamFile += "import_noise 0 " + wnFileName + "\n"
else:
    print("WARNING: no white noise file. Generating random initial conditions.")

if args.ssfactor != 1:
    if args.Nres > args.baseRes:
        genetICParamFile += "supersample " + str(args.ssfactor) + "\n"
    elif args.Nres < args.baseRes:
        genetICParamFile += "subsample " + str(args.ssfactor) + "\n"


genetICParamFileBase = genetICParamFile
genetICParamFile += "\ndone\n"

if args.genetic_dir == "":
    genetICPath = "genetIC"
else:
    genetICPath = args.genetic_dir + "/genetIC"

# Print the parameter file:
genetICFilePath = os.path.dirname(args.outname) + "/genetic_parameters.txt"
genetICFile = open(genetICFilePath,"w")
genetICFile.write(genetICParamFile)
genetICFile.close()

# Now generate the initial conditions:
os.system(genetICPath + " " + genetICFilePath)

if args.generate_reversed:
    # Move old initial conditions:
    fname, ext = os.path.splitext(args.outname)
    ext = ".gadget2"
    os.system("mv " + args.outname + ext + " " + fname + "_for" + ext)
    genetICParamFileBase += "\nreverse\n"
    genetICParamFileBase += "\ndone\n"
    reverseFilePath = os.path.dirname(args.outname) + \
        "/genetic_parameters_reverse.txt"
    genetICFile = open(reverseFilePath,"w")
    genetICFile.write(genetICParamFileBase)
    genetICFile.close()
    os.system(genetICPath + " " + reverseFilePath)
    os.system("mv " + args.outname + ext + " " + fname + "_rev" + ext)








