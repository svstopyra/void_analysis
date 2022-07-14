import numpy as np
import pynbody
import nbodykit.lab as nlab
import nbodykit
import argparse
import pickle
import os

from nbodykit.source.catalog import ArrayCatalog

parser = argparse.ArgumentParser(description = "Compute the power spectrum of a snapshot and dump it to a file (default snapshot_filename.ps.p)")
parser.add_argument('filename',help = "Name of the snapshot file to process.")
parser.add_argument('--output',nargs = 1,help='Output filename (default: snapshot_filename.ps.p)',type=float)
parser.add_argument('--Om0',nargs = 1,help="Matter density (default: inferred from snapshot or 0.3)",type=float)
parser.add_argument('--Ob0',nargs = 1,help="Baryon density (default 0.04825)",type=float)
parser.add_argument('--H0',nargs = 1,help="H0 in km/s/Mpc (default 70)",type=float)
parser.add_argument('--Ode0',nargs = 1,help="Dark energy density (default: inferred from snapshot or 0.7)",type=float)
parser.add_argument('--sigma8',nargs = 1,help="sigma8 (default 0.8288)",type=float)
parser.add_argument('--ns',nargs = 1,help="Scalar power spectrum tilt (default 0.9611)",type=float,default = 0.9611)
parser.add_argument('--N',nargs = 1,help="Resolution of mesh used for Fourier transforms (default 256)",default = 256,type=int)
parser.add_argument('--dk',nargs = 1,type=float,default = 0.005,help="k-spacing for power spectrum.")
parser.add_argument('--kmin',nargs = 1,type=float,default = 0.01,help="Minimum value of k for power spectrum.")
parser.add_argument('--z',nargs = 1,type=float,help="Redshift of snapshot (default inferred from snapshot)")
parser.add_argument('--boxsize',nargs = 1,type=float,help="Size of the simulation box in Mpc/h (default: smallest cubic box that fits the data)")

args = parser.parse_args()

fname = args.filename
# Get extension - if affects how we do this!
fnameFirst, extension = os.path.splitext(fname)
if extension == '.h5':
    import h5py
    partFile = h5py.File(fname,'r')
    pos = partFile['u_pos'][()]
    # Compute particle mass, assuming default parameters:
    if args.Om0 is None:
        args.Om0 = [0.3]
    if args.boxsize is None:
        args.boxsize = [np.max(pos)*1.000001] # Slightly larger than the particles present
    rhoCrit = 2.7754e11 # Critical density in Msol/Mpc^3 h^2
    rhoMean = args.Om0[0]*rhoCrit
    Nsnap = len(pos)
    mPart = rhoMean*(args.boxsize[0]**3)/Nsnap
    data = np.empty(Nsnap,dtype=[('Position',('f8',3)),('Mass',('f8'))])
    data['Position'] = pos
    data['Mass'][:] = mPart    
else:
    snap = pynbody.load(fname)
    Nsnap = len(snap)
    # Convert snapshot data to an n-body catalogue:
    data = np.empty(Nsnap,dtype=[('Position',('f8',3)),('Mass',('f8'))])
    data['Position'] = snap['pos']
    data['Mass'] = snap['mass']
    if args.boxsize is None:
        args.boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")

cat = ArrayCatalog(data)
if args.output is None:
    output = fname + ".ps.p"

# Compute power spectrum:
mesh = cat.to_mesh(resampler = 'tsc',Nmesh=args.N,BoxSize = args.boxsize,\
    compensated=True,position = 'Position')
ps = nlab.FFTPower(mesh,mode='1d',dk = args.dk,kmin = args.kmin)

pickle.dump(ps,open(output,"wb"))

# Theory prediction for the power spectrum
#cosmo = nbodykit.cosmology.cosmology.Cosmology(Omega_cdm = Om0 - Ob0,h=h,Omega0_b=Ob0,n_s=parser['ns']).match(sigma8=sigma8)

