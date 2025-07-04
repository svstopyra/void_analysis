import sys
import os
import pynbody
import numpy as np
from void_analysis import context, stacking
from void_analysis.tools import zobovVolumesToPhysical
from void_analysis.tools import getHaloCentresAndMassesFromCatalogue
from void_analysis.simulation_tools import processSnapshot
import multiprocessing as mp
import scipy
thread_count = mp.cpu_count()
import pickle
import argparse

# Main function, accepts the name of the snapshot to be
#  processed as an argument:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = \
        "Process a pair and forward " + \
        "and reverse snapshot files, giving the anti-halo catalogues and " + \
        "various properties of it.")
    parser.add_argument('forward',\
        help = "Name of the snapshot file to process.")
    parser.add_argument('reverse',\
        help = "Name of the reverse snapshot to process.")
    parser.add_argument('--nBins',\
        help='Number of bins to use',type=int,default=31)
    parser.add_argument('--offset',\
        nargs = 1,help='Skip this number of bytes in the volumes file',\
        type=int,default=4)
    parser.add_argument('--output',nargs = 1,\
        help='Output filename (default: "<snap_filename>.AHproperties.p")',\
        type=str,default = None)
    parser.add_argument('--rmax',default=3.0,type=float,\
        help="Maximum effective radius out to which we compute profiles.")
    parser.add_argument('--rmin',default=0.0,type=float,\
        help="Minimum effective radius from which we compute profiles.")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.forward + ".AHproperties.p"
    processSnapshot(args.forward,args.reverse,args.nBins,offset=args.offset,\
        output = args.output,rMax=args.rmax,rMin=args.rmin)
















