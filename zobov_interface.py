# Code for simplifying the running of ZOBOV
import os
import pynbody
from scipy.io import FortranFile
from shutil


# Run vozinit on the specified file:
def vozinit(position_file,boxsize,buffersize = 0.1,noDivisions = 2,
            suffix = ".vozinit"):
    # The actual vozinit script makes an error of assuming the executables are
    # all in the same folder as the file being processed. This is inconvenient,
    # so we'll actually just recreate it here.
    # os.system("vozinit " + position_file + " " + str(boxsize) + " "
    # + str(buffersize) + " " + str(noDivisions) + " " + suffix)
    f = open("scr" + suffix,'w')
    f.write("#!/bin/bash -f\n")
    for i in range(0,noDivisions):
        for j in range(0,noDivisions):
            for k in range(0,noDivisions):
                f.write("voz1b1 " + position_file + " " + str(buffersize) + " "
                        + str(boxsize) + " " + suffix + " "
                        + str(noDivisions) + " " + str(i) + " " + str(j) + " "
                        + str(k) + "\n")
    f.write("voztie " + str(noDivisions) + " " + suffix)
    f.close()


# Generate a ZOBOV particle file from a snapshot:
def zobovParticles(snap,filename):
    f = open(filename,'w+b')
    f.write(np.int32(len(snap)))
    snap['pos'][:,0].tofile(f)
    snap['pos'][:,1].tofile(f)
    snap['pos'][:,2].tofile(f)
    f.close()

# Generate VIDE scripts. There is a pipeline for this, but as it's in python2.7
# it's hard to interface with our other code. Easier to actually just create
# the scripts directly here.
def generatePipelineScripts(snapbase,scriptname,snaplist = None,continueRun = False,
                            startCatalogStage = 1,endCatalogStage = 3,
                            regenerateFlag = False,prefix = "zobov_voids_",
                            workDir="./",inputDir="./",figDir="./",logDir="./",
                            numZobovDivisions = 2,numZobovThreads = 8,
                            dataUnit = 1):
    f = open(scriptname,'w')
    f.write("#!/usr/bin/env/python\n")
    f.write("import os\n")
    f.write("from void_python_tools.backend.classes import *\n")
    f.write("\n")
    f.write("continueRun = " + str(continueRun) + " # set to True to enable restarting aborted jobs\n")
    f.write("startCatalogStage = " + str(startCatalogStage) + "\n")
    f.write("endCatalogStage = " + str(endCatalogStage) + "\n")
    f.write("\n")
    f.write("regenerateFlag = " + str(regenerateFlag) + "\n")
    # Figure out the path we need for zobov:
    zobovPath = os.path.dirname(shutil.which("vozinit"))
    ctoolsPath = os.path.dirname(zobovPath) + "/c_tools/"
    f.write("ZOBOV_PATH = \"" + zobovPath + "\"\n")
    f.write("CTOOLS_PATH = \"" + ctoolsPath + "\"\n\n")
    f.write("dataSampleList = []\n\n")
    setName = prefix + "ss1.0"
    f.write("setName = \"" + setName + "\"\n\n")
    f.write("workDir = \"" + workDir + setName + "/\"\n")
    f.write("inputDataDir = \"" + inputDir + "\"\n\n")
    f.write("figDir = \"" + figDir + setName + "/\"\n")
    f.write("logDir = \"" + logDir + setName + "/\"\n\n")
    f.write("numZobovDivisions = " + str(numZobovDivisions) + "\n")
    f.write("numZobovThreads = " + str(numZobovThreads) + "\n")

    if snaplist is None:
        # Count the number of snaps we can detect:
        snapcount = 0
        while(os.path.isfile(snapbase + "{:0>3d}".format(snapcount + 1))):
            snapcount += 1
        snaplist = range(1,snapcount+1)
    for k in range(0,len(snaplist)):
        # Add sections to the script for this snapshot
        snapname = snapbase + "{:0>3d}".format(snaplist[k])
        snap = pynbody.load(snapname)
        redshift = (1.0/snap.properties['a']) - 1.0
        boxsize = snap.properties['boxsize'].in_units("Mpc a h**-1")
        omegaM = snap.properties['omegaM0']
        addSample(f,snapname,redshift,boxsize,omegaM,dataUnit = dataUnit)
    f.close()



def addSample(file,sampleName,redshift,boxsize,omegaM,dataFormat = "gadget",
              dataUnit = 1,subsamples = 1.0,
              shiftSimZ = False,minVoidRadius = 2,profileBinSize = "auto",
              includeInHubble = True,partOfCombo = False,usePecVel=False,
              numSubvolumes = 1,mySubvolume = "00",useLightCone = False):
    nameString = sampleName + "ss" + str(subsamples) + "_z" + redshift + "0_d00"
    file.write("newSample = Sample(")
    file.write("dataFile = \"" + sampleName + "\",\n")
    file.write("dataFormat = \"" + dataFormat + "\",\n")
    file.write("dataUnit = \"" + dataUnit + "\",\n")
    file.write("fullName = \"" + nameString + "\",\n")
    file.write("nickName = \"" + nameString + "\",\n")
    file.write("dataType = \"simulation\",\n")
    file.write("zBoundary = (0.00,0.02),\n")
    file.write("zRange = (0.00,0.02),\n")
    file.write("zBoundaryMpc = (0.00," +str(boxsize) + ",\n")
    file.write("shiftSimZ = " + str(shiftSimZ) + ",\n")
    file.write("omegaM = " + str(omegaM) + ",\n")
    file.write("minVoidRadius = " + str(minVoidRadius) + ",\n")
    file.write("profileBinSize = " + profileBinSize + ",\n")
    file.write("includeInHubble = " + str(includeInHubble) + ",\n")
    file.write("partOfCombo = " + str(partOfCombo) + ",\n")
    file.write("boxLen = " + str(boxsize) + ",\n")
    file.write("usePecVel = " + str(usePecVel) + ",\n")
    file.write("numSubvolumes = " + str(numSubvolumes) + ",\n")
    file.write("mySubvolume = " +  mySubvolume + ",\n")
    file.write("subsample = " + str(subsamples) + ")\n")
