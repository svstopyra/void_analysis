# Code for simplifying the running of ZOBOV
import os
import pynbody
from scipy.io import FortranFile

# VIDE requirements:
import numpy as np
from netCDF4 import Dataset
import sys
import pickle
from periodic_kdtree import PeriodicCKDTree
import os

NetCDFFile = Dataset
ncFloat = 'f8'

# Re-implementation of the void catalog code from VIDE, to deal with memory management better.

# Bunch class from VIDE
class Bunch:
	def __init__(self, **kwds):
		self.__dict__.update(kwds)


# Catalog class from VIDE
class Catalog:
	def __init__(self):
		self.numVoids = 0
		self.numPartTot = 0
		self.numZonesTot = 0
		self.volNorm = 0
		self.boxLen = np.zeros((3))
		self.ranges = np.zeros((3,2))
		self.part = None
		self.voids = None
		self.sampleInfo = None
		self._radius = None
		self._coreDens = None
		self._voidVol = None
		self._voidID = None
		self._voidProb = None
		self._densCon = None
		self._numPart = None
		self._ellipticity = None
		self._centralDen = None
		self._numZones = None
		self._parentID = None
		self._numChildren = None
		self._radius = None
		self._voidCentres = None
	# More efficient to make these functions that retrieve from an array, than to store
	# thousands of lists. We only need to store the start and end of each list in this array.
	def zones2Parts(self,zoneNum):
		return self.particleData[self.partListStarts[zoneNum]:(self.partListStarts[zoneNum] + self.numPartList[zoneNum])].tolist()
	def void2Zones(self,voidNum):
		voidID = self.voids[voidNum].voidID
		return self.zoneData[self.zoneListStarts[voidID]:(self.zoneListStarts[voidID] + self.numZonesList[voidID])].tolist()
	# Combine zones into a void, to get a list of particles in a void (most of the time there is only one though)
	def void2Parts(self,voidNum):
		zoneList = self.void2Zones(voidNum)
		partList = np.zeros(self.voids[voidNum].numPart,dtype=np.int32)
		cursor = 0
		for k in range(0,len(zoneList)):
			partList[cursor:(cursor + self.numPartList[zoneList[k]])] = self.zones2Parts(zoneList[k])
			cursor += self.numPartList[zoneList[k]]
		return partList.tolist()
	# Cached properties:
	@property
	def radius(self):
		if self._radius is None:
			self._radius = getVoidProperty(self,"radius")
		return self._radius
	@property
	def coreDens(self):
		if self._coreDens is None:
			self._coreDens = getVoidProperty(self,"coreDens")
		return self._coreDens
	@property
	def voidVol(self):
		if self._voidVol is None:
			self._voidVol = getVoidProperty(self,"voidVol")
		return self._voidVol
	@property
	def voidID(self):
		if self._voidID is None:
			self._voidID = getVoidProperty(self,"voidID")
		return self._voidID
	@property
	def voidProb(self):
		if self._voidProb is None:
			self._voidProb = getVoidProperty(self,"voidProb")
		return self._voidProb	
	@property
	def densCon(self):
		if self._densCon is None:
			self._densCon = getVoidProperty(self,"densCon")
		return self._densCon
	@property
	def numPart(self):
		if self._numPart is None:
			self._numPart = getVoidProperty(self,"numPart")
		return self._numPart	
	@property
	def ellipticity(self):
		if self._ellipticity is None:
			self._ellipticity = getVoidProperty(self,"ellipticity")
		return self._ellipticity
	@property
	def centralDen(self):
		if self._centralDen is None:
			self._centralDen = getVoidProperty(self,"centralDen")
		return self._centralDen
	@property
	def numZones(self):
		if self._numZones is None:
			self._numZones = getVoidProperty(self,"numZones")
		return self._numZones
	@property
	def parentID(self):
		if self._parentID is None:
			self._parentID = getVoidProperty(self,"parentID")
		return self._parentID
	@property
	def numChildren(self):
		if self._numChildren is None:
			self._numChildren = getVoidProperty(self,"numChildren")
		return self._numChildren
	@property
	def treeLevel(self):
		if self._radius is None:
			self._treeLevel = getVoidProperty(self,"treeLevel")
		return self._treeLevel
	@property
	def voidCentres(self):
		if self._voidCentres is None:
			self._voidCentres = np.zeros((len(self.voids),3))
			for k in range(0,len(self.voids)):
				self._voidCentres[k,:] = self.voids[k].macrocenter
		return self._voidCentres
	def children(self,voidNum):
		return np.where(self.parentID == self.voids[voidNum].voidID)[0]

# Load VIDE catalogs, but make particles not-loaded by default since these seems to cause memory
# issues. We want to be able to only load particles as and when we need them.
def loadVIDEVoidCatalog(sampleDir, dataPortion="central", loadParticles=False,untrimmed=False):
	# loads a void catalog
	# by default, loads parent-level voids with central densities greater than 0.2*mean
	#   sampleDir: path to VIDE output directory
	#   dataPortion: "central" or "all"
	#   loadParticles: if True, also load particle information
	#   untrimmed: if True, catalog contains all voids, regardless of density or hierarchy level

	sys.stdout.flush()

	catalog = Catalog()

	# Extract sample name:
	a = open(sampleDir + "/sample_info.txt","r")
	line = a.readline()
	fullName = line[13:-1]# Extracts the sample name from sample_info.txt.
	#with open(sampleDir+"/sample_info.dat", 'rb') as input:
	#	sample = pickle.load(input)
	#catalog.sampleInfo = sample

	print("Loading info...")
	infoFile = sampleDir+"/zobov_slice_"+fullName+".par"
	File = NetCDFFile(infoFile, 'r')
	ranges = np.zeros((3,2))
	ranges[0][0] = getattr(File, 'range_x_min')
	ranges[0][1] = getattr(File, 'range_x_max')
	ranges[1][0] = getattr(File, 'range_y_min')
	ranges[1][1] = getattr(File, 'range_y_max')
	ranges[2][0] = getattr(File, 'range_z_min')
	ranges[2][1] = getattr(File, 'range_z_max')
	catalog.boxLen[0] = ranges[0][1] - ranges[0][0]
	catalog.boxLen[1] = ranges[1][1] - ranges[1][0]
	catalog.boxLen[2] = ranges[2][1] - ranges[2][0]
	catalog.ranges = ranges
	File.close()

	volNorm = getVolNorm(sampleDir)
	catalog.volNorm = volNorm

	if untrimmed:
		prefix = "untrimmed_"
	else:
		prefix = ""

	print("Loading voids...")
	fileName = sampleDir+"/"+prefix+"voidDesc_"+dataPortion+"_"+fullName+".out"
	catData = np.loadtxt(fileName, comments="#", skiprows=2)
	catalog.voids = []
	for line in catData:
		catalog.voids.append(Bunch(iVoid = int(line[0]),
		               voidID = int(line[1]),
		               coreParticle = line[2],
		               coreDens = line[3],
		               zoneVol = line[4],
		               zoneNumPart = line[5],
		               numZones = int(line[6]),
		               voidVol = line[7],
		               numPart = int(line[8]),
		               densCon = line[9],
		               voidProb = line[10],
		               radius = 0., # this is read in later
		               macrocenter = np.zeros((3)),
		               redshift = 0,
		               RA = 0,
		               Dec = 0,
		               parentID = 0,
		               treeLevel = 0,
		               numChildren = 0,
		               centralDen = 0.,
		               ellipticity = 0.,
		               eigenVals = np.zeros((3)),
		               eigenVecs = np.zeros((3,3)),
		               ))

	catalog.numVoids = len(catalog.voids)
	print("Read %d voids" % catalog.numVoids  )

	print("Loading macrocenters...")
	iLine = 0
	for line in open(sampleDir+"/"+prefix+"macrocenters_"+dataPortion+"_"+fullName+".out"):
		line = line.split()
		catalog.voids[iLine].macrocenter[0] = float(line[1])
		catalog.voids[iLine].macrocenter[1] = float(line[2])
		catalog.voids[iLine].macrocenter[2] = float(line[3])
		iLine += 1

	iLine = 0
	fileName = sampleDir+"/"+prefix+"sky_positions_"+dataPortion+"_"+fullName+".out"
	catData = np.loadtxt(fileName, comments="#")
	for line in catData:
		catalog.voids[iLine].RA = float(line[0])
		catalog.voids[iLine].Dec = float(line[1])
		iLine += 1


	print("Loading derived void information...")
	fileName = sampleDir+"/"+prefix+"centers_"+dataPortion+"_"+fullName+".out"
	catData = np.loadtxt(fileName, comments="#")
	for (iLine,line) in enumerate(catData):
		catalog.voids[iLine].volume = float(line[6])
		catalog.voids[iLine].radius = float(line[4])
		catalog.voids[iLine].redshift = float(line[5])
		catalog.voids[iLine].parentID = float(line[10])
		catalog.voids[iLine].treeLevel = float(line[11])
		catalog.voids[iLine].numChildren = float(line[12])
		catalog.voids[iLine].centralDen = float(line[13])
		iLine += 1

	fileName = sampleDir+"/"+prefix+"shapes_"+dataPortion+"_"+fullName+".out"
	catData = np.loadtxt(fileName, comments="#")
	for (iLine,line) in enumerate(catData):
		catalog.voids[iLine].ellipticity = float(line[1])
		
		catalog.voids[iLine].eigenVals[0] = float(line[2])
		catalog.voids[iLine].eigenVals[1] = float(line[3])
		catalog.voids[iLine].eigenVals[2] = float(line[4])
		
		catalog.voids[iLine].eigenVecs[0][0] = float(line[5])
		catalog.voids[iLine].eigenVecs[0][1] = float(line[6])
		catalog.voids[iLine].eigenVecs[0][2] = float(line[7])
		
		catalog.voids[iLine].eigenVecs[1][0] = float(line[8])
		catalog.voids[iLine].eigenVecs[1][1] = float(line[9])
		catalog.voids[iLine].eigenVecs[1][2] = float(line[10])
		
		catalog.voids[iLine].eigenVecs[2][0] = float(line[11])
		catalog.voids[iLine].eigenVecs[2][1] = float(line[12])
		catalog.voids[iLine].eigenVecs[2][2] = float(line[13])
		
		iLine += 1

	if loadParticles:
		print("Loading all particles...")
		partData, boxLen, volNorm, isObservationData, ranges, extraData = loadPart(sampleDir)
		numPartTot = len(partData)
		catalog.numPartTot = numPartTot
		catalog.partPos = partData
		# VIDE .part object is inefficient to construct, quickly overflowing the available RAM. I suspect this is because of the append not doing garbage collection correctly. We should simply store all this in a numpy array.
		catalog.part_pos = partData
		catalog.extraData = extraData
		


		print("Loading volumes...")
		volFile = sampleDir+"/vol_"+fullName+".dat"
		File = open(volFile)
		chk = np.fromfile(File, dtype=np.int32,count=1)
		vols = np.fromfile(File, dtype=np.float32,count=numPartTot)
		catalog.volumes = vols

		print("Loading zone-void membership info...")
		zoneFile = sampleDir+"/voidZone_"+fullName+".dat"
		File = open(zoneFile)
		catalog.zoneData = np.fromfile(File,dtype=np.int32)
		numZonesTot = catalog.zoneData[0]
		catalog.numZonesTot = numZonesTot
		catalog.numZonesList = np.zeros(numZonesTot,dtype=np.int32)
		catalog.zoneListStarts = np.zeros(numZonesTot,dtype=np.int32)
		cursor = 1
		for iZ in range(numZonesTot):
			numZones = catalog.zoneData[cursor]
			catalog.numZonesList[iZ] = numZones
			cursor += 1
			catalog.zoneListStarts[iZ] = cursor
			cursor += numZones


		print("Loading particle-zone membership info...")
		zonePartFile = sampleDir+"/voidPart_"+fullName+".dat"
		File = open(zonePartFile)
		catalog.particleData = np.fromfile(File,dtype=np.int32)
		numZonesTot = catalog.particleData[1]
		catalog.numPartList = np.zeros(numZonesTot,dtype=np.int32)
		catalog.partListStarts = np.zeros(numZonesTot,dtype=np.int32)
		cursor = 2
		for iZ in range(numZonesTot):
			numPart = catalog.particleData[cursor]
			cursor += 1
			catalog.numPartList[iZ] = numPart
			catalog.partListStarts[iZ] = cursor
			cursor += numPart

	return catalog

# Filter a catalog based on some subset of voids:
def getFilteredCatalog(cat,voidFilter):
	newCat = Catalog()
	newCat.numVoids = len(voidFilter[0])
	newCat.numPartTot = cat.numPartTot
	newCat.volNorm = cat.volNorm
	newCat.boxLen = cat.boxLen
	newCat.ranges = cat.ranges
	#newCat.voids = cat.voids[voidFilter]
	newCat.voids = []
	for k in range(0,len(voidFilter[0])):
		newCat.voids.append(cat.voids[voidFilter[0][k]])
	# Particle data:
	if hasattr(cat,'partPos'):
		newCat.partPos = cat.partPos
	if hasattr(cat,'part'):
		newCat.part = cat.part
	if hasattr(cat,'extraData'):
		newCat.extraData = cat.extraData
	if hasattr(cat,'volumes'):
		newCat.volumes = cat.volumes
	if hasattr(cat,'particleData'):
		newCat.particleData = cat.particleData
	if hasattr(cat,'numPartList'):
		newCat.numPartList = cat.numPartList
	if hasattr(cat,'partListStarts'):
		newCat.partListStarts = cat.partListStarts
	# Zone data:
	if hasattr(cat,'zoneData'):
		newCat.zoneData = cat.zoneData
	if hasattr(cat,'numZonesTot'):
		newCat.numZonesTot = cat.numZonesTot
	if hasattr(cat,'numZonesList'):
		newCat.numZonesList = cat.numZonesList
	if hasattr(cat,'zoneListStarts'):
		newCat.zoneListStarts = cat.zoneListStarts
	if hasattr(cat,'zoneListStarts'):
		newCat.zoneListStarts = cat.zoneListStarts
	return newCat

# Get ZONES only catalog. That is, catalogue where zones are never joined together.
def zonesOnlyCatalog(cat,zoneCentres=None,zoneRadii = None):
	newCat = Catalog()
	# Each zone is regarded as a voids in its own right:
	newCat.numVoids = cat.numZonesTot
	# Copy other properties:
	newCat.numPartTot = cat.numPartTot
	newCat.volNorm = cat.volNorm
	newCat.boxLen = cat.boxLen
	newCat.numZonesTot = cat.numZonesTot
	newCat.ranges = cat.ranges
	newCat.voids = []
	if (zoneCentres is None):
		zoneCentres = np.zeros((cat.numZonesTot,3))
		periodicity = [at.boxLen]*3
		for k in range(0,cat.numZonesTot):
			zoneParts = cat.zones2Parts(k)
			zoneCentres[k,:] = computePeriodicCentreWeighted(snap[zoneParts]['pos'],volumes[zoneParts],periodicity)
	for k in range(0,netCat.numVoids):
		void = Bunch(iVoid = int(line[0]),
		               voidID = int(line[1]),
		               coreParticle = line[2],
		               coreDens = line[3],
		               zoneVol = line[4],
		               zoneNumPart = line[5],
		               numZones = int(line[6]),
		               voidVol = line[7],
		               numPart = int(line[8]),
		               densCon = line[9],
		               voidProb = -1,
		               radius = zoneRadii[k], # this is read in later
		               macrocenter = zoneCentres[k,:],
		               redshift = 0,
		               RA = 0,
		               Dec = 0,
		               parentID = 0,
		               treeLevel = 0,
		               numChildren = 0,
		               centralDen = 0.,
		               ellipticity = 0.,
		               eigenVals = np.zeros((3)),
		               eigenVecs = np.zeros((3,3)),
		               )
	return newCat
	
# Filter voids with a range of values for a given property (defaults to all voids):
def filterVoidsOnRange(cat,filterProperty,rMin = -np.inf,rMax=np.inf):
	propertyVals = getattr(cat,filterProperty)
	filterVal = np.where((propertyVals >= rMin) & (propertyVals <= rMax))
	return getFilteredCatalog(cat,filterVal)

# Filter voids with a specific value for a given property:
def filterVoidsOnValue(cat,filterProperty,value):
	propertyVals = getattr(cat,filterProperty)
	filterVal = np.where(propertyVals == value)
	return getFilteredCatalog(cat,filterVal)

# Re-implemented version of getVolNorm from VIDE:
def getVolNorm(sampleDir):
	a = open(sampleDir + "/sample_info.txt","r")
	line = a.readline()
	fullName = line[13:-1]

	infoFile = sampleDir+"/zobov_slice_"+fullName+".par"
	File = NetCDFFile(infoFile, 'r')
	ranges = np.zeros((3,2))
	ranges[0][0] = getattr(File, 'range_x_min')
	ranges[0][1] = getattr(File, 'range_x_max')
	ranges[1][0] = getattr(File, 'range_y_min')
	ranges[1][1] = getattr(File, 'range_y_max')
	ranges[2][0] = getattr(File, 'range_z_min')
	ranges[2][1] = getattr(File, 'range_z_max')
	isObservation = getattr(File, 'is_observation')
	maskIndex = getattr(File, 'mask_index')
	File.close()
	mul = np.zeros((3))
	mul[:] = ranges[:,1] - ranges[:,0]

	partFile = sampleDir+"/zobov_slice_"+fullName
	File = open(partFile)
	chk = np.fromfile(File, dtype=np.int32,count=1)
	Np = np.fromfile(File, dtype=np.int32,count=1)
	File.close()

	boxLen = mul

	#if isObservation == 1:
	#  # look for the mask file
	#  if os.access(sample.maskFile, os.F_OK):
	#    maskFile = sample.maskFile
	#  else:
	#    maskFile = sampleDir+"/"+os.path.basename(sample.maskFile)
	#    print "Using maskfile found in:", maskFile
	#  props = vp.getSurveyProps(maskFile, sample.zBoundary[0],
	#                            sample.zBoundary[1], 
	#                            sample.zBoundary[0], 
	#                            sample.zBoundary[1], "all",
	#                            selectionFuncFile=sample.selFunFile,
	#                            useComoving=sample.useComoving)
	#  boxVol = props[0]
	#  volNorm = maskIndex/boxVol
	#else:
	boxVol = np.prod(boxLen) 
	volNorm = Np/boxVol

	return volNorm

# Re-implemented loadPart from VIDE:
def loadPart(sampleDir):
	print("    Loading particle data...")
	sys.stdout.flush()

	a = open(sampleDir + "/sample_info.txt","r")
	line = a.readline()
	fullName = line[13:-1]

	infoFile = sampleDir+"/zobov_slice_"+fullName+".par"
	File = NetCDFFile(infoFile, 'r')
	ranges = np.zeros((3,2))
	ranges[0][0] = getattr(File, 'range_x_min')
	ranges[0][1] = getattr(File, 'range_x_max')
	ranges[1][0] = getattr(File, 'range_y_min')
	ranges[1][1] = getattr(File, 'range_y_max')
	ranges[2][0] = getattr(File, 'range_z_min')
	ranges[2][1] = getattr(File, 'range_z_max')
	isObservation = getattr(File, 'is_observation')
	maskIndex = getattr(File, 'mask_index')
	File.close()
	mul = np.zeros((3))
	mul[:] = ranges[:,1] - ranges[:,0]

	partFile = sampleDir+"/zobov_slice_"+fullName
	iLine = 0
	partData = []
	part = np.zeros((3))
	File = open(partFile)
	chk = np.fromfile(File, dtype=np.int32,count=1)
	Np = np.fromfile(File, dtype=np.int32,count=1)[0]
	chk = np.fromfile(File, dtype=np.int32,count=1)

	chk = np.fromfile(File, dtype=np.int32,count=1)
	x = np.fromfile(File, dtype=np.float32,count=Np)
	x *= mul[0]
	if isObservation != 1:
		x += ranges[0][0]
	chk = np.fromfile(File, dtype=np.int32,count=1)

	chk = np.fromfile(File, dtype=np.int32,count=1)
	y = np.fromfile(File, dtype=np.float32,count=Np)
	y *= mul[1] 
	if isObservation != 1:
		y += ranges[1][0]
	chk = np.fromfile(File, dtype=np.int32,count=1)

	chk = np.fromfile(File, dtype=np.int32,count=1)
	z = np.fromfile(File, dtype=np.float32,count=Np)
	z *= mul[2] 
	if isObservation != 1:
		z += ranges[2][0]
	chk = np.fromfile(File, dtype=np.int32,count=1)

	chk = np.fromfile(File, dtype=np.int32,count=1)
	RA = np.fromfile(File, dtype=np.float32,count=Np)
	chk = np.fromfile(File, dtype=np.int32,count=1)

	chk = np.fromfile(File, dtype=np.int32,count=1)
	Dec = np.fromfile(File, dtype=np.float32,count=Np)
	chk = np.fromfile(File, dtype=np.int32,count=1)

	chk = np.fromfile(File, dtype=np.int32,count=1)
	redshift = np.fromfile(File, dtype=np.float32,count=Np)
	chk = np.fromfile(File, dtype=np.int32,count=1)

	chk = np.fromfile(File, dtype=np.int32,count=1)
	uniqueID = np.fromfile(File, dtype=np.int64,count=Np)
	chk = np.fromfile(File, dtype=np.int32,count=1)

	File.close()


	if isObservation == 1:
		x = x[0:maskIndex]# * 100/300000
		y = y[0:maskIndex]# * 100/300000
		z = z[0:maskIndex]# * 100/300000
		RA = RA[0:maskIndex]
		Dec = Dec[0:maskIndex]
		redshift = redshift[0:maskIndex]
		uniqueID = uniqueID[0:maskIndex]

	partData = np.column_stack((x,y,z))

	extraData = np.column_stack((RA,Dec,redshift,uniqueID))

	boxLen = mul

	#if isObservation == 1:
	#  # look for the mask file
	#  if os.access(sample.maskFile, os.F_OK):
	#    maskFile = sample.maskFile
	#  else:
	#    maskFile = sampleDir+"/"+os.path.basename(sample.maskFile)
	#    print "Using maskfile found in:", maskFile
	#  props = vp.getSurveyProps(maskFile, sample.zBoundary[0],
	#                            sample.zBoundary[1], 
	#                            sample.zBoundary[0], 
	#                            sample.zBoundary[1], "all",
	#                            selectionFuncFile=sample.selFunFile,
	#                            useComoving=sample.useComoving)
	#  boxVol = props[0]
	#  volNorm = maskIndex/boxVol
	#else:
	boxVol = np.prod(boxLen) 
	volNorm = len(x)/boxVol

	isObservationData = isObservation == 1

	return partData, boxLen, volNorm, isObservationData, ranges, extraData


# Run vozinit on the specified file:
def vozinit(position_file,boxsize,buffersize = 0.1,noDivisions = 2,suffix = ".vozinit"):
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
def generatePipelineScripts(snapbase,scriptname,snaplist = None,continueRun = False,startCatalogStage = 1,endCatalogStage = 3,regenerateFlag = False,prefix = "zobov_voids_",workDir="./",inputDir="./",figDir="./",logDir="./",numZobovDivisions = 2,numZobovThreads = 8,dataUnit = 1):
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



def addSample(file,sampleName,redshift,boxsize,omegaM,dataFormat = "gadget",dataUnit = 1,subsamples = 1.0,shiftSimZ = False,minVoidRadius = 2,profileBinSize = "auto",includeInHubble = True,partOfCombo = False,usePecVel=False,numSubvolumes = 1,mySubvolume = "00",useLightCone = False):
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

# Extract void properties to an array:
def getVoidProperty(voidCat,voidProperty):
	propertyToReturn = np.zeros(len(voidCat.voids),dtype=type(getattr(voidCat.voids[0],voidProperty)))
	for k in range(0,len(propertyToReturn)):
		propertyToReturn[k] = getattr(voidCat.voids[k],voidProperty)
	return propertyToReturn

def getCatalogVolume(cat):
	return cat.boxLen[0]*cat.boxLen[1]*cat.boxLen[2]

# Convert the halo file for a snapshot to something that VIDE can directly understand
def convertHalos(snap,filename,unitFactor = 1000.0,printHeaderInfo = False):
	h = snap.halos()
	haloFile = open(filename,'w')
	if printHeaderInfo:
		haloFile.write(str(snap.properties['boxsize'].ratio("Mpc a h**-1")) + "\n")
		haloFile.write(str(snap.properties['omegaM0']) + "\n")
		haloFile.write(str(snap.properties['h']) + "\n")
		haloFile.write(str(1.0/(snap.properties['a']) - 1.0) + "\n")
		haloFile.write(str(len(h)) + "\n")
	for k in range(0,len(h)):
		haloFile.write(str(k+1) + " ")
		haloFile.write(str(h[k+1].properties['Xc']/unitFactor) + " ")
		haloFile.write(str(h[k+1].properties['Yc']/unitFactor) + " ")
		haloFile.write(str(h[k+1].properties['Zc']/unitFactor) + " ")
		haloFile.write(str(h[k+1].properties['VXc']) + " ")
		haloFile.write(str(h[k+1].properties['VYc']) + " ")
		haloFile.write(str(h[k+1].properties['VZc']) + " ")
		haloFile.write(str(np.float32(np.sum(h[k+1]['mass']))*1e10) + "\n")
	haloFile.close()

# Turn the normalised volumes supplied by zobov into physical units appropriate to the supplied simulation:
def zobovVolumesToPhysical(zobovVolumes,snap):
	N = np.round(np.cbrt(len(snap))).astype(int)
	boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
	if type(zobovVolumes) == type(""):
		vols = np.fromfile(zobovVolumes,dtype=np.float32)
	else:
		vols = zobovVolumes
	return vols[1:]*(boxsize/N)**3
	
# Compute the masses of zobov voids:
def zobovVoidMasses(cat,particleMasses):
	zobovMasses = np.zeros(len(cat.radius))
	for k in range(0,len(zobovMasses)):
		zobovMasses[k] = np.sum(particleMasses[cat.void2Parts(k)])
	return zobovMasses






