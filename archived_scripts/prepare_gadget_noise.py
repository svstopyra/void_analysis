import pynbody
import numpy as np
from scipy import io
from void_analysis import snapedit


def importWhiteNoiseData(wnFile,inputFormat,dtypeIn = np.double):
	if inputFormat == "fortran":
		f = io.FortranFile(wnFile,mode = "r")
		row1 = f.read_ints()
	elif inputFormat == "music_binary":
		f = open(wnFile,"rb")
		row1 = 	np.hstack((np.fromfile(f,dtype=np.int32,count=3),np.array([0],dtype=np.int32)))
	else:
		raise Exception("Unrecognised file format. Options are `fortran' and `music_binary'")
	N = row1[0]
	data = np.zeros((N,N*N),dtype=dtypeIn)
	for k in range(0,N):	
		if inputFormat == "fortran":
			data[k] = f.read_record(dtype=dtypeIn)
		else:
			data[k] = np.fromfile(f,dtype=dtypeIn,count = N*N)
	return [data,row1,N]

# Functions to reproduce the index mapping from genetIC
def getCoordinateFromIndex(id,n):
	n2 = n**2
	n3 = n**3
	idx = np.reshape(np.array(np.floor(id/n2),dtype=np.int),(len(id),1))
	idz = np.reshape(id,(len(id),1))  - idx*n2
	idy = np.reshape(np.array(np.floor(idz/n),dtype=np.int),(len(id),1))
	idz -= idy*n
	return np.hstack((idx,idy,idz))

	
def getIndexFromCoordinateNoWrap(coord,n,perm = np.array([0,1,2])):
	if len(coord.shape) == 1:
		# Single coordinate
		index = (coord[perm[0]] *n + coord[perm[1]]) * n + coord[perm[2]]
	else:
		index = (coord[:,perm[0]] *n + coord[:,perm[1]]) * n + coord[:,perm[2]]
	return index


def getIndexFromCoordinate(coord,n):
	coordNew = np.mod(coord,n)
	return getIndexFromCoordinateNoWrap(coordNew,n)

def getCentroidFromCoordinate(coord,n,boxsize,offsetLower = np.array([0,0,0])):
	cellsize = boxsize/n
	result = np.array(coord,dtype=np.float)
	result *= cellsize
	result += offsetLower
	result += cellsize/2
	return result


def getCentroidFromIndex(id,n,boxsize,offsetLower = np.array([0,0,0])):
	coord = getCoordinateFromIndex(id,n)
	return getCentroidFromCoordinate(coord,n,boxsize,offsetLower = offsetLower)

	
def getIndexFromPoint(point,n,boxsize,offsetLower = np.array([0,0,0])):
	cellsize = boxsize/n
	coords = np.array(np.floor(np.mod(point-offsetLower,boxsize)/cellsize),dtype=np.int)
	return getIndexFromCoordinateNoWrap(coords,n)


def indexPermutation(N,permIn,permOut):
	if all(np.sort(permIn) != np.array([0,1,2])) or all(np.sort(permOut) != np.array([0,1,2])):
		raise Exception("Must supply a permutation of 0,1,2")
	indLin = getIndexFromCoordinateNoWrap(snapedit.gridListPermutation(N,perm = permOut),N,perm=permIn)
	return indLin


def minmax(x):
	return np.array([np.min(x),np.max(x)])

# Convert MUSIC input white noise into genetIC white noise
def generateGenetICNoise(wnFile,wnOutFile):
	[data,row1,N] = importWhiteNoiseData(wnFile,inputFormat,dtypeIn=dtypeIn)
	data = np.reshape(data,N**3)
	ind = indexPermutation(N,(0,1,2),(2,1,0))
	data = data[ind]
	data = np.reshape(data,(N,N,N))
	np.save(wnOutFile,data)


if __name__ == "__main__":
	if len(sys.argv) > 1:
		wnFile = sys.argv[1]
	else:
		raise Exception("Usage: prepare_gadget_noise.py input.dat output.dat")
	if len(sys.argv) > 2:
		wnOutFile = sys.argv[2]
	else:
		wnOutFile = "wn.npy"
	generateGenetICNoise(wnFile,wnOutFile)
