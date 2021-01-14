# This holds code used by plot that doesn't need to integrate with any graphical backends. This avoids having to load those backends on systems (such as clusters) that might not support them
import numpy as np

# Returns the centres of the bins, specified from their boundaries. Has one fewer elements.
def binCentres(bins):
	return (bins[1:len(bins)] + bins[0:(len(bins)-1)])/2

# Put the specified values in the specified bins:
# values - values to bin (array)
# bins - boundaries of the bins to use (will be one fewer bins than there are boundaries)
# Returns:
# binList - list of arrays giving the indices of all the elements of values that are in each bin.
# noInBins - number of elements in each bin
def binValues(values,bins):
	binList = []
	noInBins = np.zeros(len(bins)-1,dtype=int)
	for k in range(0,len(bins)-1):
		inThisBin = np.where((values >= bins[k]) & (values < bins[k+1]))[0]
		binList.append(inThisBin)
		noInBins[k] = len(inThisBin)		
	return [binList,noInBins]

def binValues2d(values,bins):
	binList = []
	for k in range(0,len(bins)-1):
		inThisBin = np.where((values >= bins[k]) & (values < bins[k+1]))
		binList.append(inThisBin)		
	return binList


