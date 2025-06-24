# Functions for generating void data from a pair of simulations
import pynbody


def processSimulationPair(sim1,sim2,directory=""):
	snapn = pynbody.load(directory + sim1) # forward simulation
	snapr = pynbody.load(directory + sim2) # reversed simulation
	# Load halo catalogues:
	hn = snapn.halos()
	hr = snapr.halos()
	# Bridge between simulations:
	bridge = pynbody.bridge.Bridge(snapn,snapr)
	
