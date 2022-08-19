import healpy
import pynbody
import numpy as np
from scipy import io
from void_analysis import snapedit, context, stacking
import pynbody.plot.sph as sph
import matplotlib.pylab as plt
import pickle
import os
import astropy
import scipy.special as special
import scipy
import alphashape
from descartes import PolygonPatch
import matplotlib.cm as cm
import matplotlib.colors as colors

from astropy.coordinates import SkyCoord
import astropy.units as u

# Constants:
c = 299792.458


def getClusterSkyPositions(fileroot=""):
    # Abell clusters catalogue data:
    file3 = open(fileroot + "VII_110A/table3.dat",'r')
    fileLines3 = []
    for line in file3:
        fileLines3.append(line)
    file3.close()
    file4 = open(fileroot + "VII_110A/table4.dat",'r')
    fileLines4 = []
    for line in file4:
        fileLines4.append(line)
    file4.close()
    # Extract sky co-ordinates, abell numbers, and redshift:
    abell_l3 = np.zeros(len(fileLines3))
    abell_b3 = np.zeros(len(fileLines3))
    abell_n3 = np.zeros(len(fileLines3),dtype=int)
    abell_z3 = np.zeros(len(fileLines3))
    abell_l4 = np.zeros(len(fileLines4))
    abell_b4 = np.zeros(len(fileLines4))
    abell_n4 = np.zeros(len(fileLines4),dtype=int)
    abell_z4 = np.zeros(len(fileLines4))
    for k in range(0,len(fileLines3)):
        abell_n3[k] = int(fileLines3[k][0:4])
        abell_l3[k] = np.double(fileLines3[k][118:124])
        abell_b3[k] = np.double(fileLines3[k][125:131])
        abell_z3[k] = np.double(fileLines3[k][133:138].replace(' ','0'))
    for k in range(0,len(fileLines4)):
        abell_n4[k] = int(fileLines4[k][0:4])
        abell_l4[k] = np.double(fileLines4[k][118:124])
        abell_b4[k] = np.double(fileLines4[k][125:131])
        abell_z4[k] = np.double(fileLines4[k][133:138].replace(' ','0'))
    # Indicate missing redshifts:
    abell_z3[np.where(abell_z3 == 0.0)] = -1
    abell_z4[np.where(abell_z4 == 0.0)] = -1
    havez3 = np.where(abell_z3 > 0)
    havez4 = np.where(abell_z4 > 0)
    # Combine into a single set of arrays:
    abell_l = np.hstack((abell_l3[havez3],abell_l4[havez4]))
    abell_b = np.hstack((abell_b3[havez3],abell_b4[havez4]))
    abell_n = np.hstack((abell_n3[havez3],abell_n4[havez4]))
    abell_z = np.hstack((abell_z3[havez3],abell_z4[havez4]))
    abell_d = c*abell_z/100 # Distance in Mpc/h
    # Convert to Cartesian coordinates:
    coordAbell = SkyCoord(l=abell_l*u.deg,b=abell_b*u.deg,\
        distance=abell_d*u.Mpc,frame='galactic')
    p_abell = np.zeros((len(coordAbell),3))
    p_abell[:,0] = coordAbell.icrs.cartesian.x.value
    p_abell[:,1] = coordAbell.icrs.cartesian.y.value
    p_abell[:,2] = coordAbell.icrs.cartesian.z.value
    return [abell_l,abell_b,abell_n,abell_z,abell_d,p_abell,coordAbell]


def getAntiHalosInSphere(centres,radius,origin=np.array([0,0,0]),deltaCentral = None,boxsize=None,workers=-1,filterCondition = None):
    if filterCondition is None:
        filterCondition = np.ones(len(centres),dtype=np.bool)
    if deltaCentral is not None:
        usedIndices = np.where((deltaCentral < 0) & filterCondition)[0]
        centresToUse = centres[usedIndices,:]
    else:
        usedIndices = np.where(filterCondition)[0]
        centresToUse = centres[usedIndices,:]
    if boxsize is not None:
        tree = scipy.spatial.cKDTree(snapedit.wrap(centresToUse,boxsize),boxsize=boxsize)
    else:
        tree = scipy.spatial.cKDTree(centresToUse,boxsize=boxsize)
    inRadius = tree.query_ball_point(origin,radius,workers=workers)

    if len(origin.shape) == 1:
        inRadiusFinal = usedIndices[inRadius]
        condition = np.zeros(len(centres),dtype=np.bool)
        condition[inRadiusFinal] = True
    else:
        inRadiusFinal = [list(usedIndices[k]) for k in inRadius]
        condition = np.zeros((len(centres),len(origin)),dtype=np.bool)
        for k in range(0,len(origin)):
            condition[inRadiusFinal[k],k] = True
    return [inRadiusFinal,condition]

def getClusterCounterpartPositions(abell_nums,abell_n,p_abell,snap,hncentres,hnmasses,\
        rSearch = 20,mThresh=4e14,boxsize = None):
    superClusterCentres = p_abell[np.isin(\
        abell_n,abell_nums),:]
    if boxsize is None:
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")

    [haloCounterpartsLarge,condition] =\
        getAntiHalosInSphere(hncentres,rSearch,origin=superClusterCentres,\
            boxsize=boxsize,filterCondition = (hnmasses > mThresh))
    largeHalosList = []
    for k in range(0,len(haloCounterpartsLarge)):
        if len(haloCounterpartsLarge[k]) > 0:
            largeHalosList.append(haloCounterpartsLarge[k][0])
        else:
            largeHalosList.append(-1)
    largeHalosList = np.array(largeHalosList,dtype=int)
    sort = []
    for k in range(0,len(abell_nums)):
        selection = np.where(abell_n[np.isin(abell_n,abell_nums)] == \
            abell_nums[k])[0]
        if len(selection) > 0:
            sort.append(selection[0])
    return largeHalosList


def getGriddedDensity(snap,N,redshiftSpace= False,velFudge = 1,snapPos = None,snapVel = None,
        snapMass = None):
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if snapPos is None:
        snapPos = np.array(snap['pos'].in_units("Mpc a h**-1"))
    if snapVel is None:
        snapVel = snap['vel']
    if snapMass is None:
        snapMass = np.array(snap['mass'].in_units('Msol h**-1'))
    if redshiftSpace:
        cosmo = astropy.cosmology.LambdaCDM(snap.properties['h']*100,\
            snap.properties['omegaM0'],snap.properties['omegaL0'])
        pos = eulerToZ(snapPos,snapVel,\
        cosmo,boxsize,snap.properties['h'],velFudge=velFudge)
    else:
        pos = snapPos
    H, edges = np.histogramdd(pos,bins = N,\
        range = ((-boxsize/2,boxsize/2),(-boxsize/2,boxsize/2),\
            (-boxsize/2,boxsize/2)),\
            weights = snapMass,normed=False)
    cellWidth = boxsize/N
    cellVol = cellWidth**3
    meanDensity = np.double(np.sum(snapMass))/(boxsize**3)
    density = H/(cellVol*meanDensity)
    # Deal with an ordering issue:
    density = np.reshape(np.reshape(density,256**3),(256,256,256),order='F')
    return density


def getCombinedAbellCatalogue(Om0 = 0.3111,Ode0 = 0.6889,h=0.6766,\
        catFolder="",abell_nums = [426,2147,1656,3627,3571,548,2197,2063,1367]):
    cosmo = astropy.cosmology.LambdaCDM(100*h,Om0,Ode0)
    [abell_l,abell_b,abell_n,abell_z,\
            abell_d,p_abell,coordAbell] = getClusterSkyPositions(catFolder)
    # ABELL DATA FROM XML FILE:
    # https://ui.adsabs.harvard.edu/abs/2014MNRAS.445.4073C/abstract
    import xml.etree.ElementTree as ET
    xmlTree = ET.parse(catFolder + "VII_110A/aco_redshifts.xml")
    root = xmlTree.getroot()
    xmlAbell = [int(tr[0].text) for tr in root[0][0][25][0]]
    xmlID = [tr[1].text for tr in root[0][0][25][0]]
    xmlRA = np.array([float(tr[2].text) for tr in root[0][0][25][0]])
    xmlDEC = np.array([float(tr[3].text) for tr in root[0][0][25][0]])
    xmlZ = -np.ones(len(xmlRA))
    for k in range(0,len(xmlRA)):
        if root[0][0][25][0][k][6].text is not None:
            xmlZ[k] = float(root[0][0][25][0][k][6].text)
    c = 299792.458 # Speed of light in km/s
    xmlDist = cosmo.comoving_distance(xmlZ).value*cosmo.h
    xmlFilter = np.where(xmlZ > 0)[0]
    xmlIDFiltered = [xmlID[k] for k in xmlFilter]
    # Figure out which abell clusters we know that aren't in this catalogue:
    strangeName = np.zeros(len(xmlIDFiltered),dtype=bool)
    for k in range(0,len(strangeName)):
        strangeName[k] = (xmlIDFiltered[k][0:5] != 'ABELL') and \
            (xmlIDFiltered[k][0:4] != 'SSCC') and \
            (xmlIDFiltered[k][0:4] != 'MSCC')
    strangeFilter = np.where(strangeName)[0]
    strangeNames = [xmlIDFiltered[k] for k in strangeFilter]
    strangeNamesAbellNos = [1060,1656,-1,3526,3627,-1,-1,-1,-1,-1,-1]
    xmlAbellN = -np.ones(len(xmlIDFiltered),dtype=int)
    # Identify Abell numbers from the catalogue labels (including handling some
    # special cases):
    for k in range(0,len(xmlIDFiltered)):
        if (xmlIDFiltered[k][0:6] == 'ABELL ') and \
            (xmlIDFiltered[k][6] != 'S'):
            try:
                n = int(xmlIDFiltered[k][6:10])
            except ValueError:
                n = int(xmlIDFiltered[k][6:9])
            xmlAbellN[k] = n
        elif xmlIDFiltered[k] == 'Coma Cluster':
            xmlAbellN[k] = 1656
        elif xmlIDFiltered[k] == 'Centaurus Cluster':
            xmlAbellN[k] = 3526
        elif xmlIDFiltered[k] == 'Norma Cluster':
            xmlAbellN[k] = 3627
        elif xmlIDFiltered[k] == 'Hydra CLUSTER':
            xmlAbellN[k] = 1060
    # Combine with the old catalogue:
    xmlAbellNumbers = xmlAbellN[np.where(xmlAbellN > 0)[0]]
    abellInBoth = np.intersect1d(abell_n,xmlAbellNumbers)
    abellCombined = np.union1d(abell_n,xmlAbellNumbers)
    abellOldOnly = np.setdiff1d(abell_n,xmlAbellNumbers)
    abellNewOnly = np.setdiff1d(xmlAbellNumbers,abell_n)
    abellToAdd = np.intersect1d(abell_n,abellOldOnly,return_indices=True)
    abellOldDist = cosmo.comoving_distance(abell_z).value*cosmo.h
    coordAbellOld = SkyCoord(l=abell_l*u.deg,b=abell_b*u.deg,\
        distance=abellOldDist*u.Mpc,frame='galactic')
    abellOldRA = coordAbellOld.icrs.ra.value
    abellOldDEC = coordAbellOld.icrs.dec.value
    # Combine catalogues:
    combinedAbellRA = np.hstack((xmlRA[xmlFilter],abellOldRA[abellToAdd[1]]))
    combinedAbellDEC = np.hstack((xmlDEC[xmlFilter],abellOldDEC[abellToAdd[1]]))
    combinedAbellDist = np.hstack((xmlDist[xmlFilter],\
        abellOldDist[abellToAdd[1]]))
    combinedAbellZ = np.hstack((xmlZ[xmlFilter],abell_z[abellToAdd[1]]))
    combinedAbellID = xmlIDFiltered + ['A' + str(k) for k in abellOldOnly]
    combinedAbellN = np.hstack((xmlAbellN,abell_n[abellToAdd[1]]))
    coordCombinedAbell = SkyCoord(ra=combinedAbellRA*u.deg,\
        dec=combinedAbellDEC*u.deg,distance=combinedAbellDist*u.Mpc,\
        frame='icrs')
    combinedAbellPos = np.zeros((len(coordCombinedAbell),3))
    combinedAbellPos[:,0] = coordCombinedAbell.icrs.cartesian.x.value
    combinedAbellPos[:,1] = coordCombinedAbell.icrs.cartesian.y.value
    combinedAbellPos[:,2] = coordCombinedAbell.icrs.cartesian.z.value
    return [combinedAbellN,combinedAbellPos,abell_nums]



