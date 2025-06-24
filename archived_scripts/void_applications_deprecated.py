#-------------------------------------------------------------------------------
# ELLIPTICITY FUNCTIONS:

# Compute the inertia tensor:
def get_inertia(points,centre,weights=None):
    if weights is None:
        weights = np.ones(len(points))
    disp = points - centre
    other_inds = [np.setdiff1d(range(0,3),[i]) for i in range(0,3)]
    M_diag = [np.sum(weights[:,None]*disp[:,inds]**2) for inds in other_inds]
    M_off_diag = [-np.sum(weights*disp[:,inds[0]]*disp[:,inds[1]]) 
                  for inds in other_inds]
    M = np.diag(M_diag)
    for inds, Mij in zip(other_inds,M_off_diag):
        M[inds[0],inds[1]] = Mij
        M[inds[1],inds[0]] = Mij
    return M

# Get the ellipticity of a set of points:
def get_ellipticity(points,centre=None,weights=None):
    if centre is None:
        centre = np.mean(points,0)
    # Get inertia tensor:
    M = get_inertia(points,centre,weights=None)
    # Get eigenvalues:
    eigenvalues, eigenvectors = np.linalg.eig(M)
    # Compute ellipticity:
    J1 = np.min(eigenvalues)
    J3 = np.max(eigenvalues)
    return 1.0 - (J1/J3)**(0.25)


#-------------------------------------------------------------------------------
# ELLIPTICITY DISTRIBUTION

# Get long lists of all ellipticities:
def get_all_ellipticities(snapList,snapListRev,boxsize,antihaloCentres):
    ellipticity_list = []
    for ns in range(0,len(snapList)):
        # Load snapshot (don't use the snapshot list, because that will force
        # loading of all snapshot positions, using up a lot of memory, when 
        # we only want to load these one at a time):
        print("Doing sample " + str(ns+1) + " of " + str(len(snapList)))
        snap = pynbody.load(snapList[ns].filename)
        snap_reverse = pynbody.load(snapListRev[ns].filename)
        hr_list = snap_reverse.halos()
        # Sorted indices, to allow correct referencing of particles:
        sorted_indices = np.argsort(snap['iord'])
        reverse_indices = snap_reverse['iord'] # Force loading of reverse 
        # snapshot indices
        print("Sorting complete")
        # Remap positions into correct equatorial co-ordinates:
        positions = tools.remapAntiHaloCentre(snap['pos'],boxsize,
                                              swapXZ  = False,reverse = True)
        print("Positions computed")
        # Get relative positions of particles in each halo, remembering to 
        # account for wrapping:
        print("Computing ellipticities...")
        ellipticities = np.zeros(len(antihaloCentres[ns]))
        for k in tools.progressbar(range(0,len(antihaloCentres[ns]))):
            indices = hr_list[k+1]['iord']
            halo_pos = snapedit.unwrap(positions[sorted_indices[indices],:] - 
                                       antihaloCentres[ns][k],boxsize)
            ellipticities[k] = get_ellipticity(halo_pos,np.array([0]*3))
        ellipticity_list.append(ellipticities)
    return ellipticity_list

# Compute or load ellipticity data:
ellipticity_list = tools.loadOrRecompute(data_folder + "ellipticities.p",
                                         get_all_ellipticities,
                                         snapList,snapListRev,boxsize,
                                         antihaloCentres)

# Lambda-CDM reference:
if not low_memory_mode:
    antihaloCentresUn = [tools.remapAntiHaloCentre(props[5],boxsize,
                                                   swapXZ  = False,
                                                   reverse = True) \
                         for props in ahPropsUn]



ellipticity_list_lcdm = tools.loadOrRecompute(
    data_folder + "ellipticities_lcdm.p",get_all_ellipticities,snapListUn,
    snapListRevUn,boxsize,antihaloCentresUn)

rad_filters = [(rad > 10) & (rad <= 20) for rad in antihaloRadiiUn]

combined_ellipticities = np.hstack([epsilon[filt] 
    for epsilon, filt in zip(ellipticity_list_lcdm, rad_filters)])

# Pure catalogue elipticities only:
ellipticities_shortened = cat300.getShortenedQuantity(ellipticity_list,
    cat300.centralAntihalos)
[eps_mean,eps_std] = cat300.getMeanProperty(ellipticities_shortened,
    void_filter=True)


# Distribution of ellipticities:
plt.clf()
eps_bins = np.linspace(0,0.4,11)

[probLCDM,sigmaLCDM,noInBinsLCDM,inBinsLCDM] = plot.computeHistogram(
    combined_ellipticities,eps_bins,density=True,useGaussianError=True)
[probCat,sigmaCat,noInBinsCat,inBinsCat] = plot.computeHistogram(
    eps_mean,eps_bins,density=True,useGaussianError=True)

fig, ax = plt.subplots()
plt.hist(combined_ellipticities,bins=eps_bins,alpha=0.5,label="$\\Lambda$-CDM",
    density=True,color=seabornColormap[0])
#plt.hist(eps_mean,bins=eps_bins,alpha=0.5,label="Combined Catalogue",
#    density=True,color=seabornColormap[1])

#plot.histWithErrors(probLCDM,sigmaLCDM,eps_bins,ax=ax,color=seabornColormap[0],
#    label="$\\Lambda$-CDM")
plot.histWithErrors(probCat,sigmaCat,eps_bins,ax=ax,color=seabornColormap[1],
    label="Combined Catalogue")

plt.xlabel('Ellipticity')
plt.ylabel('Probability Density')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "ellipticity_distribution.pdf")
plt.show()






def estimate_ellipticity(los_pos,limits=[1e-5,2],npoints=101,R=10,weights=None):
    eps_list = np.linspace(limits[0],limits[1],npoints)
    eps_calc = np.array([get_los_ellipticity_in_ellipse(los_pos,eps,R,
        weights=weights) for eps in eps_list])
    # Look for changes of sign:
    diff = eps_calc - eps_list
    signs = diff[1:]*diff[0:-1]
    sign_change = np.where(signs < 0)[0]
    if len(sign_change) == 0:
        # No changes of sign!
        return eps_list[np.argmin(diff**2)]
    else:
        # Get first change of sign:
        lower_bound = eps_list[sign_change[0]]
        upper_bound = eps_list[sign_change[0]+1]
        return (lower_bound + upper_bound)/2

def solve_ellipticity(los_pos,limits=[1e-5,2],R=10,weights=None,guess=1.0):
    func = lambda x: get_los_ellipticity_in_ellipse(los_pos,x,R,
        weights=weights) - x
    return scipy.optimize.fsolve(func,guess)


# Scatter test:
R=20
#eps_est = estimate_ellipticity(los_pos,R=R)
eps_est = solve_ellipticity(los_pos,R=R)
drange = np.linspace(0,(R/eps_est)*1.1,101)

plt.clf()
plt.scatter(los_pos[:,1],los_pos[:,0],marker='.')
plt.plot(drange,np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k',
    label="Ellipse, $R =  " + ("%.2g" % R) + ", \\epsilon = " + 
    ("%.2g" % eps_est) + "$")
plt.plot(drange,-np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k')
plt.xlabel('d [$\\mathrm{Mpc}h^{-1}$]')
plt.ylabel('z [$\\mathrm{Mpc}h^{-1}$]')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "ellipticity_scatter.pdf")
plt.show()



def verify_los_pos(los_pos):
    if len(los_pos.shape) != 2:
        raise Exception("los_pos must have two dimensions")
    if los_pos.shape[1] != 2:
        raise Exception("los_pos shape must have size 2")



def get_points_in_ellipse(los_pos,eps,R):
    verify_los_pos(los_pos)
    z = los_pos[:,0]
    d = los_pos[:,1]
    return (d <= R/eps) & (z**2 <= R**2 - eps**2*d**2)

def get_los_pos(pos,los,boxsize):
    los_unit = los/np.sqrt(np.sum(los**2))
    pos_rel = snapedit.unwrap(pos - los,boxsize)
    z = np.dot(pos_rel,los_unit)
    d = np.sqrt(np.sum(pos_rel**2,1) - z**2)
    return np.vstack((z,d)).T

def get_los_ellipticity(los_pos,weights=None):
    verify_los_pos(los_pos)
    z = los_pos[:,0]
    d = los_pos[:,1]
    if weights is None:
        weights = np.ones(los_pos.shape[0])
    return np.sqrt(2*np.sum(z**2*weights)/np.sum(d**2*weights))

def get_los_ellipticity_in_ellipse(los_pos,eps,R,weights=None):
    verify_los_pos(los_pos)
    if weights is None:
        weights = np.ones(los_pos.shape[0])
    filt = get_points_in_ellipse(los_pos,eps,R)
    return get_los_ellipticity(los_pos[filt],weights=weights[filt])

# Check ellipticity calculation:
ns = 0
k = 0
snap = pynbody.load(snapList[ns].filename)
snap_reverse = pynbody.load(snapListRev[ns].filename)
hr_list = snap_reverse.halos()
indices = hr_list[k+1]['iord']
positions = tools.remapAntiHaloCentre(snap['pos'],boxsize,swapXZ  = False,
    reverse = True)
pos = snapedit.unwrap(positions[sorted_indices[indices],:],boxsize)
los = antihaloCentres[ns][0]


los_pos = get_los_pos(pos,los,boxsize)

eps_list = np.linspace(0,2,101)

eps_calc = np.array([get_los_ellipticity_in_ellipse(los_pos,eps,10) 
    for eps in eps_list])

eps_calc = np.array([get_los_ellipticity_in_ellipse(los_pos,eps,20,
    weights=1.0/(2.0*np.pi*los_pos[:,1])) 
    for eps in eps_list])

plt.clf()
plt.plot(eps_list,eps_calc,linestyle='-',color=seabornColormap[0],
    label="Calculated ellipticity")
plt.plot(eps_list,eps_list,linestyle='--',color=seabornColormap[0],
    label="Equal ellipticities")
plt.xlabel('Cut ellipticity')
plt.ylabel('Calculated Ellipticity')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "ellipticity_test.pdf")
plt.show()


# Plot comparing LCDM with out voids:
upper_dist = 20
bins_z = np.linspace(0,upper_dist,21)
bins_d = np.linspace(0,upper_dist,21)
cell_volumes = np.outer(np.diff(bins_z),np.diff(bins_d))
hist_lcdm = np.histogramdd(stacked_particles_lcdm_abs,bins=[bins_z,bins_d],
                           density=False,weights = 1.0/\
                           (2*np.pi*stacked_particles_lcdm_abs[:,1]))
count_lcdm = len(stacked_particles_lcdm_abs)
num_voids_lcdm = np.sum([len(x) for x in los_list_lcdm])

hist_borg = np.histogramdd(stacked_particles_borg_abs,bins=[bins_z,bins_d],
                           density=False,weights = 1.0/\
                           (2*np.pi*stacked_particles_borg_abs[:,1]))
count_borg = len(stacked_particles_borg_abs)
num_voids_borg = np.sum([len(x) for x in los_list_borg]) # Not the actual number
    # but the effective number being stacked, so the number of voids multiplied
    # by the number of samples.


nmean = len(snapList[0])/(boxsize**3)

plt.clf()
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))
#ax[0].hist2d(stacked_particles_lcdm_abs[:,1],
#           stacked_particles_lcdm_abs[:,0],
#           bins=[np.linspace(0,60,31),np.linspace(0,60,31)],density=True,
#           cmap="Blues",
#           weights=(1.0/(2*np.pi*stacked_particles_lcdm_abs[:,1])))

#im = ax[1].hist2d(stacked_particles_borg_abs[:,1],
#           stacked_particles_borg_abs[:,0],
#           bins=[np.linspace(0,60,31),np.linspace(0,60,31)],density=True,
#           cmap="Blues",weights=
#           (1.0/((len(snapList)*2*np.pi*stacked_particles_borg_abs[:,1]))))

im1 = ax[0].imshow(hist_lcdm[0]/(2*cell_volumes*num_voids_lcdm*nmean),
                   cmap='PuOr_r',vmin=0,vmax = 2,
                   extent=(0,upper_dist,0,upper_dist),origin='lower')
im2 = ax[1].imshow(hist_borg[0]/(2*cell_volumes*num_voids_borg*nmean),
                   cmap='PuOr_r',vmin=0,vmax = 2,
                   extent=(0,upper_dist,0,upper_dist),origin='lower')

#ax.plot(drange,np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k',
#    label="Ellipse, $R =  " + ("%.2g" % R) + ", \\epsilon = " + 
#    ("%.2g" % eps_est) + "$")
#ax.plot(drange,-np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k')

#Rvals = [10,20,30,40,50,60]
Rvals = [5,10,15,20]
titles = ['$\\Lambda$-CDM Simulations','BORG Catalogue']
for axi, title in zip(ax,titles):
    axi.set_xlabel('d (Perpendicular distance) [$\\mathrm{Mpc}h^{-1}$]',
                   fontsize=fontsize,fontfamily=fontfamily)
    axi.set_ylabel('z (LOS distance)[$\\mathrm{Mpc}h^{-1}$]',
                   fontsize=fontsize,fontfamily=fontfamily)
    for r in Rvals:
        draw_ellipse(axi,r,1.0)
    axi.set_xlim([0,upper_dist])
    axi.set_ylim([0,upper_dist])
    axi.set_aspect('equal')
    axi.set_title(title,fontsize=fontsize,fontfamily=fontfamily)
    #axi.legend(frameon=False)

# Remove y labels on axis 2:
ax[1].yaxis.label.set_visible(False)
ax[1].yaxis.set_major_formatter(NullFormatter())
ax[1].yaxis.set_minor_formatter(NullFormatter())
ax[0].xaxis.get_major_ticks()[4].set_visible(False)

fig.colorbar(im1, ax=ax.ravel().tolist(),shrink=0.9,
    label='(Tracer density)/(Mean Density)')
plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.1,right=0.75,bottom=0.15,
                    top=0.95)
plt.savefig(figuresFolder + "ellipticity_scatter_comparison.pdf")
plt.show()

#-------------------------------------------------------------------------------

# With all particles:
#los_lcdm = los_list_lcdm
#los_borg = los_list_borg
#filename = "ellipticity_scatter_comparison_reff.pdf"
#density_unit = "relative"
# Void particles only:
los_lcdm = los_list_void_only_lcdm
los_borg = los_list_void_only_borg
filename = "ellipticity_scatter_comparison_reff_void_only.pdf"
density_unit = "probability"

voids_used_lcdm = [np.array([len(x) for x in los]) > 0 for los in los_lcdm]
voids_used_borg = [np.array([len(x) for x in los]) > 0 for los in los_borg]
# Filter out any unused voids as they just cause problems:
los_lcdm = [ [x for x in los if len(x) > 0] for los in los_lcdm]
los_borg = [ [x for x in los if len(x) > 0] for los in los_borg]

# Stacked Profile with rescaled voids:
upper_dist_reff = 2
bins_z_reff = np.linspace(0,upper_dist_reff,41)
bins_d_reff = np.linspace(0,upper_dist_reff,41)
bin_z_centres = plot.binCentres(bins_z_reff)
bin_d_centres = plot.binCentres(bins_d_reff)
cell_volumes_reff = np.outer(np.diff(bins_z_reff),np.diff(bins_d_reff))

# Get void effective radii:
void_radii_lcdm = [rad[filt] 
                   for rad, filt in zip(antihaloRadiiUn,voids_used_lcdm)]
void_radii_borg = cat300.getMeanProperty("radii",void_filter=True)[0]
# Express los co-ords in units of reff:
los_list_reff_lcdm = [
    [los/rad for los, rad in zip(all_los,all_radii)] 
    for all_los, all_radii in zip(los_lcdm,void_radii_lcdm)]
los_list_reff_borg = [
    [los/rad for los, rad in zip(all_los,void_radii_borg)] 
    for all_los in los_borg]

# Stack all the particles:
stacked_particles_lcdm_reff = np.vstack([np.vstack(los_list) 
    for los_list in los_list_reff_lcdm ])
stacked_particles_borg_reff = np.vstack([np.vstack(los_list) 
    for los_list in los_list_reff_borg ])
stacked_particles_reff_lcdm_abs = np.abs(stacked_particles_lcdm_reff)
stacked_particles_reff_borg_abs = np.abs(stacked_particles_borg_reff)


stacked_particles_r_lcdm_reff = np.sqrt(np.sum(stacked_particles_lcdm_reff**2,1))

# Volume weights for each particle:
v_weight_lcdm = [
    [rad**3*np.ones(len(los)) for los, rad in zip(all_los,all_radii)] 
    for all_los, all_radii in zip(los_lcdm,void_radii_lcdm)]
v_weight_lcdm = np.hstack([np.hstack(rad) for rad in v_weight_lcdm])
v_weight_borg = [
    [rad**3*np.ones(len(los)) for los, rad in zip(all_los,void_radii_borg)] 
    for all_los in los_borg]
v_weight_borg = np.hstack([np.hstack(rad) for rad in v_weight_borg])


# Histograms to get the density:
hist_lcdm_reff = np.histogramdd(stacked_particles_reff_lcdm_abs,
                           bins=[bins_z_reff,bins_d_reff],
                           density=False,weights = 1.0/\
                           (2*np.pi*v_weight_lcdm*
                           stacked_particles_reff_lcdm_abs[:,1]))
count_lcdm = len(stacked_particles_reff_lcdm_abs)
num_voids_lcdm = np.sum([np.sum(x) for x in voids_used_lcdm])

hist_borg_reff = np.histogramdd(stacked_particles_reff_borg_abs,
                           bins=[bins_z_reff,bins_d_reff],
                           density=False,weights = 1.0/\
                           (2*v_weight_borg*np.pi*
                           stacked_particles_reff_borg_abs[:,1]))
count_borg = len(stacked_particles_reff_borg_abs)
num_voids_borg = np.sum([np.sum(x) for x in voids_used_borg]) # Not the actual number
    # but the effective number being stacked, so the number of voids multiplied
    # by the number of samples.
nmean = len(snapList[0])/(boxsize**3)


plt.clf()
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))
#ax[0].hist2d(stacked_particles_lcdm_abs[:,1],
#           stacked_particles_lcdm_abs[:,0],
#           bins=[np.linspace(0,60,31),np.linspace(0,60,31)],density=True,
#           cmap="Blues",
#           weights=(1.0/(2*np.pi*stacked_particles_lcdm_abs[:,1])))

#im = ax[1].hist2d(stacked_particles_borg_abs[:,1],
#           stacked_particles_borg_abs[:,0],
#           bins=[np.linspace(0,60,31),np.linspace(0,60,31)],density=True,
#           cmap="Blues",weights=
#           (1.0/((len(snapList)*2*np.pi*stacked_particles_borg_abs[:,1]))))
if density_unit == "relative":
    field_lcdm = hist_lcdm_reff[0]/(2*cell_volumes_reff*num_voids_lcdm*nmean)
    field_borg = hist_borg_reff[0]/(2*cell_volumes_reff*num_voids_borg*nmean)
    im1 = ax[0].imshow(
        field_lcdm,
        cmap='PuOr_r',vmin=0,vmax = 2,
        extent=(0,upper_dist_reff,0,upper_dist_reff),origin='lower')
    im2 = ax[1].imshow(
        field_borg,
        cmap='PuOr_r',vmin=0,vmax = 2,
        extent=(0,upper_dist_reff,0,upper_dist_reff),origin='lower')
elif density_unit == "absolute":
    field_lcdm = hist_lcdm_reff[0]/(2*cell_volumes_reff*num_voids_lcdm)
    field_borg = hist_borg_reff[0]/(2*cell_volumes_reff*num_voids_borg)
    im1 = ax[0].imshow(field_lcdm,
                       cmap='Blues',vmin=0,vmax = None,
                       extent=(0,upper_dist_reff,0,upper_dist_reff),
                       origin='lower')
    im2 = ax[1].imshow(field_borg,
                       cmap='Blues',vmin=0,vmax = None,
                       extent=(0,upper_dist_reff,0,upper_dist_reff),
                       origin='lower')
elif density_unit == "probability":
    field_lcdm = hist_lcdm_reff[0]/(2*count_lcdm*cell_volumes_reff)
    field_borg = hist_borg_reff[0]/(2*count_borg*cell_volumes_reff)
    im1 = ax[0].imshow(field_lcdm,cmap='Blues',vmin=0,vmax = 1e-4,
                       extent=(0,upper_dist_reff,0,upper_dist_reff),
                       origin='lower')
    im2 = ax[1].imshow(field_borg,cmap='Blues',vmin=0,vmax = 1e-4,
                       extent=(0,upper_dist_reff,0,upper_dist_reff),
                       origin='lower')
else:
    raise Exception("Unknown density_unit")

contours = True
countour_list = [1e-6,1e-5,2.5e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,2e-4]
if contours:
    CS = ax[0].contour(bin_d_centres,bin_z_centres,field_lcdm,
        levels=countour_list)
    ax[0].clabel(CS, inline=True, fontsize=10)
    CS = ax[1].contour(bin_d_centres,bin_z_centres,field_borg,
        levels=countour_list)
    ax[1].clabel(CS, inline=True, fontsize=10)
#ax.plot(drange,np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k',
#    label="Ellipse, $R =  " + ("%.2g" % R) + ", \\epsilon = " + 
#    ("%.2g" % eps_est) + "$")
#ax.plot(drange,-np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k')

#Rvals = [10,20,30,40,50,60]
Rvals = [1,2]
titles = ['$\\Lambda$-CDM Simulations','BORG Catalogue']
for axi, title in zip(ax,titles):
    axi.set_xlabel('$d/R_{\\mathrm{eff}}$ (Perpendicular distance)',
                   fontsize=fontsize,fontfamily=fontfamily)
    axi.set_ylabel('$z/R_{\\mathrm{eff}}$ (LOS distance)',
                   fontsize=fontsize,fontfamily=fontfamily)
    for r in Rvals:
        draw_ellipse(axi,r,1.0)
    axi.set_xlim([0,upper_dist_reff])
    axi.set_ylim([0,upper_dist_reff])
    axi.set_aspect('equal')
    axi.set_title(title,fontsize=fontsize,fontfamily=fontfamily)
    #axi.legend(frameon=False)

# Remove y labels on axis 2:
ax[1].yaxis.label.set_visible(False)
ax[1].yaxis.set_major_formatter(NullFormatter())
ax[1].yaxis.set_minor_formatter(NullFormatter())
ax[0].xaxis.get_major_ticks()[4].set_visible(False)

if density_unit == "absolute":
    colorbar_title = "Average Tracer density [$h^{3}\\mathrm{MPc}^{-3}$]"
elif density_unit == "relative":
    colorbar_title = '(Tracer density)/(Mean Density)'
elif density_unit == "probability":
    colorbar_title = 'Probability Density [$h^{3}\\mathrm{MPc}^{-3}$]'

fig.colorbar(im1, ax=ax.ravel().tolist(),shrink=0.9,
    label=colorbar_title)
plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.1,right=0.75,bottom=0.15,top=0.95)
plt.savefig(figuresFolder + filename)
plt.show()


def get_los_stack_field(hist,num_voids,density_unit = "probability"):
    # Extract the bins:
    bins_z_reff = hist[1][0]
    bins_d_reff = hist[1][1]
    # Get cell volumes:
    cell_volumes_reff = np.outer(np.diff(bins_z_reff),np.diff(bins_d_reff))
    if density_unit == "relative":
        field = hist[0]/(2*cell_volumes_reff*num_voids*nmean)
    elif density_unit == "absolute":
        field = hist[0]/(2*cell_volumes_reff*num_voids)
    elif density_unit == "probability":
        field = hist[0]/(2*num_voids*cell_volumes_reff)
    return field

#-------------------------------------------------------------------------------
# FITTING CONTOURS

import contourpy

# To get the contour data:
CG = contourpy.contour_generator(bin_d_centres,bin_z_centres,field_lcdm)
contours = CG.lines(countour_list[0])

# We can then fit an ellipse to this.

def data_model(d,R,eps):
    if np.isscalar(d):
        if d >= R/eps:
            return 0.0
        else:
            return np.sqrt(R**2 - eps**2*d**2)
    else:
        nz = np.where(d <= R/eps)
        result = np.zeros(d.shape)
        result[nz] = np.sqrt(R**2 - eps**2*d[nz]**2)
        return result

# A Lambda-cdm data model where we compute the expected epsilon
# from the value of Omega_m:
def data_model_lcdm(d,R,Om0,cosmo_fid=None,zfid = 0.01529,Dafid=None):
    # Get cosmology
    if cosmo_fid is None:
        cosmo_fid = astropy.cosmology.FlatLambdaCDM(H0=100*0.7,Om0=0.3)
    Om0_fid = cosmo_fid.Om0
    H0fid = cosmo_fid.H0.value
    cosmo_test = astropy.cosmology.FlatLambdaCDM(H0=H0fid,Om0=Om0)
    # Get the ratio:
    Hz = H0fid*np.sqrt(Om0*(1 + zfid)**3 + 1.0 - Om0)
    Hzfid = H0fid*np.sqrt(Om0_fid*(1 + zfid)**3 + 1.0 - Om0_fid)
    Da = cosmo_test.angular_diameter_distance(zfid).value
    if Dafid is None:
        Dafid = cosmo_fid.angular_diameter_distance(zfid).value
    eps = Hz*Da/(Hzfid*Dafid)
    return data_model(d,R,eps)

def residual(z,d,params):
    R = params[0]
    eps = params[1]
    return z - data_model(d,R,eps)

zi = contours[0][:,0]
di = contours[0][:,1]
R_bounds = [0,2]
eps_bounds = [0,2]
lower_bounds = np.array([R_bounds[0],eps_bounds[0]])
upper_bounds = np.array([R_bounds[1],eps_bounds[1]])
ls_guess = scipy.optimize.least_squares(
    lambda x: residual(zi,di,x),np.array([1.0,1.0]),
    bounds=(lower_bounds,upper_bounds))

# Check plot:

plt.clf()
fig, ax = plt.subplots()
ax.scatter(di,zi,marker='x',color='k')
draw_ellipse(ax,ls_guess.x[0],ls_guess.x[1])
ax.set_xlabel('$d/R_{\\mathrm{eff}}$ (Perpendicular distance)',
               fontsize=fontsize,fontfamily=fontfamily)
ax.set_ylabel('$z/R_{\\mathrm{eff}}$ (LOS distance)',
               fontsize=fontsize,fontfamily=fontfamily)
plt.savefig(figuresFolder + "least_squares_test.pdf")
plt.show()


# Good for an estimate, but we really want to get errors on this, so we might
# need to setup a likelihood and do an MCMC on the two parameters.

# Error estimate assuming that the error is the distance to the closest point 
# along each dimension:
def get_interpolation_error_estimate(contour,di,zi,tol=1e-10):
    errors = np.zeros(contour.shape)
    num_points = contour.shape[0]
    for k in range(0,num_points):
        dist_d = np.abs(di - contour[k,0])
        dist_z = np.abs(zi - contour[k,1])
        closest_d = np.min(dist_d[dist_d > tol])
        closest_z = np.min(dist_z[dist_z > tol])
        errors[k,:] = np.array([closest_d,closest_z])
    return errors


# Now setup an MCMC:

errors = get_interpolation_error_estimate(contours[0],
                                          bin_d_centres,bin_z_centres)
yerr = errors[:,1]

# Log likelihood function:
def log_likelihood(theta, x, y, yerr):
    R, eps = theta
    model = data_model(x, R, eps)
    sigma2 = yerr**2
    return -0.5 * np.sum( (y - model)**2/sigma2 + np.log(sigma2) )

# Priors:
def log_prior(theta):
    R, eps = theta
    if (0 <= R < 2.0) & (0 <= eps < 2.0):
        return 0.0
    else:
        return -np.inf

def log_probability(theta,x,y,yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


# ML estimate:
x = contours[0][:,0]
y = contours[0][:,1]
nll = lambda *args: -log_likelihood(*args)
initial = np.array([1.0,1.0])
soln = scipy.optimize.minimize(nll, initial, 
    args=(x, y, yerr))
R_ml, eps_ml = soln.x

print("Maximum likelihood estimates:")
print("R = {0:.3f}".format(R_ml))
print("\\epsilon = {0:.3f}".format(eps_ml))

plt.clf()
x0 = np.linspace(0,2,101)
fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
ax.plot(x0, data_model(x0,ls_guess.x[0],ls_guess.x[1]), "--k", label="LS")
ax.plot(x0, data_model(x0,R_ml,eps_ml), ":k", label="ML")
ax.legend(fontsize=fontsize)
ax.set_xlim(0,2)
ax.set_ylim(0,2)
ax.set_xlabel('$d/R_{\\mathrm{eff}}$ (Perpendicular distance)',
               fontsize=fontsize,fontfamily=fontfamily)
ax.set_ylabel('$z/R_{\\mathrm{eff}}$ (LOS distance)',
               fontsize=fontsize,fontfamily=fontfamily)
plt.savefig(figuresFolder + "ml_test.pdf")
plt.show()


# MCMC run:

import emcee

pos = soln.x + 1e-4*np.random.randn(32,2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x, y, yerr)
)
sampler.run_mcmc(pos, 5000, progress=True)

tau = sampler.get_autocorr_time()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)


# Corner plot:
import corner

plt.clf()
fig = corner.corner(flat_samples, labels=["$R$","$\\epsilon$"])
fig.suptitle("$\\Lambda$-CDM Simulations Contour Ellipticity")
plt.savefig(figuresFolder + "corner_plot_R_eps.pdf")



def mcmc_contour_ellipticity(field,d_vals,z_vals,level,guess = 'ML',
                             initial=np.array([1.0,1.0]),nwalkers=32,disp=1e-4,
                             n_mcmc = 5000):
    # Fit the contour:
    CG = contourpy.contour_generator(d_vals,z_vals,field)
    contours = CG.lines(level)
    if len(contours) < 1:
        raise Exception("Failed to find contour at level " + ("%.2g" % level))
    # Estimate errors for the contour:
    errors = get_interpolation_error_estimate(contours[0],d_vals,z_vals)
    # Data to fit:
    yerr = errors[:,1]
    x = contours[0][:,0]
    y = contours[0][:,1]
    ndims = 2
    # Get the initial guess:
    if guess == 'ML':
        nll = lambda *args: -log_likelihood(*args)
        soln = scipy.optimize.minimize(nll, initial, args=(x, y, yerr))
        pos = soln.x + disp*np.random.randn(nwalkers,ndims)
    elif guess == 'initial':
        pos = initial + disp*np.random.randn(nwalkers,ndims)
    else:
        raise Exception("Guess not recognised.")
    # Setup and run the sampler:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndims, log_probability, args=(x, y, yerr)
    )
    sampler.run_mcmc(pos, n_mcmc, progress=True)
    # Filter the MCMC samples to account for correlation:
    tau = sampler.get_autocorr_time()
    tau_max = np.max(tau)
    flat_samples = sampler.get_chain(discard=int(3*tau_max), 
                                     thin=int(tau_max/2), flat=True)
    return flat_samples

flat_samples_borg = mcmc_contour_ellipticity(field_borg,
    bin_d_centres,bin_z_centres,1e-5)

flat_samples_lcdm = mcmc_contour_ellipticity(field_lcdm,
    bin_d_centres,bin_z_centres,1e-5)



plt.clf()
fig = corner.corner(flat_samples_lcdm, labels=["$R$","$\\epsilon$"])
fig.suptitle("$\\Lambda$-CDM Simulations Contour Ellipticity")
plt.savefig(figuresFolder + "corner_plot_R_eps.pdf")
plt.show()



plt.clf()
fig = corner.corner(flat_samples_borg, labels=["$R$","$\\epsilon$"])
fig.suptitle("Combined Catalogue Contour Ellipticity")
plt.savefig(figuresFolder + "corner_plot_R_eps_borg.pdf")
plt.show()


# Test for the conditions void stacks:

# Get conditioned regions:
from void_analysis.simulation_tools import get_mcmc_supervolume_densities

deltaMCMCList = tools.loadOrRecompute(data_folder2 + "delta_list.p",
                                      get_mcmc_supervolume_densities,
                                      snapList,r_sphere=135)

# MAP value of the density of the local super-volume:
from void_analysis.simulation_tools import get_map_from_sample

deltaMAPBootstrap = scipy.stats.bootstrap((deltaMCMCList,),\
    get_map_from_sample,confidence_level = 0.68,vectorized=False,\
    random_state=1000)
deltaMAPInterval = deltaMAPBootstrap.confidence_interval

# Get comparable density regions:


# Select random centres in the random simulations, and compute their
# density contrast:
[randCentres,randOverDen] = tools.loadOrRecompute(\
    data_folder2 + "random_centres_and_densities.p",\
    simulation_tools.get_random_centres_and_densities,rSphere,snapListUn,
    _recomputeData=False)



comparableDensityMAP = [(delta <= deltaMAPInterval[1]) & \
    (delta > deltaMAPInterval[0]) for delta in randOverDen]
centresToUse = [randCentres[comp] for comp in comparableDensityMAP]
deltaToUse = [randOverDen[ns][comp] \
    for ns, comp in zip(range(0,len(snapList)),comparableDensityMAP)]

rSep = 2*135
indicesUnderdenseNonOverlapping = simulation_tools.getNonOverlappingCentres(
    centresToUse,rSep,boxsize,returnIndices=True)

centresUnderdenseNonOverlapping = [centres[ind] \
    for centres,ind in zip(centresToUse,indicesUnderdenseNonOverlapping)]

densityListUnderdenseNonOverlapping = [density[ind] \
    for density, ind in zip(comparableDensityMAP,\
    indicesUnderdenseNonOverlapping)]

densityUnderdenseNonOverlapping = np.hstack(
    densityListUnderdenseNonOverlapping)



# Get the stacks of voids:
regionAndVoidDensityConditionDict = tools.loadPickle(\
    data_folder2 + "regionAndVoidDensityCondition_stack.p")

# Which simulation each centre belongs to:
ns_list = np.hstack([np.array([ns for k in range(0,len(centre))],dtype=int) 
    for ns, centre in zip(range(0,len(snapList)),
                          centresUnderdenseNonOverlapping)])

# Get histograms for each of the conditioned stacks:

recompute=False
field_list = []
histogram_list = []
ns_last = -1
snap = None
snap_reverse=None
hr_list = None
sorted_indices = None
reverse_indices=None
positions=None
for k in tools.progressbar(range(0,len(ns_list))):
    ns = ns_list[k]
    if (ns != ns_last) and (recompute):
        ns_last = ns
        snap = tools.getPynbodySnap(snapList[ns].filename)
        snap_reverse = pynbody.load(snapListRev[ns].filename)
        hr_list = snap_reverse.halos()
        sorted_indices = np.argsort(snap['iord'])
        reverse_indices = snap_reverse['iord'] # Force loading of reverse 
            # snapshot indices
        print("Sorting complete")
        positions = tools.remapAntiHaloCentre(snap['pos'],boxsize,
                                              swapXZ  = False,reverse = True)
    void_indices = regionAndVoidDensityConditionDict['indices'][k]
    filter_list = np.isin(np.arange(0,len(antihaloRadiiUn[ns])),
                          regionAndVoidDensityConditionDict['indices'][k])
    los_list_void_only = tools.loadOrRecompute(data_folder2 + "region_los_" + 
        str(k) + ".p",get_los_pos_for_snapshot,snap,snap_reverse,
        antihaloCentres[ns][void_indices],
        antihaloRadii[ns][void_indices],filter_list=None,
        _recomputeData=recompute,void_indices=void_indices,dist_max=60,rmin=10,
        rmax=20,all_particles=False,sorted_indices=sorted_indices,
        reverse_indices=reverse_indices,positions = positions,hr_list=hr_list)
    void_radii = antihaloRadiiUn[ns][void_indices]
    v_weight_lcdm = np.hstack([rad**3*np.ones(len(los))
        for los, rad in zip(los_list_void_only,void_radii)])
    los_list_reff = [los/rad 
        for los, rad in zip(los_list_void_only,void_radii)]
    stacked_particles_reff = np.vstack(los_list_reff)
    stacked_particles_reff_abs = np.abs(stacked_particles_reff)
    count_lcdm = len(stacked_particles_reff_abs)
    hist_reff = np.histogramdd(stacked_particles_reff_abs,
                           bins=[bins_z_reff,bins_d_reff],
                           density=False,weights = 1.0/\
                           (2*np.pi*v_weight_lcdm*
                           stacked_particles_reff_abs[:,1]))
    field = hist_reff[0]/(2*count_lcdm*cell_volumes_reff)
    field_list.append(field)
    histogram_list.append(hist_reff)

# Get samples:

sample_list = []
for k in range(0,len(ns_list)):
    try:
        mcmc_samples = mcmc_contour_ellipticity(field_list[k],
            bin_d_centres,bin_z_centres,1e-5)
    except:
        mcmc_samples = None
    sample_list.append(mcmc_samples)

def get_mcmc_samples_for_fields(field_list,level,bin_d_centres,bin_z_centres):
    sample_list = []
    for k in range(0,len(field_list)):
        try:
            mcmc_samples = mcmc_contour_ellipticity(field_list[k],
                bin_d_centres,bin_z_centres,1e-5)
        except:
            mcmc_samples = None
        sample_list.append(mcmc_samples)
    return sample_list

sample_list = tools.loadOrRecompute(
    data_folder + "mcmc_samples_conditioned_voids.p",
    get_mcmc_samples_for_fields,field_list,level,bin_d_centres,bin_z_centres)

clean_sample_list = [x for x in sample_list if x is not None]

means = np.array([np.mean(x,0) for x in clean_sample_list])

borg_mean = np.mean(flat_samples_borg,0)

plt.clf()
plt.hist(means[:,1],alpha=0.5,color=seabornColormap[0],
    bins=np.linspace(0.5,1.5,21),density=True,
    label='Conditioned\n$\\Lambda$-CDM samples')
plt.axvline(borg_mean[1],linestyle='--',color='grey',label='Combined Catalogue')
plt.xlabel('Ellipticity, $\\epsilon$')
plt.ylabel('Probability Density')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "lcdm_ellipticity_distribution_histogram.pdf")
plt.show()

# Test:
plot_los_void_stack(field_list[0],contour_list=[1e-5],Rvals=[1,2],vmax=1e-4,
    savename = figuresFolder + "conditioned_stack_test.pdf",colorbar=True,
    colorbar_title = 'Probability Density [$h^{3}\\mathrm{MPc}^{-3}$]')




#-------------------------------------------------------------------------------
# WRONG COSMO SIMULATIONS

plus_sim = pynbody.load(
    "new_chain/sample10000/gadget_wrong_cosmo_forward_512/" + 
    "snapshot_domegam_plus_000")
plus_sim_reverse = pynbody.load(
    "new_chain/sample10000/gadget_wrong_cosmo_reverse_512/" + 
    "snapshot_domegam_plus_000")

ns_ref = 9

regular_sim = snapList[ns_ref]
regular_sim_reverse = snapListRev[ns_ref]

final_cat = cat300.get_final_catalogue(void_filter=True)
halo_indices = [-np.ones(len(final_cat),dtype=int) 
    for ns in range(0,len(snapList))]
for ns in range(0,len(snapList)):
    have_void = final_cat[:,ns] >= 0
    halo_indices[ns][have_void] = \
        cat300.indexListShort[ns][final_cat[have_void,ns]-1]

halo_indices = np.array(halo_indices).T

clean_indices = halo_indices[halo_indices[:,ns_ref] > 0,ns_ref]

# Need to cross-reference the halo catalogue for this to work:
antihalos_reg = regular_sim_reverse.halos()
antihalos_plus = plus_sim_reverse.halos()

bridge = pynbody.bridge.OrderBridge(plus_sim,regular_sim,monotonic=False)

bridge_reverse = pynbody.bridge.OrderBridge(
    regular_sim_reverse,plus_sim_reverse,monotonic=False)

match = bridge_reverse.match_catalog(min_index=1,max_index=15000,
                                     groups_1 = antihalos_reg,
                                     groups_2 = antihalos_plus)[1:]

ah_props_plus = pickle.load(open(plus_sim.filename + ".AHproperties.p","rb"))

antihalo_radii_plus = ah_props_plus[7]
antihalo_centres_plus = tools.remapAntiHaloCentre(
    ah_props_plus[5],boxsize,swapXZ  = False,reverse = True)

# Get indices for the voids that we have matched for this sample:
successful_match = match[clean_indices] >= 0
matched_indices = match[clean_indices][successful_match]-1

# Get the change in radii:
radii_plus = antihalo_radii_plus[matched_indices]
radii_reg = antihaloRadii[ns_ref][clean_indices[successful_match]]
radii_diff = radii_plus - radii_reg

# Compare with the change in radii between MCMC samples:
radii_all_samples = cat300.getAllProperties("radii",void_filter=True)
radii_cleaned = radii_all_samples[halo_indices[:,ns_ref] > 0,:]
radii_cleaned_mean = np.nanmean(radii_cleaned,1)
radii_cleaned_std = np.nanstd(radii_cleaned,1)

# Get the change in centres:
centres_plus = antihalo_centres_plus[matched_indices,:]
centres_reg = antihaloCentres[ns_ref][clean_indices[successful_match],:]
centres_mean = cat300.getMeanCentres(void_filter=True)[halo_indices[:,ns_ref] > 0,:]

# Compare to the change in centres between MCMC samples:
centres_all_samples = cat300.getAllCentres(void_filter=True)
centres_cleaned = centres_all_samples[:,halo_indices[:,ns_ref] > 0,:]
centres_cleaned_mean = np.nanmean(centres_cleaned,0)
centres_cleaned_std = np.nanstd(centres_cleaned,0)
dist_cleaned = np.sqrt(np.sum((centres_cleaned - centres_cleaned_mean)**2,2))
dist_cleaned_mean = np.nanmean(dist_cleaned,0)
dist_cleaned_std = np.nanstd(dist_cleaned,0)

centre_diff = centres_plus - centres_reg
centre_dist = np.sqrt(np.sum(centre_diff**2,1))


# Plots:

bins_rad = np.linspace(-0.5,0.5,21)
bins_dist = np.linspace(0,1.5,21)

import seaborn

plt.clf()
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))
ax[0].hist(radii_diff/radii_cleaned_std,bins=bins_rad,alpha=0.5,
           color=seabornColormap[0])
ax[1].hist(centre_dist/dist_cleaned_std,bins=bins_dist,alpha=0.5,
           color=seabornColormap[0])

#seaborn.kdeplot(radii_diff/radii_cleaned_std,alpha=0.5,color=seabornColormap[0],
#            ax=ax[0])
#seaborn.kdeplot(centre_dist/dist_cleaned_std,alpha=0.5,color=seabornColormap[0],
#            ax=ax[1])
ax[0].set_xlabel('$\\mathrm{\\Delta}r_{\\mathrm{eff}}/\sigma_{r_{\\mathrm{eff}}}$')
ax[1].set_xlabel('$\\mathrm{\\Delta}d/\\sigma_{d}$')
ax[0].set_ylabel('Number of Voids')
ax[1].set_ylabel('Number of Voids')
ax[0].set_xlim([-0.5,0.5])
ax[1].set_xlim([0,1.5])
plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.1,right=0.95,bottom=0.15,
                    top=0.9)
ax[0].set_title('Radii change')
ax[1].set_title('Centre Displacement')
plt.savefig(figuresFolder + "wrong_cosmo_displacements.pdf")
plt.show()



