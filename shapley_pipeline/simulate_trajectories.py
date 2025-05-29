import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from crpropa import *
from astropy import units as u
from astropy.coordinates import SkyCoord
from analyse_trajectories import visualize_3D_shapley
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=67.6, Om0=0.286)

def setupInnerSimulation(seed):
    '''
    This function setups the simulation rules for backtracking EHECR events
    inside our Galaxy
    '''
    # magnetic field setup
    B = JF12Field()
    B.randomStriated(seed)
    B.randomTurbulent(seed)

    # simulation setup
    sim = ModuleList()
    sim.add(PropagationCK(B, 1e-4, 0.1 * parsec, 100 * parsec))
    obs = Observer()
    obs.add(ObserverSurface(Sphere(Vector3d(0), 20 * kpc)))
    sim.add(obs)

    return sim, obs

def setupOuterSimulation():
    '''A random realization of a turbulent field with a Kolmogorov power spectrum on (100 pc???)50kpc - 1 Mpc lengthscales 
       and an RMS field strength of 1 nG. The field is stored on a 512^3 grid with (20 pc???) 10kpc grid spacing, and thus 
       has an extent of (512*4kpc)^3 > Lmax. The field is by default periodically repeated in space to cover an arbitrary volume.
    '''
    #Simpler model
    #Analytical formula check notes

    #Turbulence
    randomSeed = 42
    Brms = 0.1*nG #FIRST MODEL
    lMin = 50*kpc #FIRST MODEL
    lMax = 5*Mpc #FIRST MODEL
    #Brms = 1e-4*nG # SECOND MODEL
    #lMin = 1*kpc # SECOND MODEL
    #lMax = 50*kpc # SECOND MODEL
    sIndex = 5./3.
    turbSpectrum = SimpleTurbulenceSpectrum(Brms, lMin, lMax, sIndex)
    gridprops = GridProperties(Vector3d(0), 512, 10*kpc) #SECOND MODEL 512 0.1
    BField = SimpleGridTurbulence(turbSpectrum, gridprops, randomSeed)

    print('Lc = {:.1f} kpc'.format(BField.getCorrelationLength() / kpc))  # correlation length
    print('sqrt(<B^2>) = {:.1f} nG'.format(BField.getBrms() / nG))   # RMS
    print('<|B|> = {:.1f} nG'.format(BField.getMeanFieldStrength() / nG))  # mean

    #CREATING SIMULATION
    sim = ModuleList()
    sim.add(PropagationCK(BField))
    sim.add(PhotoPionProduction(CMB()))
    sim.add(PhotoPionProduction(IRB_Kneiske04()))
    #sim.add(PhotoDisintegration(CMB()))
    #sim.add(PhotoDisintegration(IRB_Kneiske04()))
    sim.add(ElectronPairProduction(CMB()))
    sim.add(ElectronPairProduction(IRB_Kneiske04()))
    #sim.add(NuclearDecay())
    sim.add(MaximumTrajectoryLength(280 * Mpc)) # Max clust dist is 254 Mpc
    #output = TextOutput('test_trajectory.txt', Output.Trajectory3D)

    shapley_coords = pd.read_csv("shapley_with_radii.csv")
    for index, clust in shapley_coords.iterrows():
        obs = Observer()

        distance = cosmo.comoving_distance(clust["z"])
        cords = SkyCoord(ra=clust["RAJ2000"]*u.deg, dec=clust["DEJ2000"]*u.deg, 
                        frame='icrs').transform_to("galactic")
        lon = cords.galactic.l
        lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
        lat = cords.galactic.b
        cords_wrapped = SkyCoord(l = lon.rad*u.rad, b = lat.rad*u.rad, distance = distance, frame='galactic').transform_to("galactocentric")
        
        obs.add(ObserverSurface(Sphere(Vector3d(cords_wrapped.x.value, cords_wrapped.y.value, cords_wrapped.z.value) * Mpc, 
                                       clust["R500"] * kpc)))
        obs.setDeactivateOnDetection(True)
        sim.add(obs)

    return sim


def innerGalacticSimulator(events, sigma_energy, sigma_dir, particle):
    '''This function performs backtracking of charged particles to the edge of the Galaxy.
    Returns pd.DataFrame with coordinates, energies etc'''

    lons0, lats0, dir_lons, dir_lats, dir_r, pos_x, pos_y, pos_z, init_energies, res_energies = [], [], [], [], [], [], [], [], [], []
    inner_sim, gal_obs = setupInnerSimulation(seed)
    
    #TEMP SOLUTION, ADDITIONAL OUTPUT
    #output = ShapleyTrajectoryOutput(f'test_trajectories_inner.txt')
    #inner_sim.add(output)

    for event in tqdm(events):
        NUM_OF_SIMS = 100
        particle_charge = particle

        mean_energy = event[3] * EeV
        position = Vector3d(-8.122, 0, 0.0208) * kpc #Astropy params
        
        coords = SkyCoord(ra=event[1], dec=event[2], frame='icrs', unit='deg')
        #Here we have longtitudes [0, 2pi] and latitudes
        lon = coords.galactic.l
        lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
        lon = lon.radian
        lat = coords.galactic.b.radian
        # CRPropa uses
        #   longitude (phi) [-pi, pi] with 0 pointing in x-direction
        #   colatitude (theta) [0, pi] with 0 pointing in z-direction
        lons0.append(lon) # For plotting maps in [-pi, pi]
        lats0.append(lat) # For plotting maps in [pi/2, -pi/2]
        init_energies.append(event[3])
        #lon = lon - np.pi #CrPropa uses [-pi, pi]
        lat = np.pi/2 - lat #CrPropa uses colatitude, e.g. 90 - lat in degrees

        mean_dir = Vector3d()
        mean_dir.setRThetaPhi(1, lat, lon)

        for i in range(NUM_OF_SIMS):
            energy = R.randNorm(mean_energy, sigma_energy*mean_energy)
            direction = R.randVectorAroundMean(mean_dir, sigma_dir)       
            candidate = Candidate(ParticleState(particle_charge, energy, position, direction))
            inner_sim.run(candidate)

            res = candidate.current
            direction_lon, direction_lat, direction_r = res.getDirection().getPhi(), res.getDirection().getTheta(), res.getDirection().getR()
            dir_lons.append(direction_lon)
            dir_lats.append(direction_lat)
            dir_r.append(direction_r)
            position_x, position_y, position_z = res.getPosition().x / kpc, res.getPosition().y / kpc, res.getPosition().z / kpc
            pos_x.append(position_x)
            pos_y.append(position_y)
            pos_z.append(position_z)
            #res_energies.append(candidate.current.getRigidity() * 1 / 10**18)#FOR PROTON in EeV
            res_energies.append(candidate.current.getEnergy() / 10**18)#FOR every charge

    galactic_results = pd.DataFrame()
    galactic_results["Event_Id"] = np.repeat([i for i in range(len(events))], NUM_OF_SIMS)
    galactic_results["Sim_Id"] = np.tile([i for i in range(NUM_OF_SIMS)], len(events))
    galactic_results["Init_Dir_Lon"] = np.repeat([i for i in lons0], NUM_OF_SIMS)
    galactic_results["Init_Dir_Lat"] = np.repeat([i for i in lats0], NUM_OF_SIMS)
    galactic_results["Dir_R"] = dir_r
    galactic_results["Dir_Lon"] = dir_lons
    galactic_results["Dir_CoLat"] = dir_lats
    galactic_results["Pos_X, kpc"] = pos_x
    galactic_results["Pos_Y, kpc"] = pos_y
    galactic_results["Pos_Z, kpc"] = pos_z
    galactic_results["Init_Energy, EeV"] = np.repeat([i for i in init_energies], NUM_OF_SIMS)
    galactic_results["Res_Energy, EeV"] = res_energies

    return galactic_results

def innerGalacticVisualizer(data, fname):
    import matplotlib.colors as colors
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    from analyse_trajectories import calculate_kde

    #plt.figure(figsize=(12,7))
    plt.figure(figsize=(16,9))
    ax = plt.subplot(111, projection = 'hammer')
    
    #KDE
    '''
    xyz_kde = calculate_kde(-data["Init_Dir_Lon"], data["Init_Dir_Lat"])
    confidence_levels = [0.6827, 0.9545, 0.9973]
    Z_flat = xyz_kde[2].flatten()
    Z_sorted = np.sort(Z_flat)[::-1]
    cumsum = np.cumsum(Z_sorted)
    cumsum /= cumsum[-1]
    levels = sorted([Z_sorted[np.searchsorted(cumsum, cl)] for cl in confidence_levels])

    ax.contour(xyz_kde[0], xyz_kde[1], xyz_kde[2], levels=levels, colors=['blue', 'green', 'red'])#, zorder=0)
    mesh = ax.pcolormesh(xyz_kde[0], xyz_kde[1], xyz_kde[2], cmap='viridis')#, zorder=0)
    '''
    #Colormaps have same limits DEFINED BY FINAL ENERGIES because they can be smaller than init min energy
    init_energies = data["Init_Energy, EeV"]*1e18
    final_energies = data["Res_Energy, EeV"]*1e18
    
    finals = ax.scatter(-data["Dir_Lon"], np.pi/2 - data["Dir_CoLat"], marker='o', c=final_energies, 
                cmap='viridis', norm = colors.LogNorm(vmin = final_energies.min(), vmax = final_energies.max()),
                s=1, alpha=0.3, label='Simulated particles')
    
    inits = ax.scatter(-data["Init_Dir_Lon"], data["Init_Dir_Lat"], marker='*', c=init_energies, 
                cmap='viridis', norm = colors.LogNorm(vmin = final_energies.min(), vmax = final_energies.max()),
                s=10, label='Observed events')
    
    #SHAPLEY
    shapley_coords = pd.read_csv("shapley_with_radii.csv")

    cords = SkyCoord(ra=shapley_coords["RAJ2000"]*u.deg, dec=shapley_coords["DEJ2000"]*u.deg, 
                    frame='icrs').transform_to("galactic")
    lon = cords.galactic.l
    lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
    lon = lon.radian
    lat = cords.galactic.b.radian
    #ax.scatter(-lon, lat, marker='+', c='r', s=5, label='Shapley member clusters')

    distance = cosmo.comoving_distance(shapley_coords["z"]) * 1000 # as radii in kpc
    radius_kpc = shapley_coords["R500"]*3 #3R00
    
    radius = radius_kpc / distance
    for i, lon, lat, radius in zip([i for i in range(len(lon))], -lon, lat, np.array(radius)):
        '''
        if i == 0:
            #FOR LEGEND ONLY
            circle = Circle((lon, lat), radius, edgecolor='red', facecolor='none', 
                                linewidth=0.3, label='Shapley member clusters')
            ax.add_patch(circle)
            continue
        '''
        circle = Circle((lon, lat), radius, edgecolor='red', facecolor='none', linewidth=0.5)
        circle._resolution = 1000
        ax.add_patch(circle)

    #plt.text(+90*np.pi/180, 55*np.pi/180, "Shapley Supercluster", fontsize=10, fontweight='bold')
    #ax.text(30*np.pi/180, 60*np.pi/180 - np.pi/60, "Shapley Supercluster", fontsize=10, fontweight='bold')

    #GENERAL TICKS
    y_tick_labels = ['-75°', '-60°', '-45°', '-30°', '-15°', '0°', '15°', '30°', '45°', '60°', '75°']
    y_tick_positions = [-75*np.pi/180, -60*np.pi/180, -45*np.pi/180, -30*np.pi/180, -15*np.pi/180, 
                        0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180,]
    
    plt.yticks(y_tick_positions, labels=y_tick_labels)

    #x_tick_labels = ['', '150°', '120°', '90°', '60°', '30°', '0', '-30°', '-60°', '-90°', '', '', '']
    #x_tick_positions = [-np.pi, -5*np.pi/6, -2*np.pi/3, -np.pi/2, -1*np.pi/3, -np.pi/6, 0, 
    #                    np.pi/6, 1*np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi]
    x_tick_labels = ['', '150°', '120°', '90°', '60°', '30°', '0', '', '', '', '']
    x_tick_positions = [-np.pi, -5*np.pi/6, -2*np.pi/3, -np.pi/2, -1*np.pi/3, -np.pi/6, 0, 
                         np.pi, 2*np.pi/9, 11*np.pi/36, 7*np.pi/18]
    
    plt.xticks(x_tick_positions, labels=x_tick_labels)

    yticks_right = [0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180]
    ylabels_right = ['', '15°', '30°', '', '60°', '']
    for pos, label in zip(yticks_right, ylabels_right):
        ax.text(np.pi + np.pi/16, pos, label, fontsize=10)
    
    #TICKS FOR CROP
    xticks_crop = [2*np.pi/9 - np.pi/30, 11*np.pi/36 - np.pi/30, 7*np.pi/18 - np.pi/30] # To plot on left of lat lines we substract 6 degrees
    xlabels_crop = ['320°', '305°', '290°']
    for pos, label in zip(xticks_crop, xlabels_crop):
        ax.text(pos, np.pi/12 - np.pi/60, label, fontsize=6)
    
    yticks_crop = [np.pi/6 - np.pi/70, np.pi/4 - np.pi/70] # To plot on top of lon lines we add 2 degrees
    ylabels_crop = ['30°', '45°']
    for pos, label in zip(yticks_crop, ylabels_crop):
        ax.text(np.pi/2 - np.pi/60 - 16*np.pi/180, pos, label, fontsize=6)

    ax.set_axisbelow(True)
    ax.grid(True)
    plt.tight_layout()
    #plt.title("Map of simulated cosmic rays inside the Galaxy")

    handles, labels = ax.get_legend_handles_labels()
    legend_circle = Line2D([0], [0], marker='o', color='red', linestyle='None',
                       markersize=5, markerfacecolor='none', markeredgewidth=0.3)
    handles.append(legend_circle)
    labels.append('Shapley member clusters')
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0.69, 0.8), loc='center', fontsize=3.5)
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    #plt.show()
    plt.close()
    #exit()
    #CROPPING SIGNIFICANT REGION
    from PIL import Image

    map = Image.open(fname)
    
    #plt.imshow(map)
    #plt.show()
    #exit()
    fig, axs = plt.subplots(figsize=(9, 9))
    
    #axs.imshow(map.crop((1740, 300, 2740, 955)), aspect="auto") 300 dpi 12 7
    axs.imshow(map.crop((5475, 1010, 6700, 2250)), aspect="auto")
    axs.set_xticks([])
    axs.set_yticks([])

    #COLORBAR
    
    cbar = fig.colorbar(inits, orientation = 'horizontal', pad=0.02)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    c_ticks = np.array([10**(19.7), 10**(19.8), 10**(20), 10**(20.2), 10**(20.4)
                        , 6*(10**19), 4*(10**19), 3*(10**19)]).astype(np.float64)
    c_labels = ['19.7', '19.8', '20.0', '20.2', '20.4', '', '', '']
    cbar.set_ticks(c_ticks)
    cbar.set_ticklabels(c_labels)
    cbar.set_label("Log₁₀ of observed and simulated CR energies")
    
    #KDE COLORBAR
    #cbar = fig.colorbar(mesh, orientation='horizontal', pad=0.02)
    #cbar.set_label('Probability density value')
    #cbar.solids.set_rasterized(True)
    #cbar.solids.set_edgecolor("face")
    
    #plt.box(False)
    plt.tight_layout()
    plt.title("Map of simulated cosmic rays inside the Galaxy (Z = 26)")
    plt.savefig(fname[:fname.rfind('/')] + '/cropped_' + fname[fname.rfind('/')+1:], dpi=600, bbox_inches='tight')
    plt.show()

def outerGalacticVisualizer(fname_final_directions, fname_detections, data, fname):
    import matplotlib.colors as colors
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    data_final_directions = np.genfromtxt(fname_final_directions, unpack=True, skip_footer=1)
    #data_detections = np.genfromtxt(fname_detections, unpack=True, skip_footer=1)

    plt.figure(figsize=(16,9))
    ax = plt.subplot(111, projection = 'hammer')

    I, dir_lons, dir_colats, final_energies = data_final_directions
    '''TEMPORARY INF HANDLER, CHECK LATER'''
    finite_mask = np.isfinite(final_energies)
    final_energies = final_energies[finite_mask]
    dir_lons = dir_lons[finite_mask]
    dir_colats = dir_colats[finite_mask]

    #Colormaps have same limits DEFINED BY FINAL ENERGIES because they can be smaller than init min energy
    init_energies = data["Init_Energy, EeV"]*1e18
    final_energies = final_energies*1e18
    finals = ax.scatter(-dir_lons, np.pi/2 - dir_colats, marker='o', c=final_energies, 
                cmap='viridis', norm = colors.LogNorm(vmin = init_energies.min(), vmax = final_energies.max()),
                s=1, alpha=0.3, label='Simulated particles')
    
    inits = ax.scatter(-data["Init_Dir_Lon"], data["Init_Dir_Lat"], marker='*', c=init_energies, 
                cmap='viridis', norm = colors.LogNorm(vmin = init_energies.min(), vmax = init_energies.max()),
                s=10, label='Observed events')
    
    #SHAPLEY
    shapley_coords = pd.read_csv("shapley_with_radii.csv")

    cords = SkyCoord(ra=shapley_coords["RAJ2000"]*u.deg, dec=shapley_coords["DEJ2000"]*u.deg, 
                    frame='icrs').transform_to("galactic")
    lon = cords.galactic.l
    lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
    lon = lon.radian
    lat = cords.galactic.b.radian
    #ax.scatter(-lon, lat, marker='+', c='r', s=5, label='Shapley member clusters')

    distance = cosmo.comoving_distance(shapley_coords["z"]) * 1000 # as radii in kpc
    radius_kpc = shapley_coords["R500"]*3 #3R00
    
    radius = radius_kpc / distance
    for i, lon, lat, radius in zip([i for i in range(len(lon))], -lon, lat, np.array(radius)):
        '''
        if i == 0:
            #FOR LEGEND ONLY
            circle = Circle((lon, lat), radius, edgecolor='red', facecolor='none', 
                                linewidth=0.3, label='Shapley member clusters')
            ax.add_patch(circle)
            continue
        '''
        circle = Circle((lon, lat), radius, edgecolor='red', facecolor='none', linewidth=0.5)
        circle._resolution = 1000
        ax.add_patch(circle)

    #plt.text(+90*np.pi/180, 55*np.pi/180, "Shapley Supercluster", fontsize=10, fontweight='bold')
    #ax.text(30*np.pi/180, 60*np.pi/180 - np.pi/60, "Shapley Supercluster", fontsize=10, fontweight='bold')

    #GENERAL TICKS
    y_tick_labels = ['-75°', '-60°', '-45°', '-30°', '-15°', '0°', '15°', '30°', '45°', '60°', '75°']
    y_tick_positions = [-75*np.pi/180, -60*np.pi/180, -45*np.pi/180, -30*np.pi/180, -15*np.pi/180, 
                        0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180,]
    
    plt.yticks(y_tick_positions, labels=y_tick_labels)

    #x_tick_labels = ['', '150°', '120°', '90°', '60°', '30°', '0', '-30°', '-60°', '-90°', '', '', '']
    #x_tick_positions = [-np.pi, -5*np.pi/6, -2*np.pi/3, -np.pi/2, -1*np.pi/3, -np.pi/6, 0, 
    #                    np.pi/6, 1*np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi]
    x_tick_labels = ['', '150°', '120°', '90°', '60°', '30°', '0', '', '', '', '']
    x_tick_positions = [-np.pi, -5*np.pi/6, -2*np.pi/3, -np.pi/2, -1*np.pi/3, -np.pi/6, 0, 
                         np.pi, 2*np.pi/9, 11*np.pi/36, 7*np.pi/18]
    
    plt.xticks(x_tick_positions, labels=x_tick_labels)

    yticks_right = [0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180]
    ylabels_right = ['', '15°', '30°', '', '60°', '']
    for pos, label in zip(yticks_right, ylabels_right):
        ax.text(np.pi + np.pi/16, pos, label, fontsize=10)
    
    #TICKS FOR CROP
    xticks_crop = [2*np.pi/9 - np.pi/30, 11*np.pi/36 - np.pi/30, 7*np.pi/18 - np.pi/30] # To plot on left of lat lines we substract 6 degrees
    xlabels_crop = ['320°', '305°', '290°']
    for pos, label in zip(xticks_crop, xlabels_crop):
        ax.text(pos, np.pi/12 - np.pi/60, label, fontsize=6)
    
    yticks_crop = [np.pi/6 - np.pi/70, np.pi/4 - np.pi/70] # To plot on top of lon lines we add 2 degrees
    ylabels_crop = ['30°', '45°']
    for pos, label in zip(yticks_crop, ylabels_crop):
        ax.text(np.pi/2 - np.pi/60 - 16*np.pi/180, pos, label, fontsize=6)

    ax.set_axisbelow(True)
    ax.grid(True)
    plt.tight_layout()
    #plt.title("Map of simulated cosmic rays inside the Galaxy")

    handles, labels = ax.get_legend_handles_labels()
    legend_circle = Line2D([0], [0], marker='o', color='red', linestyle='None',
                       markersize=5, markerfacecolor='none', markeredgewidth=0.3)
    handles.append(legend_circle)
    labels.append('Shapley member clusters')
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0.69, 0.8), loc='center', fontsize=3.5)
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    #plt.show()
    plt.close()
    #exit()
    #CROPPING SIGNIFICANT REGION
    from PIL import Image

    map = Image.open(fname)
    
    #plt.imshow(map)
    #plt.show()
    #exit()
    fig, axs = plt.subplots(figsize=(9, 9))
    
    #axs.imshow(map.crop((1740, 300, 2740, 955)), aspect="auto") 300 dpi 12 7
    axs.imshow(map.crop((5475, 1010, 6700, 2250)), aspect="auto")
    axs.set_xticks([])
    axs.set_yticks([])

    #COLORBAR
    
    cbar = fig.colorbar(inits, orientation = 'horizontal', pad=0.02)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    c_ticks = np.array([10**(19.7), 10**(19.8), 10**(20), 10**(20.2), 10**(20.4)
                        , 6*(10**19), 4*(10**19), 3*(10**19)]).astype(np.float64)
    c_labels = ['19.7', '19.8', '20.0', '20.2', '20.4', '', '', '']
    cbar.set_ticks(c_ticks)
    cbar.set_ticklabels(c_labels)
    cbar.set_label("Log₁₀ of observed and simulated CR energies")
    
    #KDE COLORBAR
    #cbar = fig.colorbar(mesh, orientation='horizontal', pad=0.02)
    #cbar.set_label('Probability density value')
    #cbar.solids.set_rasterized(True)
    #cbar.solids.set_edgecolor("face")
    
    #plt.box(False)
    plt.tight_layout()
    plt.title("Map of simulated cosmic rays outside the Galaxy (Z = 1)")
    plt.savefig(fname[:fname.rfind('/')] + '/cropped_' + fname[fname.rfind('/')+1:], dpi=600, bbox_inches='tight')
    plt.show()

def plot_kde(data, fname):
    import matplotlib.colors as colors
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    from analyse_trajectories import calculate_kde

    plt.figure(figsize=(16,9))
    ax = plt.subplot(111, projection = 'hammer')
    
    #KDE
    
    #xyz_kde = calculate_kde(-data["Init_Dir_Lon"], data["Init_Dir_Lat"])
    xyz_kde = calculate_kde(-data["Dir_Lon"], np.pi/2 - data["Dir_CoLat"])
    confidence_levels = [0.6827, 0.9545, 0.9973] #Make 90 or 95
    Z_flat = xyz_kde[2].flatten()
    Z_sorted = np.sort(Z_flat)[::-1]
    cumsum = np.cumsum(Z_sorted)
    cumsum /= cumsum[-1]
    levels = sorted([Z_sorted[np.searchsorted(cumsum, cl)] for cl in confidence_levels])

    ax.contour(xyz_kde[0], xyz_kde[1], xyz_kde[2], levels=levels, colors=['blue', 'green', 'red'])#, zorder=0)
    mesh = ax.pcolormesh(xyz_kde[0], xyz_kde[1], xyz_kde[2], cmap='viridis')#, zorder=0)
    

    #Colormaps have same limits DEFINED BY FINAL ENERGIES because they can be smaller than init min energy
    init_energies = data["Init_Energy, EeV"]*1e18
    final_energies = data["Res_Energy, EeV"]*1e18
    
    
    #finals = ax.scatter(-data["Dir_Lon"], np.pi/2 - data["Dir_CoLat"], marker='o', c=final_energies, 
    #            cmap='viridis', norm = colors.LogNorm(vmin = final_energies.min(), vmax = final_energies.max()),
    #           s=1, alpha=0.3, label='Simulated particles')
    
    #inits = ax.scatter(-data["Init_Dir_Lon"], data["Init_Dir_Lat"], marker='*', c=init_energies, 
    #            cmap='viridis', norm = colors.LogNorm(vmin = final_energies.min(), vmax = final_energies.max()),
    #            s=10, label='Observed events')
    
    #SHAPLEY
    shapley_coords = pd.read_csv("shapley_with_radii.csv")

    cords = SkyCoord(ra=shapley_coords["RAJ2000"]*u.deg, dec=shapley_coords["DEJ2000"]*u.deg, 
                    frame='icrs').transform_to("galactic")
    lon = cords.galactic.l
    lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
    lon = lon.radian
    lat = cords.galactic.b.radian

    distance = cosmo.comoving_distance(shapley_coords["z"]) * 1000 # as radii in kpc
    radius_kpc = shapley_coords["R500"]*3 #3R00
    
    radius = radius_kpc / distance
    for i, lon, lat, radius in zip([i for i in range(len(lon))], -lon, lat, np.array(radius)):
        circle = Circle((lon, lat), radius, edgecolor='red', facecolor='none', linewidth=0.5)
        circle._resolution = 1000
        ax.add_patch(circle)
    
    #CENTAURUS
    cen_gal_cords = SkyCoord(l=309.5*u.deg, b=19.4*u.deg, frame='galactic')
    cen_gal_lon = cen_gal_cords.galactic.l
    cen_gal_lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
    cen_gal_lon = cen_gal_lon.radian
    cen_gal_lat = cen_gal_cords.galactic.b.radian
    cen_gal_r500 = 50000
    cen_gal_dist = 100
    cen_gal_radius = cen_gal_r500 / cen_gal_dist
    circle = Circle((-cen_gal_lon, cen_gal_lat), cen_gal_radius, edgecolor='purple', facecolor='none', linewidth=0.5)
    circle._resolution = 1000
    ax.add_patch(circle)

    #GENERAL TICKS

    y_tick_labels = ['-75°', '-60°', '-45°', '-30°', '-15°', '0°', '15°', '30°', '45°', '60°', '75°']
    y_tick_positions = [-75*np.pi/180, -60*np.pi/180, -45*np.pi/180, -30*np.pi/180, -15*np.pi/180, 
                        0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180,]
    
    plt.yticks(y_tick_positions, labels=y_tick_labels)

    x_tick_labels = ['', '150°', '120°', '90°', '60°', '30°', '0', '', '', '', '']
    x_tick_positions = [-np.pi, -5*np.pi/6, -2*np.pi/3, -np.pi/2, -1*np.pi/3, -np.pi/6, 0, 
                         np.pi, 2*np.pi/9, 11*np.pi/36, 7*np.pi/18]
    
    plt.xticks(x_tick_positions, labels=x_tick_labels)

    yticks_right = [0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180]
    ylabels_right = ['', '15°', '30°', '', '60°', '']
    for pos, label in zip(yticks_right, ylabels_right):
        ax.text(np.pi + np.pi/16, pos, label, fontsize=10)
    
    #TICKS FOR CROP
    xticks_crop = [2*np.pi/9 - np.pi/30, 11*np.pi/36 - np.pi/30, 7*np.pi/18 - np.pi/30] # To plot on left of lat lines we substract 6 degrees
    xlabels_crop = ['320°', '305°', '290°']
    for pos, label in zip(xticks_crop, xlabels_crop):
        ax.text(pos, np.pi/12 - np.pi/60, label, fontsize=6)
    
    yticks_crop = [np.pi/6 - np.pi/70, np.pi/4 - np.pi/70] # To plot on top of lon lines we add 2 degrees
    ylabels_crop = ['30°', '45°']
    for pos, label in zip(yticks_crop, ylabels_crop):
        ax.text(np.pi/2 - np.pi/60 - 16*np.pi/180, pos, label, fontsize=6)

    #GRID

    #ax.set_axisbelow(True)
    ax.grid(True)
    plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    legend_circle = Line2D([0], [0], marker='o', color='red', linestyle='None',
                           markersize=5, markerfacecolor='none', markeredgewidth=0.3)
    legend_circle_cen_gal = Line2D([0], [0], marker='o', color='purple', linestyle='None',
                                   markersize=5, markerfacecolor='none', markeredgewidth=0.3)
    handles.append(legend_circle)
    labels.append('Shapley member clusters')
    handles.append(legend_circle_cen_gal)
    labels.append('Cen A')
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0.69, 0.8), loc='center', fontsize=3.5)
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    #plt.show()
    plt.close()
    #exit()

    #CROPPING REGION OF INTEREST
    from PIL import Image

    map = Image.open(fname)
    fig, axs = plt.subplots(figsize=(9, 9))
    
    #axs.imshow(map.crop((1740, 300, 2740, 955)), aspect="auto") 300 dpi 12 7
    axs.imshow(map.crop((5475, 1010, 6700, 2250)), aspect="auto")
    axs.set_xticks([])
    axs.set_yticks([])
    
    #KDE COLORBAR
    cbar = fig.colorbar(mesh, orientation='horizontal', pad=0.02)
    cbar.set_label('Probability density value')
    #cbar.solids.set_rasterized(True)
    #cbar.solids.set_edgecolor("face")
    
    #plt.box(False)
    plt.tight_layout()
    #plt.title("Probability density map for observed Auger events with lg(E) > 19.7")
    #plt.title(f"Probability density map for simulated CR (Z = 1)")
    plt.title(f"Probability density map for {fname.split('/')[1][:fname.rfind('.')]} (Z = 1)")
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    #plt.show()
    plt.close()

def vis_double_kde():
    pass

class ShapleyTrajectoryOutput(Module):
    """
    Custom trajectory output: i, x, y, z
    where i is a running cosmic ray number
    and x,y,z are the Galactocentric coordinates in [Mpc].
    Also returns velocities to measure time.
    Creates file, which is fulled by coordinates of detected 
    particles by various potential sources determined by a condition
    """
    def __init__(self, fname, fname_det, fname_dirs):
        Module.__init__(self)
        self.fout = open(fname, 'w')
        self.fout.write('#i\tX\tY\tZ\tLon\tCoLat\n')
        self.fout_detections = open(fname_det, 'w')
        self.fout_detections.write('#X\tY\tZ\tLon\tCoLat\n')
        self.final_directions = open(fname_dirs, 'w')
        self.final_directions.write('#i\tLon\tCoLat\tEnergy\n')
        self.i = 0
        self.detection_counter = 0

    def process(self, c):
        r = c.current.getPosition()
        v = c.current.getVelocity()
        #v_mod = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        x = r.x / Mpc
        y = r.y / Mpc
        z = r.z / Mpc
        #self.fout.write('%i\t%.3f\t%.3f\t%.3f\t%.3f\n'%(self.i, x, y, z, v_mod))
        self.fout.write('%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n'%(self.i, x, y, z, 
                                                  c.current.getDirection().getPhi(), c.current.getDirection().getTheta())) # Lon and CoLat
        if not(c.isActive()):
            lon, lat, energy = c.current.getDirection().getPhi(), c.current.getDirection().getTheta(), c.current.getRigidity() * 1 / 10**18 #FOR PROTONS
            self.final_directions.write('%i\t%.3f\t%.3f\t%.3f\n'%(self.i, lon, lat, energy))
            self.i += 1         
            if np.sqrt(x**2 + y**2 + z**2) < 260.0 and np.sqrt(x**2 + y**2 + z**2) > 185.0 : #THIS CONDITION SHOULD BE REVISED LATER
                self.fout_detections.write('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n'%(x, y, z, c.current.getDirection().getPhi(), c.current.getDirection().getTheta()))
                #print(f"Detected with coords {x, y, z}") 
                #self.detection_counter += 1
                #print(f"NUMBER OF DETECTED PARTICLES : {self.detection_counter}")

    def close(self):
        #print(f"Number of detected particles : {self.detection_counter}") #Somehow not working
        self.fout.close()
        self.fout_detections.close()
        self.final_directions.close()


if __name__ == '__main__':
    seed = 42
    R = Random(seed)

    '''
    Uploading events from the Shapley quadrant
    '''
    
    events = []
    with open('Auger_lowE_shapley.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[5]), float(temp_event[6]), float(temp_event[7])))
    

    #particles = [- nucleusId(1,1), - nucleusId(4,2), - nucleusId(12,6), - nucleusId(52,26)]
    particles = {'p': - nucleusId(1,1), 
                 'Fe': - nucleusId(52,26)
                 }

    #INSTRUMENTAL FROM PA PAPER
    sigma_energy = 0.07
    sigma_dir = 0.002 #1 degree directional uncertainty
    shapley_coords = pd.read_csv("shapley_with_radii.csv")
    
    '''PERFORMING SIMULATIONS INSIDE THE GALAXY'''
    
    inner_results = innerGalacticSimulator(sigma_energy=sigma_energy, sigma_dir=sigma_dir, events=events, particle = particles['p'])
    
    #innerGalacticVisualizer(inner_results)
    #print(inner_results.head())
    #visualize_3D_shapley('test_trajectories_inner.txt')
    #exit()
    #PERFORMING SIMULATIONS OUTSIDE THE GALAXY
    '''
    sim = setupOuterSimulation()
    output = ShapleyTrajectoryOutput('sim_results/outer_trajectories_1model_p.txt', 'sim_results/outer_detections_1model_p.txt', 
                                     'sim_results/outer_directions_1model_p.txt')
    sim.add(output)
    '''
    '''
    for i in tqdm(range(len(inner_results))):
        position = Vector3d(inner_results["Pos_X, kpc"][i], inner_results["Pos_Y, kpc"][i], inner_results["Pos_Z, kpc"][i]) * kpc  # position
        direction = Vector3d()  
        direction.setRThetaPhi(1, inner_results["Dir_CoLat"][i], inner_results["Dir_Lon"][i])# direction
        candidate = Candidate(particles['p'], inner_results["Res_Energy, EeV"][i] * EeV, position, direction)#Energy in Joules
        sim.run(candidate, True)
    '''
    
    #visualize_3D_shapley('sim_results/test_trajectories_10_1model_p.txt')
    #innerGalacticVisualizer(inner_results, fname="paper_plots/inner_gal_base_Fe.jpeg")
    plot_kde(inner_results, fname="paper_plots/simulated CR.jpeg")
    #outerGalacticVisualizer('sim_results/outer_directions_1model_p.txt', 'sim_results/outer_detections_1model_p.txt', inner_results, 
    #                        fname="paper_plots/outer_gal_Lc1Mpc_B01nG_p.jpeg")
    


         