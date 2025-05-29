from pylab import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import time
from typing import Tuple

cosmo = FlatLambdaCDM(H0=67.6, Om0=0.286)

def plot_sphere(ax, center, radius, color='black', alpha=0.3, resolution=20):
    """
    Plots a sphere on a 3D Axes.

    Parameters:
        ax      : matplotlib 3D axes
        center  : tuple or list of 3 elements (x0, y0, z0)
        radius  : float, radius of the sphere
        color   : sphere color
        alpha   : transparency (0 to 1)
        resolution : int, number of subdivisions for the sphere surface
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

def calculate_kde(x, y) -> Tuple[Tuple[np.array, np.array, np.array], float]:
    '''Calculates pdf using kde with bandwith from the gridsearch
    and returns the denstiy value of needed object for this pdf divided by
    the max density value for this pdf
    https://gist.github.com/daleroberts/7a13afed55f3e2388865b0ec94cd80d2
    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/'''
    xz = np.vstack([x, y])
    d = xz.shape[0]
    n = xz.shape[1]

    #Creating grid for search and finding best estimator in terms of bandwidth
    print('---STARTING KDE GRIDSEARCH---')
    start_time = time.time()
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': np.linspace(0.01, 1, 100)},
                    cv=20) # 20-fold cross-validation with 1000 bandwidths
    grid.fit(xz.T)
    kde = grid.best_estimator_
    print(f"Gridsearch FINISHED in {time.time() - start_time} s.")

    xmin = x.min() - 0.2
    xmax = x.max() + 0.4
    ymin = y.min() - 0.2
    ymax = y.max() + 0.2

    X, Y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(np.exp(kde.score_samples(positions.T)), X.shape)

    return (X, Y, Z) #score_samples returns the log density, so exp is needed. Also prob density can be more than 1


def visualize_3D_shapley(fname):
    data = np.genfromtxt(fname, unpack=True, skip_footer=1)

    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot(111, projection='3d')

    # trajectory pointss
    I,X,Y,Z,dir_lon,dir_colat = data
    for i in tqdm(np.unique(I)):
        #if i > 50: break
        ax.plot(X[I == i], Y[I == i], Z[I == i], lw=1, alpha=0.5, c='g')

    shapley_coords = pd.read_csv("shapley_with_radii.csv")

    for index, clust in shapley_coords.iterrows():
        #shapley_cords = {"RA": 201.9934, "DEC": -31.5014, "z": 0.0487}
        distance = cosmo.comoving_distance(clust["z"])
        cords = SkyCoord(ra=clust["RAJ2000"]*u.deg, dec=clust["DEJ2000"]*u.deg, 
                        frame='icrs').transform_to("galactic")
        lon = cords.galactic.l
        lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
        lat = cords.galactic.b

        cords_wrapped = SkyCoord(l = lon.rad*u.rad, b = lat.rad*u.rad, distance = distance, frame='galactic').transform_to("galactocentric")

        ax.scatter(cords_wrapped.x.value, cords_wrapped.y.value, cords_wrapped.z.value, s=1, c='r')
        plot_sphere(ax = ax, center = (cords_wrapped.x.value, cords_wrapped.y.value, cords_wrapped.z.value), 
                    radius = clust["R500"] / 1000)#radius in Mpc

    ax.set_xlabel('x / Mpc', fontsize=18)
    ax.set_ylabel('y / Mpc', fontsize=18)
    ax.set_zlabel('z / Mpc', fontsize=18)
    #ax.set_xlim((-270, 270))
    #ax.set_ylim((-270, 270))
    #ax.set_zlim((-270, 270))
    #ax.xaxis.set_ticks((0, 5, 10, 15))
    #ax.yaxis.set_ticks((0, 5, 10, 15))
    #ax.zaxis.set_ticks((0, 5, 10, 15))

    plt.show()

def plot2DProjection(data, title = '', fname = None) -> None:
    plt.ioff() # Turn off interactive mode to speed up building
    plt.figure(figsize=(6,8))

    cb_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    
    I, X, Y, Z, lon, colat = data
    
    for i in np.unique(I):
        plt.plot(X[I == i], Y[I == i], lw=0.05, alpha=0.1, color=cb_color_cycle[0], zorder=-1)

    #Plot SHapley subclass
    #sgr_cords = objects_list["sgr"]
    #plt.scatter(sgr_cords[1], sgr_cords[2], marker='*', c=cb_color_cycle[1], s=90, zorder=1) #43.02 0.77 12.5±1.7
    #plt.text(sgr_cords[1]-1.3, sgr_cords[2]+0.5, s="SGR 1900+14", fontsize=12, c=cb_color_cycle[1], zorder=1)
    
    plt.title(title)
    #plt.xlim([-8.5, 7.5])
    #plt.ylim([-0.5, 12.5])
    plt.xlabel(f"X / Mpc")
    plt.ylabel(f"Y / Mpc")

    if fname: plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()

def plot_detections(data, fname, events):
    '''
    x, y, z, det_dir_lons, det_dir_colats = data

    plt.figure(figsize=(12,7))
    plt.subplot(111, projection = 'hammer')

    lons0, lats0 = [], []
    for event in events:  
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

    #VISUALISING INITS
    plt.scatter(lons0, lats0, marker='*', c='orange', s=10, label='Observed events')

    #VISUALISING DETECTIONS
    plt.scatter(det_dir_lons, np.pi/2 - det_dir_colats, marker='o', c='purple', s=20, alpha=0.6, label='Detected Particles')

    shapley_coords = pd.read_csv("shapley_with_radii.csv")

    #CHANGE TO VECTORIZED APPROACH IN FUTURE
    cords = SkyCoord(ra=shapley_coords["RAJ2000"]*u.deg, dec=shapley_coords["DEJ2000"]*u.deg, 
                    frame='icrs').transform_to("galactic")
    lon = cords.galactic.l
    lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
    lon = lon.radian
    lat = cords.galactic.b.radian
    plt.scatter(lon, lat, marker='+', c='r', s=25, label='Shapley member clusters')
    plt.text(-90*np.pi/180, 55*np.pi/180, "Shapley Supercluster", fontsize=10, fontweight='bold')

    y_tick_labels = ['-75°', '-60°', '-45°', '-30°', '-15°', '0°', '15°', '30°', '45°', '60°', '']
    y_tick_positions = [-75*np.pi/180, -60*np.pi/180, -45*np.pi/180, -30*np.pi/180, -15*np.pi/180, 
                        0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180,]
    
    plt.yticks(y_tick_positions, labels=y_tick_labels)

    plt.grid(True)
    plt.tight_layout()
    plt.title("Map of detections outside the Galaxy")
    plt.legend()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()   
    plt.close() 
    '''
    x, y, z, det_dir_lons, det_dir_colats = data
    shapley_coords = pd.read_csv("shapley_with_radii.csv")
    clust_A = [14, 19, 21, 40, 36]
    clust_B = [4, 10]
    clust_C = [34, 37]
    clust_D = [23]
    clust_CD = [34, 37, 23, 31, 42]
    clust_E = [1, 3, 7, 9, 11, 13, 16]
    clust_F = [12, 15, 29, 22, 30, 39]
    clust_G = [0, 6]
    clust_H = [41]
    clust_I = [33, 43]
    clust_J = [2, 5, 17, 18, 20, 24, 28, 38, 44, 27, 32, 35]
    
    # Combine all values into a single set
    used = set(clust_A + clust_B + clust_CD + clust_E + clust_F + clust_G + clust_H + clust_I + clust_J)

    # Create list of numbers from 0 to 43 that are not used
    unused = [x for x in range(44) if x not in used]
    
    '''
    CLUST BY CLUST
    for i, row in shapley_coords.loc[clust_A].iterrows():
        plt.figure(figsize=(12,7))
        plt.subplot(111, projection = 'hammer')

        lons0, lats0 = [], []
        for event in events:  
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

        #VISUALISING INITS
        plt.scatter(lons0, lats0, marker='*', c='orange', s=10, label='Observed events')

        #VISUALISING DETECTIONS
        plt.scatter(det_dir_lons, np.pi/2 - det_dir_colats, marker='o', c='purple', s=20, alpha=0.6, label='Detected Particles')

        #CHANGE TO VECTORIZED APPROACH IN FUTURE
        cords = SkyCoord(ra=row["RAJ2000"]*u.deg, dec=row["DEJ2000"]*u.deg, 
                        frame='icrs').transform_to("galactic")
        lon = cords.galactic.l
        lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
        lon = lon.radian
        lat = cords.galactic.b.radian
        plt.scatter(lon, lat, marker='+', c='r', s=25, label='Shapley member clusters')
        plt.text(-90*np.pi/180, 55*np.pi/180, "Shapley Supercluster", fontsize=10, fontweight='bold')

        y_tick_labels = ['-75°', '-60°', '-45°', '-30°', '-15°', '0°', '15°', '30°', '45°', '60°', '']
        y_tick_positions = [-75*np.pi/180, -60*np.pi/180, -45*np.pi/180, -30*np.pi/180, -15*np.pi/180, 
                            0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180,]
        
        plt.yticks(y_tick_positions, labels=y_tick_labels)

        plt.grid(True)
        plt.tight_layout()
        plt.title(f"{i}")
        plt.legend()
        plt.show()   
        plt.close() 
    '''
    plt.figure(figsize=(12,7))
    plt.subplot(111, projection = 'hammer')

    lons0, lats0 = [], []
    for event in events:  
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

    #VISUALISING INITS
    plt.scatter(lons0, lats0, marker='*', c='orange', s=10, label='Observed events')

    #VISUALISING DETECTIONS
    plt.scatter(det_dir_lons, np.pi/2 - det_dir_colats, marker='o', c='purple', s=20, alpha=0.6, label='Detected Particles')

    #CHANGE TO VECTORIZED APPROACH IN FUTURE
    cords = SkyCoord(ra=shapley_coords.loc[clust_G]["RAJ2000"]*u.deg, dec=shapley_coords.loc[clust_G]["DEJ2000"]*u.deg, 
                    frame='icrs').transform_to("galactic")
    lon = cords.galactic.l
    lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
    lon = lon.radian
    lat = cords.galactic.b.radian
    plt.scatter(lon, lat, marker='+', c='r', s=40, label='Shapley member clusters')
    plt.text(-90*np.pi/180, 55*np.pi/180, "Shapley Supercluster", fontsize=10, fontweight='bold')

    y_tick_labels = ['-75°', '-60°', '-45°', '-30°', '-15°', '0°', '15°', '30°', '45°', '60°', '']
    y_tick_positions = [-75*np.pi/180, -60*np.pi/180, -45*np.pi/180, -30*np.pi/180, -15*np.pi/180, 
                        0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180,]
    
    plt.yticks(y_tick_positions, labels=y_tick_labels)

    plt.grid(True)
    plt.tight_layout()
    #plt.title(f"{i}")
    plt.legend()
    plt.show()   
    plt.close() 

def plot_zero_approx_map(directions):
    I, dir_lons, dir_colats = directions
    shapley_coords = pd.read_csv("shapley_with_radii.csv")

    fig, ax = plt.subplots(figsize=(16, 9))

    #SIMULATED EVENTS
    
    xyz_kde = calculate_kde(dir_lons, np.pi/2 - dir_colats)
    confidence_levels = [0.6827, 0.9545, 0.9973]
    Z_flat = xyz_kde[2].flatten()
    Z_sorted = np.sort(Z_flat)[::-1]
    cumsum = np.cumsum(Z_sorted)
    cumsum /= cumsum[-1]
    levels = sorted([Z_sorted[np.searchsorted(cumsum, cl)] for cl in confidence_levels])

    ax.contour(xyz_kde[0], xyz_kde[1], xyz_kde[2], levels=levels, colors=['blue', 'green', 'red'])
    mesh = ax.pcolormesh(xyz_kde[0], xyz_kde[1], xyz_kde[2], cmap='Greys')

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical')
    cbar.set_label('Probability density value')
    
    #plt.scatter(dir_lons, np.pi/2 - dir_colats, marker='o', c='b', s=1, alpha=0.5, label='Simulated events')

    #SHAPLEY MEMBERS
    cords = SkyCoord(ra=shapley_coords["RAJ2000"]*u.deg, dec=shapley_coords["DEJ2000"]*u.deg, 
                        frame='icrs').transform_to("galactic")
    lon = cords.galactic.l
    lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
    lon = lon.radian
    lat = cords.galactic.b.radian
    plt.scatter(lon, lat, marker='+', c='r', s=50, label='Shapley member clusters')

    plt.xlabel("Galactic Longtitude, rad")
    plt.ylabel("Galactic Latitude, rad")
    plt.xlim(-1.5, -0.5)
    plt.ylim(0.3, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.title(f"Probability density function for the simulated particles directions distribution")
    plt.legend()
    plt.savefig("paper_plots/kde_3levels.jpeg", bbox_inches = "tight", dpi = 300)
    plt.show()   
    plt.close() 



def plot_detections_local(detections):
    from matplotlib.patches import Circle

    x, y, z, det_dir_lons, det_dir_colats = detections
    shapley_coords = pd.read_csv("shapley_with_radii.csv")
    clust_A = [14, 19, 21, ]# Full cluster [14, 19, 21, 40, 36]
    clust_A_detections = [20-2, 21-2, 22-2, 23-2]
    clust_B = [4, 10]
    clust_CD = [34, 37, 23, 31, 42]
    clust_E = [1, 3, 7, 9, 11, 13, 16]
    clust_F = [12, 15, 29, 22, 30, 39]
    clust_G = [0, 6]
    clust_H = [41]
    clust_I = [33, 43]
    clust_J = [2, 5, 17, 18, 20, 24, 28, 38, 44, 27, 32, 35]

    #plt.figure(figsize=(4,3))
    fig, ax = plt.subplots(figsize=(4, 3))

    #VISUALISING DETECTIONS
    
    for i in clust_A_detections:
        ax.scatter(det_dir_lons[i], np.pi/2 - det_dir_colats[i], marker='o', c='green', s=20, alpha=0.6, label='Detected Particles')
    
    #CIRCLES
    for i, target in shapley_coords.loc[clust_A].iterrows():
        cords = SkyCoord(ra=target["RAJ2000"]*u.deg, dec=target["DEJ2000"]*u.deg, 
                    frame='icrs').transform_to("galactic")
        lon = cords.galactic.l
        lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
        lon = lon.radian
        lat = cords.galactic.b.radian

        distance = cosmo.comoving_distance(target["z"]).value * 1000 # as radii in kpc
        radius_kpc = target["R500"]
        
        radius = radius_kpc / distance
        circle = Circle((lon, lat), radius, edgecolor='blue', facecolor='none', linewidth=1.5)
        ax.add_patch(circle)
        
        '''
        theta = np.linspace(0, 2 * np.pi, 100)  # Angles from 0 to 2*pi
        x_circle = lon + radius * np.cos(theta)  # X coordinates of the circle
        y_circle = lat + radius * np.sin(theta)  # Y coordinates of the circle

        # Plot the circle
        plt.plot(x_circle, y_circle, label=f'Circle (r={radius})', c='r')
        '''
        ax.scatter(lon, lat, marker='+', c='r', s=40, label='Shapley member clusters')
    
    #plt.scatter(lon, lat, marker='+', c='r', s=40, label='Shapley member clusters')
    #plt.text(-90*np.pi/180, 55*np.pi/180, "Shapley Supercluster", fontsize=10, fontweight='bold')

    plt.grid(True)
    plt.tight_layout()
    #plt.title(f"{i}")
    #plt.legend()
    plt.show()   
    plt.close() 


if __name__ == "__main__":
    #traj_data = np.genfromtxt('sim_results/test_trajectories_1model.txt', unpack=True, skip_footer=1)
    data_final_directions = np.genfromtxt('sim_results/test_final_directions_1model.txt', unpack=True, skip_footer=1)
    #data_detections = np.genfromtxt('sim_results/test_detections_1model.txt', unpack=True, skip_footer=1)

    
    events = []
    with open('Auger_lowE_shapley.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[5]), float(temp_event[6]), float(temp_event[7])))
    
    #plot_zero_approx_map(data_final_directions)
    #plot_detections_local(data_detections)
    #plot_detections(data_detections, 'paper_plots/detections.jpeg', events)
    #plot2DProjection(traj_data)
    #I, dir_lons, dir_colats = data_final_directions
    #x, y, z, det_dir_lons, det_dir_colats = data_detections
    from PIL import Image

    map = Image.open('paper_plots/inner_gal_base_p.jpeg')
    fig, axs = plt.subplot_mosaic([["H"]], figsize=(8, 8))#, layout='constrained')

    axs["H"].imshow(map.crop((1740, 65, 3540, 955)), aspect="auto")
    axs["H"].set_xticks([])
    axs["H"].set_yticks([])
    
    plt.show()
    #plt.imshow(map)
    #plt.show()
    #plt.show()   
    #plt.close() 
    