from crpropa import *
from useful_funcs import eqToGal
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from side_checks.calc_metric_for_seed_check import makeCut, calcPerpPlane, calcEdge, r_mat, calculate_kde, calculate_hit, calculate_mahalanobis
import astropy.units as u
from astropy.coordinates import SkyCoord

import glob
import os

def get_objects_params():
    distances = {
        "sgr": 12.5, #2.9+-0.2, 8.1+-0.5 https://arxiv.org/pdf/2308.03484, 12.5, 3.8
        "grs": 8.6, #+2-1.6  https://arxiv.org/pdf/1409.2453
        "ss": 5.5, #+-0.2   https://www.aanda.org/articles/aa/full_html/2018/09/aa32488-17/aa32488-17.html
        "ngc": 7.58,#       https://doi.org/10.1093/mnras/stab1475
        "shapley": 12.5# Placeholder for Shapley supercluster to check turbulent mag field effects
    }
    object_coords_eq = {"sgr": {"RA": 286.8097083, "DEC": 9.3222500, "dist": distances["sgr"]}, #https://arxiv.org/pdf/2412.20050
                     "grs": {"RA": 288.798, "DEC": 10.946, "dist": distances["grs"]},  #https://swift.gsfc.nasa.gov/results/transients/GRS1915p105/ 
                     "ss": {"RA": 287.956, "DEC": 4.99, "dist": distances["ss"]},      #https://swift.gsfc.nasa.gov/results/transients/SS433/
                     "ngc": {"RA": 287.800, "DEC": 1.030, "dist": distances["ngc"]}, #https://doi.org/10.1093/mnras/stab1475
                     "shapley": {"RA": 192.8015, "DEC": -22.537, "dist": distances["shapley"]}
                     }
    
    '''
    objects_list_old = {
        "sgr": [0, d_list['sgr']*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.2, d_list['sgr']*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), d_list['sgr']*np.sin(0.77*np.pi/180) + 0.0208],
        "grs": [0, d_list['grs']*np.cos(45.37*np.pi/180)*np.cos(-0.22*np.pi/180) - 8.2, d_list['grs']*np.sin(45.37*np.pi/180)*np.cos(-0.22*np.pi/180), d_list['grs']*np.sin(-0.22*np.pi/180) + 0.0208],
        "ss": [0, d_list['ss']*np.cos(39.69*np.pi/180)*np.cos(-2.24*np.pi/180) - 8.2, d_list['ss']*np.sin(39.69*np.pi/180)*np.cos(-2.24*np.pi/180), d_list['ss']*np.sin(-2.24*np.pi/180) + 0.0208],
        "ngc": [0, d_list['ngc']*np.cos(36.11*np.pi/180)*np.cos(-3.9*np.pi/180) - 8.2, d_list['ngc']*np.sin(36.11*np.pi/180)*np.cos(-3.9*np.pi/180), d_list['ngc']*np.sin(-3.9*np.pi/180) + 0.0208]
    }
    '''
    coords_sgr = SkyCoord(ra=object_coords_eq["sgr"]["RA"]*u.deg, dec=object_coords_eq["sgr"]["DEC"]*u.deg,
                      distance=object_coords_eq["sgr"]["dist"]*u.kpc, frame='icrs')
    coords_grs = SkyCoord(ra=object_coords_eq["grs"]["RA"]*u.deg, dec=object_coords_eq["grs"]["DEC"]*u.deg,
                      distance=object_coords_eq["grs"]["dist"]*u.kpc, frame='icrs')
    coords_ss = SkyCoord(ra=object_coords_eq["ss"]["RA"]*u.deg, dec=object_coords_eq["ss"]["DEC"]*u.deg,
                      distance=object_coords_eq["ss"]["dist"]*u.kpc, frame='icrs')
    coords_ngc = SkyCoord(ra=object_coords_eq["ngc"]["RA"]*u.deg, dec=object_coords_eq["ngc"]["DEC"]*u.deg,
                      distance=object_coords_eq["ngc"]["dist"]*u.kpc, frame='icrs')
    coords_shapley = SkyCoord(ra=object_coords_eq["shapley"]["RA"]*u.deg, dec=object_coords_eq["shapley"]["DEC"]*u.deg,
                      distance=object_coords_eq["shapley"]["dist"]*u.kpc, frame='icrs')
    
    g_sgr = coords_sgr.transform_to('galactocentric') 
    g_grs = coords_grs.transform_to('galactocentric') 
    g_ss = coords_ss.transform_to('galactocentric') 
    g_ngc = coords_ngc.transform_to('galactocentric') 
    g_shapley = coords_shapley.transform_to('galactocentric')

    objects_coords_galactocentric = {
        "sgr": [0, g_sgr.x.value, g_sgr.y.value, g_sgr.z.value],
        "grs": [0, g_grs.x.value, g_grs.y.value, g_grs.z.value],
        "ss": [0, g_ss.x.value, g_ss.y.value, g_ss.z.value],
        "ngc": [0, g_ngc.x.value, g_ngc.y.value, g_ngc.z.value],
        "shapley": [0, g_shapley.x.value, g_shapley.y.value, g_shapley.z.value]
    }

    return objects_coords_galactocentric, distances, object_coords_eq

def transform_pandas_galactocentric_to_equatorial(data_cut):
    '''Transforms a pandas DataFrame with galactocentric coordinates to equatorial coordinates.
    Args:
        data_cut (pd.DataFrame): DataFrame with 'X', 'Y', 'Z' columns in galactocentric coordinates.
    Returns:
        ra (np.ndarray): Right Ascension in degrees.
        dec (np.ndarray): Declination in degrees.
    '''

    #TRANSFORM GALACTOCENTRIC TO EQUATORIAL
    galactocentric_coords = SkyCoord(
        x=data_cut['X'] * u.kpc,
        y=data_cut['Y'] * u.kpc,
        z=data_cut['Z'] * u.kpc,
        representation_type = 'cartesian',
        frame = 'galactocentric'
    )

    #Transform the coordinates to the Galactic frame
    galactic_coords = galactocentric_coords.transform_to('galactic')

    #Transform the Galactic coordinates to the Equatorial frame (ICRS)
    equatorial_coords = galactic_coords.transform_to('icrs')

    # Extract Right Ascension and Declination in degrees
    ra = equatorial_coords.ra.deg
    dec = equatorial_coords.dec.deg

    return ra, dec

def transform_pandas_galactocentric_to_galactic(data_cut):
    '''Transforms a pandas DataFrame with galactocentric coordinates already wrapped for simulation to galactic coordinates.
    Args:
        data_cut (pd.DataFrame): DataFrame with 'X', 'Y', 'Z' columns in galactocentric coordinates.
    Returns:
        l (np.ndarray): Galactic longitude in degrees.
        
        b (np.ndarray): Galactic latitude in degrees.
    '''
    
    #TRANSFORM GALACTOCENTRIC TO GALACTIC
    galactocentric_coords = SkyCoord(
        x=data_cut['X'] * u.kpc,
        y=data_cut['Y'] * u.kpc,
        z=data_cut['Z'] * u.kpc,
        representation_type = 'cartesian',
        frame = 'galactocentric'
    )

    # Extract galactic lon and galactic lat in degrees
    #IF WE TRANSFORM FROM GALACTOCENTRIC TO GALACTIC, NO DONT NEED TO TRANSFORM FROM COLATITUDE!!!!!!!!!
    lon = galactocentric_coords.galactic.l
    lon.wrap_angle = 180 * u.deg
    lon = lon.radian
    lat = galactocentric_coords.galactic.b.radian
    #colat = galactocentric_coords.galactic.b.radian
    #lat = np.pi/2 - colat #Transform from colatitiude
    
    return lon, lat

def plot3D(data, objects=None) -> None:
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111, projection='3d')

    # plot trajectories
    I,X,Y,Z = data
    for i in np.unique(I):
        if i > 50: break
        ax.plot(X[I == i], Y[I == i], Z[I == i], lw=1, alpha=1, c='g')

    # plot Galactic border
    r = 20
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100))
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    ax.plot_surface(x, y, z, rstride=2, cstride=2, color='r', alpha=0.1, lw=0)
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='k', alpha=0.5, lw=0.1)

    # plot Galactic center
    ax.scatter(0,0,0, marker='o', color='r')
    # plot Earth
    ax.scatter(-8.2,0,0.0208, marker='P', color='b')

    #Plotting potential sources
    #Plot SGR 1900+14
    #for d in [2.9-0.2, 2.9, 2.9+0.2]:

    '''
    for d in [8.1-0.5, 8.1-0.3, 8.1-0.1, 8.1+0.1, 8.1+0.3, 8.1+0.5]:
        sgr_cords=[0, d*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.2, d*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), d*np.sin(0.77*np.pi/180)]
        ax.scatter(sgr_cords[1], sgr_cords[2], sgr_cords[3], marker='+', c='r', s=70) #43.02 0.77 12.5±1.7
    '''
    '''
    ax.scatter(objects['sgr'][1], objects['sgr'][2], objects['sgr'][3], marker='+', c='red', s=70)
    #plt.text(0.751-2*0.751 - 15*np.pi/180, 0.0135+7*np.pi/180, 'SGR 1900+14', fontsize=8, fontweight='bold')
    #Plot GRS 1915+105
    ax.scatter(objects['grs'][1], objects['grs'][2], objects['grs'][3], marker='+', c='green', s=70) #45.37 -0.22 8.6+2.0-1.6
    #Plot SS 433 Мікроквазар 39.69 -2.24 5.5±0.2
    ax.scatter(objects['ss'][1], objects['ss'][2], objects['ss'][3], marker='+', c='purple', s=70) #39.69 -2.24 5.5±0.2
    #Plot NGC 6760 Кулясте скупчення 36.11 -3.9 7.4±0.4
    ax.scatter(objects['ngc'][1], objects['ngc'][2], objects['ngc'][3], marker='+', c='magenta', s=70) #36.11 -3.9 7.4±0.4
    '''
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=1, label='Simulated CRs'),
                        Line2D([0], [0], marker='+', color='red', label='SGR 1900+14', markerfacecolor='red', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='green', label='GRS 1915+105', markerfacecolor='green', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='purple', label='SS 433', markerfacecolor='purple', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='magenta', label='NGC 6760', markerfacecolor='magenta', linestyle='', markersize=8)
                        ]
    fig.legend(handles=legend_elements, loc='upper right')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlabel('x / kpc', fontsize=18)
    ax.set_ylabel('y / kpc', fontsize=18)
    ax.set_zlabel('z / kpc', fontsize=18)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    ax.set_zlim((-20, 20))
    ax.xaxis.set_ticks((-20,-10,0,10,20))
    ax.yaxis.set_ticks((-20,-10,0,10,20))
    ax.zaxis.set_ticks((-20,-10,0,10,20))
    plt.show()

def plot3D_zoomed(data, data_cut, target_cords, zoom_kpc=2.0):
    '''
    Creates a 3D plot zoomed in on the target object.
    
    Args:
        data (tuple): The raw trajectory data (I, X, Y, Z)
        data_cut (pd.DataFrame): The intersection points from makeCut
        target_cords (list): The [0, x, y, z] coords of the *single* target
        zoom_kpc (float): The size of the "box" to zoom in on (e.g., +/- 2 kpc)
    '''
    
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, projection='3d')

    # 1. Plot trajectories (limit to 50, as before)
    I, X, Y, Z = data
    for i in np.unique(I):
        #if i > 50: break
        # Plot trajectory in light green
        ax.plot(X[I == i], Y[I == i], Z[I == i], lw=1, alpha=0.3, c='g')

    # 2. Plot the intersection points (THE "HITS")
    ax.scatter(data_cut['X'], data_cut['Y'], data_cut['Z'], 
               marker='x', c='k', s=50, label='Intersections')

    # 3. Plot the target object (the one we are zoomed in on)
    target_x, target_y, target_z = target_cords[1], target_cords[2], target_cords[3]
    ax.scatter(target_x, target_y, target_z, 
               marker='*', c='red', s=250, label='Target Object', 
               edgecolors='black', zorder=10)

    # 4. Plot Earth and Galactic Center (for context, if they are in view)
    ax.scatter(-8.122, 0, 0.0208, marker='P', c='b', s=100, label='Earth')
    ax.scatter(0, 0, 0, marker='o', c='k', s=100, label='Galactic Center')

    # 5. Plot the target plane
    norm, D_plane = calcPerpPlane(target_cords)
    A, B, C = norm

    # Create a grid to plot the plane. We must solve for one variable.
    # To be robust, we solve for the variable with the largest |coefficient|
    # This prevents dividing by zero if the plane is aligned with an axis.
    
    # Define a grid based on the zoomed-in coordinates
    x_grid = np.linspace(target_x - zoom_kpc, target_x + zoom_kpc, 10)
    y_grid = np.linspace(target_y - zoom_kpc, target_y + zoom_kpc, 10)
    z_grid = np.linspace(target_z - zoom_kpc, target_z + zoom_kpc, 10)
    
    abs_coeffs = np.abs(norm)
    if abs_coeffs[2] >= abs_coeffs[0] and abs_coeffs[2] >= abs_coeffs[1]:
        # Plane is "flattest" in Z, solve for z(x,y)
        xx, yy = np.meshgrid(x_grid, y_grid)
        zz = (-A * xx - B * yy - D_plane) / C
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='cyan', rstride=1, cstride=1)
    elif abs_coeffs[1] >= abs_coeffs[0]:
        # Plane is "flattest" in Y, solve for y(x,z)
        xx, zz = np.meshgrid(x_grid, z_grid)
        yy = (-A * xx - C * zz - D_plane) / B
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='cyan', rstride=1, cstride=1)
    else:
        # Plane is "flattest" in X, solve for x(y,z)
        yy, zz = np.meshgrid(y_grid, z_grid)
        xx = (-B * yy - C * zz - D_plane) / A
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='cyan', rstride=1, cstride=1)

    # 6. SET THE ZOOMED LIMITS
    ax.set_xlim(target_x - zoom_kpc, target_x + zoom_kpc)
    ax.set_ylim(target_y - zoom_kpc, target_y + zoom_kpc)
    ax.set_zlim(target_z - zoom_kpc, target_z + zoom_kpc)

    ax.set_xlabel('x / kpc', fontsize=18)
    ax.set_ylabel('y / kpc', fontsize=18)
    ax.set_zlabel('z / kpc', fontsize=18)
    ax.legend()
    plt.show()

def plot3D_from_pandas(data, data_cut, objects, target_transformed = None, norms=None, save_file = None):
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111, projection='3d')

    # plot trajectories
    I,X,Y,Z = data
    for i in np.unique(I):
        if i > 50: break
        ax.plot(X[I == i], Y[I == i], Z[I == i], lw=1, alpha=0.2, c='g')
    #Plot cut surface
    ax.scatter(data_cut['X'], data_cut['Y'], data_cut['Z'], lw=1, alpha=1, c='r', s=10)

    # plot Galactic border
    r = 20
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100))
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    ax.plot_surface(x, y, z, rstride=2, cstride=2, color='r', alpha=0.1, lw=0)
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='k', alpha=0.5, lw=0.1)

    # plot Galactic center
    ax.scatter(0,0,0, marker='o', color='yellow')
    # plot Earth
    ax.scatter(-8.2,0,0.0208, marker='P', color='b')
    
    if norms:
        earth = [-8.2,0,0.0208]
        #Plot Normale
        ax.quiver(
                earth[0], earth[1], earth[2], 
                norms[0][0], norms[0][1], norms[0][2], color='r', label='Base norm'
            )
        ax.quiver(
                earth[0], earth[1], earth[2], 
                norms[1][0], norms[1][1], norms[1][2], color='g', label='Base norm'
            )
        ax.quiver(
                earth[0], earth[1], earth[2], 
                norms[2][0], norms[2][1], norms[2][2], color='b', label='Base norm'
            )

    #Plotting potential sources

    #plt.text(0.751-2*0.751 - 15*np.pi/180, 0.0135+7*np.pi/180, 'SGR 1900+14', fontsize=8, fontweight='bold')
    #Plot SGR 1900+14
    ax.scatter(objects['sgr'][1], objects['sgr'][2], objects['sgr'][3], marker='+', c='r', s=70) #OLD CORDS 43.02 0.77 12.5±1.7
    #Plot GRS 1915+105
    ax.scatter(objects['grs'][1], objects['grs'][2], objects['grs'][3], marker='+', c='green', s=70) #45.37 -0.22 8.6+2.0-1.6
    #Plot SS 433 Мікроквазар 39.69 -2.24 5.5±0.2
    ax.scatter(objects['ss'][1], objects['ss'][2], objects['ss'][3], marker='+', c='purple', s=70) #39.69 -2.24 5.5±0.2
    #Plot NGC 6760 Кулясте скупчення 36.11 -3.9 7.4±0.4
    ax.scatter(objects['ngc'][1], objects['ngc'][2], objects['ngc'][3], marker='+', c='magenta', s=70) #36.11 -3.9 7.4±0.4
    if target_transformed is not None:
        ax.scatter(target_transformed[0], target_transformed[1], target_transformed[2], marker='+', c='black', s=80) #36.11 -3.9 7.4±0.4

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=1, label='Simulated CRs'),
                        Line2D([0], [0], marker='+', color='red', label='SGR 1900+14', markerfacecolor='red', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='green', label='GRS 1915+105', markerfacecolor='green', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='purple', label='SS 433', markerfacecolor='purple', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='magenta', label='NGC 6760', markerfacecolor='magenta', linestyle='', markersize=8)
                        ]
    fig.legend(handles=legend_elements, loc='upper right')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlabel('x / kpc', fontsize=18)
    ax.set_ylabel('y / kpc', fontsize=18)
    ax.set_zlabel('z / kpc', fontsize=18)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    ax.set_zlim((-20, 20))
    ax.xaxis.set_ticks((-20,-10,0,10,20))
    ax.yaxis.set_ticks((-20,-10,0,10,20))
    ax.zaxis.set_ticks((-20,-10,0,10,20))
    if save_file is not None: plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot2D_projection_orthogonal(x, z, target, radius, xyz_kde, save_name=None) -> None:
    import seaborn as sns
    '''Func for plotting XZ projection with target object and 1 degree circle around it'''

    theta = np.linspace(0, 2 * np.pi, 100)  # Angles from 0 to 2*pi
    x_circle = target[0] + radius * np.cos(theta)  # X coordinates of the circle
    z_circle = target[2] + radius * np.sin(theta)  # Y coordinates of the circle

    # Plot the circle
    plt.figure(figsize=(12, 8))
    plt.plot(x_circle, z_circle, label=f'Circle (r={radius})', c='r')

    '''Simple 2D scatterplot'''
    
    plt.scatter(x, z, s = 5, label="Trajectories crossing the object's plane")
    plt.scatter(target[0], target[2], c='r', label='Target object')
    plt.text(target[0]-0.15, target[2]+0.06, 'SGR 1900+14', fontsize=10, fontweight='bold', c = 'red')
    plt.xlabel("Cartesian coordinate X, kPc")
    plt.ylabel("Cartesian coordinate Z, kPc")
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()
    

    '''KDE colormesh with levels'''
    '''
    confidence_levels = [0.6827, 0.9545, 0.9973]
    Z_flat = xyz_kde[2].flatten()
    Z_sorted = np.sort(Z_flat)[::-1]
    cumsum = np.cumsum(Z_sorted)
    cumsum /= cumsum[-1]
    levels = sorted([Z_sorted[np.searchsorted(cumsum, cl)] for cl in confidence_levels])

    plt.contour(xyz_kde[0], xyz_kde[1], xyz_kde[2], levels=levels, colors=['blue', 'green', 'red'])
    plt.pcolormesh(xyz_kde[0], xyz_kde[1], xyz_kde[2], cmap='plasma')
    plt.scatter(target[0], target[2], marker='+', c='red', s=80, label = 'SGR 1900+14')
    plt.text(target[0]-0.15, target[2]+0.06, 'SGR 1900+14', fontsize=10, fontweight='bold', c = 'red')
    #plt.legend()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    '''
    '''Histplot'''
    '''
    plt.hist2d(x, z, bins=30, cmap='binary')
    cb = plt.colorbar()
    cb.set_label('counts in bin')
    plt.scatter(target[0], target[2], marker='+', c='red', s=80, label = "Target object")
    plt.xlabel('X, kpc')
    plt.ylabel('Z, kpc')
    #plt.text(8.2, 0.3, 'SGR 1900+14', fontsize=10, fontweight='bold', c = 'red')
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()
    '''

    '''Joint histplot'''
    '''
    nbins = int(np.ceil(np.log2(len(x))) + 1)
    g0 = sns.jointplot(x=np.array(x), y=np.array(z), kind="hist", height=8, ratio=6,
                       marginal_ticks=True,
                       marginal_kws=dict(bins=nbins, fill=True))
    #g0.plot_joint(sns.kdeplot, color="grey", zorder=0, levels = [0.003, 0.05, 0.32])#levels=[0.68, 0.95, 0.997])
    g0.ax_joint.scatter(target[0], target[2], marker='+', c='red', s=80)
    sns.kdeplot(x = x, y = z, color="grey", zorder=0, levels = [0.003, 0.05, 0.32], ax=g0.ax_joint)
    g0.ax_joint.text(target[0], target[2], 'SGR 1900+14', fontsize=9, fontweight='bold', c = 'red')

    g0.set_axis_labels('X, kpc', 'Z, kpc', fontsize=10)
    g0.figure.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    '''

def plot2D_projection_equatorial_nonrot(ra, dec, target, radius, radec_kde=0, save_name=None) -> None:
    import seaborn as sns
    '''Func for plotting RA DEC projection with target object and 1 degree circle around it'''

    theta = np.linspace(0, 2 * np.pi, 100)  # Angles from 0 to 2*pi
    ra_circle = target['RA'] + radius * np.cos(theta)  # RA coordinates of the circle
    dec_circle = target['DEC'] + radius * np.sin(theta)  # DEC coordinates of the circle

    # Plot the circle
    plt.figure(figsize=(12, 8))
    plt.plot(ra_circle, dec_circle, label=f'Circle (r={radius})', c='r')

    '''Simple 2D scatterplot'''

    plt.scatter(ra, dec, s=5, label="Trajectories crossing the object's plane")
    plt.scatter(target['RA'], target['DEC'], c='r', label='Target object')
    plt.text(target['RA']-0.15, target['DEC']+0.06, 'SGR 1900+14', fontsize=10, fontweight='bold', c = 'red')
    plt.xlabel("RA, degrees")
    plt.ylabel("Dec, degrees")
    plt.legend()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

def plot_xz_from_above_projection(data, data_cut, target_coords_galactocentric, save_name=None) -> None:
    I, X, Y, Z = data
    norm, D_plane = calcPerpPlane(target_coords_galactocentric)
    
    #Plotting data, cut data and target in XY plane from above
    plt.plot(X, Y, alpha=0.1)
    plt.plot(data_cut['X'], data_cut['Y'], 'x')
    plt.plot(target_coords_galactocentric[1], target_coords_galactocentric[2], 'ro')

    #Add the perpendicular plane as a line
    # We are plotting the line Ax + By + D = 0 (the plane's z=0 intercept)
    A, B, C = norm
    x_lim = plt.xlim()  # Get current plot limits
    x_line = np.linspace(x_lim[0], x_lim[1], 10)

    # Need to check for vertical line (B=0)
    if abs(B) > 1e-6:
        y_line = (-A * x_line - D_plane) / B
        plt.plot(x_line, y_line, 'g--', linewidth=2, label='Target Plane (at z=0)')
    else:
        # Line is vertical: x = -D/A
        x_vert = -D_plane / A
        y_lim = plt.ylim()
        plt.plot([x_vert, x_vert], y_lim, 'g--', linewidth=2, label='Target Plane (at z=0)')

    # Define Earth's XY coordinates
    earth_x = -8.122
    earth_y = 0

    # Plot Earth's position
    plt.plot(earth_x, earth_y, 'co', markersize=10, label='Earth (XY Proj.)')

    # Draw the arrow. The vector components (A, B) are exactly
    # (target_x - earth_x) and (target_y - earth_y).
    # We can get the plot width to set a reasonable head_width
    plot_width = x_lim[1] - x_lim[0]
    plt.arrow(earth_x, earth_y, 
            A, B,  # A is dx, B is dy
            color='m', 
            head_width=plot_width / 40, 
            length_includes_head=True,
            label='Line of Sight (Norm)')
    '''
    #Final perp check
    A = norm[0]
    B = norm[1]

    # Create the 2D norm vector
    norm_2d = np.array([A, B])
    # Create a vector parallel to the 2D plane line
    plane_vec = np.array([B, -A])
    # Calculate the dot product
    dot_product = np.dot(norm_2d, plane_vec)

    print(f"--- Perpendicularity Check ---")
    print(f"2D Norm Vector: {norm_2d}")
    print(f"2D Plane Vector: {plane_vec}")
    print(f"Dot Product: {dot_product}")

    if np.isclose(dot_product, 0):
        print("Result: PASSED. The vectors are perpendicular.")
    else:
        print(f"Result: FAILED. Dot product is not zero: {dot_product}")
    print("--------------------------------")
    '''
    if save_name: plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == '__main__':
    #Sim params
    mag_field = 'JF12'
    particles = ['H', 'He', 'C', 'N', 'O', 'Fe']
    event_nums = [2, 40, 74]
    sim_types = ['base', 'striated', 'turbulent', 'striated+turbulent']
    sim_num = 1000
    seeds = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    #Targets
    objects_coords_galactocentric, distances, object_coords_equatorial = get_objects_params()
    targets_names = list(objects_coords_galactocentric.keys()) # All available objects
    targets_names = ['ss', 'grs', 'ngc']

    #plot3D(np.genfromtxt(f'trajectories_data/JF12/H/base/traj_PA+TA_H_40_event_1000sims_seed1000.txt', 
    #                                        unpack=True, skip_footer=1))
    #exit()

    '''
    2D SURFACE APPROACH
    '''
    #Scanning all combinations
    for target in tqdm(targets_names, desc="Targets", leave=False):
        target_results_list = []
        output_dir = f'paper_results/projections_and_statistics/{target}/'
        output_csv = f'{output_dir}hit_statistics_1000.csv'

        #Check existing calculations to avoid overwriting
        processed_combinations = set()
        if os.path.exists(output_csv):
            try:
                existing_df = pd.read_csv(output_csv)
                # Create a set of tuples (particle, event_num, sim_type) that are already done
                # We convert event_num to int to ensure matching works correctly
                for _, row in existing_df.iterrows():
                    processed_combinations.add((row['particle'], int(row['event_num']), row['sim_type']))
                print(f"Found {len(existing_df)} existing records for {target}. Skipping them.")
            except pd.errors.EmptyDataError:
                pass # File exists but is empty
        
        for seed in tqdm(seeds, desc="Seeds", leave=False):
            for particle in tqdm(particles, desc="Particles", leave=False):
                for event_num in tqdm(event_nums, desc="Event Numbers", leave=False):
                    for sim_type in sim_types:

                        #Skip already processed combinations
                        if (particle, event_num, sim_type) in processed_combinations:
                            continue

                        target_coords_galactocentric = objects_coords_galactocentric[target]
                        data = np.genfromtxt(f'trajectories_data/{mag_field}/{particle}/{sim_type}/traj_PA+TA_{particle}_{event_num}_event_{sim_num}sims_seed{seed}.txt', 
                                            unpack=True, skip_footer=1)
                        data_cut = makeCut(data, target_coords_galactocentric, rot=False)

                        #Debugging plots
                        #plot3D_zoomed(data, data_cut, target_coords_galactocentric, zoom_kpc=2.0)
                        #plot_xz_from_above_projection(data, data_cut, target_coords_galactocentric, )
                                                    #save_name='paper_results/various_tests/perp_plane_view_from_above_old_approach.jpeg')

                        #Angles and hit calculation
                        count1, hit1 = calculate_hit(data_cut, target_coords_galactocentric, np.pi*1*distances[target]/180) #small angle approx is good enough here
                        count2, hit2 = calculate_hit(data_cut, target_coords_galactocentric, np.pi*2*distances[target]/180)
                        count3, hit3 = calculate_hit(data_cut, target_coords_galactocentric, np.pi*3*distances[target]/180)

                        #print(f"\n Num of trajectories: {count}, Hit is :{hit}")
                        target_results = {
                            'particle': particle,
                            'event_num': event_num,
                            'sim_type': sim_type,
                            'mag_field': mag_field,
                            'seed': seed,
                            'hit_count_1degree': count1,
                            'hit_count_2degree': count2,
                            'hit_count_3degree': count3,
                            'hit_fraction_1degree': hit1,
                            'hit_fraction_2degree': hit2,
                            'hit_fraction_3degree': hit3,
                            'intersections_found': len(data_cut) # Also a useful stat
                        }

                        #Save immediately after each calculation to avoid data loss
                        df_row = pd.DataFrame([target_results])
                        header_needed = not os.path.exists(output_csv)
                        df_row.to_csv(output_csv, mode='a', header=header_needed, index=False)
                        '''
                        #Transform to equatorial for 2D plotting
                        ra, dec = transform_pandas_galactocentric_to_equatorial(data_cut)
                        plot2D_projection_equatorial_nonrot(ra, dec, object_coords_equatorial[target],
                                                            1.0, save_name=f'paper_results/projections_and_statistics/{target}/RA_DEC_projection_{mag_field}_{particle}_{sim_type}_event{event_num}.jpeg')
                        '''
        
        #Save results to CSV
        #results_df = pd.DataFrame(target_results_list)
        #results_df.to_csv(output_csv, index=False)
        #print(f"\n--- Successfully saved results for {target} to {output_csv} ---")