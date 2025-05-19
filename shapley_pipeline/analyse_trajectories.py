from pylab import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm

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

if __name__ == "__main__":
    data = np.genfromtxt('test_trajectories.txt', unpack=True, skip_footer=1)
    '''
    detections = np.genfromtxt('test_detections.txt', unpack=True, skip_footer=1)
    shapley_coords = pd.read_csv("shapley_with_radii.csv")

    # trajectory points
    x, y, z, dir_lon, dir_colat = detections

    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot(111, projection='3d')

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

    ax.scatter(x, y, z, s=200, c='r')

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
    
    I, x, y, z = data
    for i in range(len(x)):
        cords = SkyCoord(x = x[i] * u.Mpc, y = y[i] * u.Mpc, z = z[i] * u.Mpc,
                        frame='galactocentric').transform_to("galactic")
        lon = cords.galactic.l
        lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
        lon = lon.radian
        lat = cords.galactic.b.radian
        
        print(lon, lat)
        coords = SkyCoord(x = x * u.Mpc, y = y * u.Mpc, z = z * u.Mpc,
                        frame='galactocentric').transform_to("galactic")
        lons = coords.galactic.l
        lons.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
        lons = lons.radian
        lats = coords.galactic.b.radian
        print(lons, lats)
        break
    '''
    data_final_directions = np.genfromtxt('test_final_directions.txt', unpack=True, skip_footer=1)
    data_detections = np.genfromtxt('test_detections.txt', unpack=True, skip_footer=1)

    I, dir_lons, dir_colats = data_final_directions
    x, y, z, det_dir_lons, det_dir_colats = data_detections
    print(x, data_final_directions)