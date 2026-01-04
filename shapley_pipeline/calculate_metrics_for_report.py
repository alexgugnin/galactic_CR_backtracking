import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from crpropa import *
from astropy import units as u
from astropy.coordinates import SkyCoord
from analyse_trajectories import visualize_3D_shapley
from astropy.cosmology import FlatLambdaCDM
from simulate_trajectories import innerGalacticSimulator, setupInnerSimulation

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

    # INSTRUMENTAL FROM PA PAPER
    sigma_energy = 0.07
    sigma_dir = 0.002 #1 degree directional uncertainty
    shapley_coords = pd.read_csv("shapley_with_radii.csv")

    #CONVERSION TO GALACTIC COORDINATES
    cords = SkyCoord(ra=shapley_coords["RAJ2000"]*u.deg, dec=shapley_coords["DEJ2000"]*u.deg, 
                    frame='icrs').transform_to("galactic")
    lon = cords.galactic.l
    lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
    lon = lon.radian
    lat = cords.galactic.b.radian
    #ax.scatter(-lon, lat, marker='+', c='r', s=5, label='Shapley member clusters')

    cosmo = FlatLambdaCDM(H0=70.39, Om0=0.301)
    distance = cosmo.comoving_distance(shapley_coords["z"]) * 1000 # as radii in kpc
    radius_kpc = shapley_coords["R500"]*3 #3R00
    
    radius = radius_kpc / distance
    shapley_coords_galactic = pd.DataFrame({'l_rad': -lon, 'b_rad': lat, 'radius_radian': radius})

    inner_results = innerGalacticSimulator(sigma_energy=sigma_energy, sigma_dir=sigma_dir, 
                                           events=events, particle = particles['p'],
                                           seed = seed, R = R)
    inner_results_transformed = pd.DataFrame({'Dir_Lon': -inner_results["Dir_Lon"], 
                                              'Dir_Lat': np.pi/2 - inner_results["Dir_CoLat"]})
    
    '''Calculating hits for galactic approach'''
    # Extract event coordinates (Radians)
    # Assuming inner_results_transformed is your event dataframe
    evt_lon = inner_results_transformed['Dir_Lon'].values
    evt_lat = inner_results_transformed['Dir_Lat'].values

    # Convert Events to 3D Unit Vectors (x, y, z)
    # x = cos(lat) * cos(lon)
    # y = cos(lat) * sin(lon)
    # z = sin(lat)
    print(f"Converting {len(evt_lon)} events to unit vectors...")
    evt_x = np.cos(evt_lat) * np.cos(evt_lon)
    evt_y = np.cos(evt_lat) * np.sin(evt_lon)
    evt_z = np.sin(evt_lat)

    counts = []

    # Extract Target Data
    # Assuming shapley_coords_galactic has 'l_rad', 'b_rad', and 'radius_kpc' (which acts as radius in radians)
    target_lons = shapley_coords_galactic['l_rad'].values
    target_lats = shapley_coords_galactic['b_rad'].values
    target_radii = shapley_coords_galactic['radius_radian'].values # In Radians!

    print(f"Iterating over {len(target_lons)} clusters...")

    for i in range(len(target_lons)):
        # A. Get current cluster's unit vector
        t_lon = target_lons[i]
        t_lat = target_lats[i]
        
        tx = np.cos(t_lat) * np.cos(t_lon)
        ty = np.cos(t_lat) * np.sin(t_lon)
        tz = np.sin(t_lat)
        
        # B. Calculate Dot Product with ALL events at once
        # Dot Product = (x1*x2) + (y1*y2) + (z1*z2)
        # This creates an array of size N_events representing Cosine(angle)
        dot_products = (evt_x * tx) + (evt_y * ty) + (evt_z * tz)
        
        # C. Determine Threshold
        # We need points where angle < radius. 
        # Since Cosine decreases as angle increases (from 0 to pi),
        # we need Cosine(angle) > Cosine(radius).
        threshold = np.cos(target_radii[i])
        
        # D. Count Matches
        # np.sum on a boolean array counts the Trues
        count = np.sum(dot_products >= threshold)
        counts.append(count)

    # =========================================================
    # 3. Store Results
    # =========================================================
    shapley_coords_galactic['sim_hits'] = counts

    print("-" * 30)
    print(f"Total hits: {sum(counts)}")
    print(shapley_coords_galactic)

    '''Calculating hits for galactocentric approach'''
    '''
    cosmo = FlatLambdaCDM(H0=70.39, Om0=0.301)
    distance = cosmo.comoving_distance(shapley_coords["z"]) * 1000 # as radii in kpc
    radius_kpc = shapley_coords["R500"]*3 #3R00
    cords_wrapped = SkyCoord(l=lon*u.rad, b=lat*u.rad, distance=distance*u.kpc, frame='galactic').transform_to("galactocentric")
    shapley_coords_galactocentric = pd.DataFrame({'X': cords_wrapped.x, 'Y': cords_wrapped.y, 'Z': cords_wrapped.z, 
                                                  'radius_kpc': radius_kpc})
    
    #SIMULATION OF THE INNER GALACTIC TRAJECTORIES
    inner_results = innerGalacticSimulator(sigma_energy=sigma_energy, sigma_dir=sigma_dir, 
                                           events=events, particle = particles['p'],
                                           seed = seed, R = R)
    
    #PROCCEEDING TO COUNT HITs

    targets_xyz = shapley_coords_galactocentric[['X', 'Y', 'Z']].values
    target_radii = shapley_coords_galactocentric['radius_kpc'].values
    events_xyz = inner_results[['Pos_X, kpc', 'Pos_Y, kpc', 'Pos_Z, kpc']].values
    tree = cKDTree(events_xyz)
    exit()
    #innerGalacticVisualizer(inner_results)
    print(inner_results.columns)
    #visualize_3D_shapley('test_trajectories_inner.txt')
    exit()
    '''