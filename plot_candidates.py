import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from cut_visualisation import get_objects_params
from side_checks.calc_metric_for_seed_check import makeCut

def plot_hammer_equatorial_with_candidate(candidate_coords_equatorial, data_cut):
    '''Plots a Hammer projection in equatorial coordinates with the candidate position and simulated trajectories.
    Saves both full projection and zoomed-in version around the candidate. Also adds 1 degree and 3 degree circles 
    around the candidate.'''
    #PLOT CONFIGURATION
    plt.figure(figsize=(16,9))
    ax = plt.subplot(111, projection = 'hammer')
    ax.grid(True)
    ax.set_title("Hammer Projection in Equatorial Coordinates", fontsize=16)

    #OBTAINING SIMULATED TRAJECTORIES IN EQUATORIAL COORDINATES
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
    simulation_ra = equatorial_coords.ra.deg
    simulation_dec = equatorial_coords.dec.deg

    #PLOTTING SIMULATED TRAJECTORIES
    ax.scatter(np.radians(simulation_ra), np.radians(simulation_dec))

    #PLOTTING CANDIDATE
    #ax.scatter(np.radians(candidate_coords_equatorial.ra.wrap_at(180 * u.deg).deg),
    #           np.radians(candidate_coords_equatorial.dec.deg),) 
    ax.scatter(np.radians(candidate_coords_equatorial["RA"]), np.radians(candidate_coords_equatorial["DEC"]), marker="+")
    plt.show()



if __name__ == "__main__":
    #Defining directory where our statistics lies
    targets_directory = "paper_results/projections_and_statistics/"

    #Defining directory where plots of candidates will be saved
    save_directory = "paper_results/projections_and_statistics/candidates_plots/"

    #Defining directory with simulations on which statistics were calculated
    simulations_directory = f"trajectories_data"

    #Checking metrics for candidates
    target_coords_galactocentric, distances, target_coords_equatorial = get_objects_params()
    targets_names = list(target_coords_galactocentric.keys())[2:] # ALL AVAILABLE OBJECTS

    #Iterating through all objects to find and plot candidates
    for target_name in targets_names:
        hit_stats = pd.read_csv(os.path.join(targets_directory, f"{target_name}/hit_statistics.csv"))
        #ADD HIT STATS EVALUATION IF ELSE
        #Getting simulated data and performing cut
        mag_field = 'JF12'
        particle = 'C'
        sim_type = 'base'
        event_num = 30
        sim_num = 10000

        sim_data = np.genfromtxt(f'{simulations_directory}/{mag_field}/{particle}/{sim_type}/traj_PA+TA_{particle}_{event_num}_event_{sim_num}sims.txt', 
                                            unpack=True, skip_footer=1)
        data_cut = makeCut(sim_data, target_coords_galactocentric[target_name], rot=False)

        #Plotting results
        plot_hammer_equatorial_with_candidate(candidate_coords_equatorial = target_coords_equatorial[target_name],
                                              data_cut = data_cut)
        exit()