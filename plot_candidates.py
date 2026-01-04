import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm
from PIL import Image
from cut_visualisation import get_objects_params, transform_pandas_galactocentric_to_galactic
from side_checks.calc_metric_for_seed_check import makeCut

def plot_hammer_galactic_with_candidate(ax, type_name, candidate_label, 
                                        candidate_coords_equatorial, data_cut_galactic, crop = False):
    '''Plots a single Hammer projection in galactic coordinates with the candidate position and simulated trajectories.
    Also adds 1 degree and 3 degree circles around the candidate. Cropping if needed'''
    #PLOT CONFIGURATION
    if not crop:
        #USING AX PASSED AS ARGUMENT
        ax.set_title(f"{type_name}", fontsize=14)

        #PLOTTING SIMULATED TRAJECTORIES
        ax.scatter(-data_cut_galactic["Lon"], data_cut_galactic["Lat"], 
                alpha=0.5, s=2, color='blue', label=f'Simulations {type_name}')

        #PLOTTING CANDIDATE
        coords_candidate = SkyCoord(ra=candidate_coords_equatorial["RA"]*u.deg, dec=candidate_coords_equatorial["DEC"]*u.deg,
                        distance=candidate_coords_equatorial["dist"]*u.kpc, frame='icrs').transform_to("galactic")
        lon = coords_candidate.galactic.l
        lon.wrap_angle = 180 * u.deg
        lon = lon.radian
        lat = coords_candidate.galactic.b.radian
        ax.scatter(-lon, lat, marker="+", color='red', s=150, zorder=10, label='Candidate')

        #ADDING CIRCLES AROUND CANDIDATE
        circle_1deg = 1 * (np.pi/180)  # 1 degree in radians
        circle_3deg = 3 * (np.pi/180)  # 3 degrees in radians       
        circle_1deg_x = circle_1deg * np.cos(np.linspace(0, 2 * np.pi, 100))
        circle_1deg_y = circle_1deg * np.sin(np.linspace(0, 2 * np.pi, 100))
        circle_3deg_x = circle_3deg * np.cos(np.linspace(0, 2 * np.pi, 100))
        circle_3deg_y = circle_3deg * np.sin(np.linspace(0, 2 * np.pi, 100))
        ax.plot(-lon + circle_1deg_x, lat + circle_1deg_y, color='red', linestyle='--', label='1 Degree Circle')
        ax.plot(-lon + circle_3deg_x, lat + circle_3deg_y, color='green', linestyle='--', label='3 Degree Circle')

        ax.grid(True)
        ax.legend(loc='upper right', fontsize='small')
    
    if crop:
        #PLOTTING ON TEMP FIGURE TO CROP
        fig_temp = plt.figure(figsize=(16,9))
        ax_temp = plt.subplot(111, projection = 'hammer')

        ax_temp.set_title(f"{type_name}", fontsize=14)

        #PLOTTING SIMULATED TRAJECTORIES
        ax_temp.scatter(-data_cut_galactic["Lon"], data_cut_galactic["Lat"], 
                alpha=0.5, s=2, color='blue', label=f'Simulated trajectories')

        #PLOTTING CANDIDATE
        coords_candidate = SkyCoord(ra=candidate_coords_equatorial["RA"]*u.deg, dec=candidate_coords_equatorial["DEC"]*u.deg,
                        distance=candidate_coords_equatorial["dist"]*u.kpc, frame='icrs').transform_to("galactic")
        lon = coords_candidate.galactic.l
        lon.wrap_angle = 180 * u.deg
        lon = lon.radian
        lat = coords_candidate.galactic.b.radian
        ax_temp.scatter(-lon, lat, marker="+", color='red', s=150, zorder=10, label='Candidate')
        ax_temp.text(-lon, lat + 5*np.pi/180, candidate_label, fontsize=6, color = 'red')
        ax_temp.set_axisbelow(True)

        #ADDING CIRCLES AROUND CANDIDATE
        circle_1deg = 1 * (np.pi/180)  # 1 degree in radians
        circle_3deg = 3 * (np.pi/180)  # 3 degrees in radians       
        circle_1deg_x = circle_1deg * np.cos(np.linspace(0, 2 * np.pi, 100))
        circle_1deg_y = circle_1deg * np.sin(np.linspace(0, 2 * np.pi, 100))
        circle_3deg_x = circle_3deg * np.cos(np.linspace(0, 2 * np.pi, 100))
        circle_3deg_y = circle_3deg * np.sin(np.linspace(0, 2 * np.pi, 100))
        ax_temp.plot(-lon + circle_1deg_x, lat + circle_1deg_y, color='red', linestyle='solid', label='1 Degree Circle')
        ax_temp.plot(-lon + circle_3deg_x, lat + circle_3deg_y, color='green', linestyle='--', label='3 Degree Circle')

        ax_temp.grid(True)
        ax_temp.legend(loc='upper right', fontsize='small')

        #GENERAL TICKS
        x_tick_labels = ['', '', '', '', '', '', '', '', '', '', '']
        x_tick_positions = [-np.pi, -5*np.pi/6, -2*np.pi/3, -np.pi/2, -1*np.pi/3, -np.pi/6, 0, 
                            np.pi, 2*np.pi/9, 11*np.pi/36, 7*np.pi/18]
        ax_temp.set_xticks(x_tick_positions)
        ax_temp.set_xticklabels(x_tick_labels)

        y_tick_labels = ['', '']
        y_tick_positions = [-np.pi/18, np.pi/18]
        ax_temp.set_yticks(y_tick_positions)
        ax_temp.set_yticklabels(y_tick_labels)

        #TICKS FOR CROP
        xticks_crop = [-np.pi/3 + np.pi/150, -np.pi/6 + np.pi/150] # To plot on left of lat lines we add 6 degrees
        xlabels_crop = ['60°', '30°']
        for pos, label in zip(xticks_crop, xlabels_crop):
            ax_temp.text(pos, -np.pi/12 - np.pi/50, label, fontsize=8)
        
        yticks_crop = [-np.pi/18 + np.pi/90, np.pi/18 + np.pi/90] # To plot on top of lon lines we add 2 degrees
        ylabels_crop = ['-10°', '10°']
        ax_temp.text(-np.pi/3 - np.pi/28, yticks_crop[0], ylabels_crop[0], fontsize=8)#-10 deg
        ax_temp.text(-np.pi/3 - np.pi/30, yticks_crop[1], ylabels_crop[1], fontsize=8)#10 deg
        #for pos, label in zip(yticks_crop, ylabels_crop):
        #    ax_temp.text(-np.pi/3 - np.pi/24, pos, label, fontsize=10)

        ax_temp.grid(True)
        ax_temp.legend(loc='upper right', fontsize='small')
        handles, labels = ax_temp.get_legend_handles_labels()
        
        fig_temp.savefig('temp.png', dpi=600, bbox_inches='tight')
        plt.close(fig_temp)

        #PLOTTING CROPPED IMAGE ON ORIGINAL AX
        map = Image.open('temp.png')
        ax.imshow(map.crop((2280, 1600, 3300, 2600)), aspect="auto") # VALID FOR TRIPLET
        #ax.imshow(map.crop((4700, 700, 5300, 1300)), aspect="auto") #VALID FOR SHAPLEY TESTING
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{type_name}", fontsize=14)
        os.remove('temp.png')

        return handles, labels

def plot_combined_figure(targets_names, simulations_directory, seeds,
                         target_coords_galactocentric, target_coords_equatorial,
                         crop = False, save_name = None):
    '''Plots a combined figure with 4 images each representing one of the types (candidate or particle type etc)'''

    for seed in seeds:
        if not crop: fig, axes = plt.subplots(2, 2, figsize=(16, 9), subplot_kw={'projection': 'hammer'})
        if crop: fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        # Flatten the 2x2 array into a 1D array (length 4) for easy looping
        axes_flat = axes.flatten()
        #Iterating through all objects to find and plot candidates

        mag_field = 'JF12'
        particle = 'C'
        event_num = 30
        sim_num = 1000

        target_labels = {'sgr': 'SGR 1900+14', 'grs': 'GRS 1915+105', 'ngc': 'NGC 6760', 
                         'ss': 'SS 433', 'shapley': 'J125113.4-223227'}
        
        for target_name in targets_names:
            #hit_stats = pd.read_csv(os.path.join(targets_directory, f"{target_name}/hit_statistics.csv"))
            #ADD HIT STATS EVALUATION IF ELSE
            #Getting simulated data and performing cut
            sim_types = ['base', 'striated', 'turbulent', 'striated+turbulent']
            save_name = os.path.join(save_directory, f"combined_cropped_maps_{particle}_{event_num}_event_{target_name}_seed{seed}.png")

            for i, sim_type in enumerate(tqdm(sim_types)):
                sim_data = np.genfromtxt(f'{simulations_directory}/{mag_field}/{particle}/{sim_type}/traj_PA+TA_{particle}_{event_num}_event_{sim_num}sims_seed{seed}.txt', 
                                                    unpack=True, skip_footer=1)

                #Galactocentric coords
                data_cut = makeCut(sim_data, target_coords_galactocentric[target_name], rot=False)
                # Transform galactocentric coordinates to equatorial coordinates
                lon, lat = transform_pandas_galactocentric_to_galactic(data_cut)
                data_cut_galactic = pd.DataFrame({'Lon': lon, 'Lat': lat})
                
                #Plotting
                handles, labels = plot_hammer_galactic_with_candidate(ax = axes_flat[i], type_name = sim_type,
                                                                      candidate_label = target_labels[target_name],
                                                    candidate_coords_equatorial = target_coords_equatorial[target_name],
                                                    data_cut_galactic = data_cut_galactic, crop = crop)

        plt.tight_layout()
        plt.suptitle("Galactic Coordinates Trajectory Distribution for C nuclei PA event", fontsize=20, y=1.02)
        fig.legend(handles=handles, labels=labels)
        if save_name: plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.close()
        #plt.show()

if __name__ == "__main__":
    #Defining directory where our statistics lies
    targets_directory = "paper_results/projections_and_statistics/"

    #Defining directory where plots of candidates will be saved
    save_directory = "paper_results/projections_and_statistics/candidates_plots/"

    #Defining directory with simulations on which statistics were calculated
    simulations_directory = f"trajectories_data"

    #Checking metrics for candidates
    target_coords_galactocentric, distances, target_coords_equatorial = get_objects_params()
    #targets_names = list(target_coords_galactocentric.keys())[:1] # ALL AVAILABLE OBJECTS
    targets_names = ['ngc']
    #targets_names = ['shapley']
    #save_name = os.path.join(save_directory, f"combined_cropped_maps_C_30_event_SGR_seed4200.png")
    seeds = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    plot_combined_figure(targets_names = targets_names, 
                         simulations_directory = simulations_directory, 
                         seeds = seeds,
                         target_coords_galactocentric = target_coords_galactocentric, 
                         target_coords_equatorial = target_coords_equatorial, 
                         crop = True,)
                         #save_name = save_name)