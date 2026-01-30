import os
import pandas as pd
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from cut_visualisation import get_objects_params, transform_pandas_galactocentric_to_galactic
from side_checks.calc_metric_for_seed_check import makeCut
from shapley_pipeline.analyse_trajectories import calculate_kde


def plot_hammer_galactic_with_candidate(ax, type_name, candidate_label, 
                                        candidate_coords_equatorial, data_cut_galactic, xyz_kde, vmax,
                                        crop = False, kde = True, cmap = 'viridis'):
    '''Plots a single Hammer projection in galactic coordinates with the candidate position and simulated trajectories.
    Also adds 1 degree and 3 degree circles around the candidate. Cropping if needed'''
    #PLOT CONFIGURATION
    #Add kde flag check

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

        #PLOTTING PRECALCULATED KDE 
        mesh = ax_temp.pcolormesh(xyz_kde[0], xyz_kde[1], xyz_kde[2], cmap=cmap, zorder=0, 
                                  vmin = 0, vmax = vmax,)
                                  #label='Simulated probability distribution')#, shading='auto')
        #PLOTTING SIMULATED TRAJECTORIES
        '''
        ax_temp.scatter(-data_cut_galactic["Lon"], data_cut_galactic["Lat"], 
                s=1, alpha=0.3, color='blue', label=f'Simulated trajectories')
        '''
        #PLOTTING CANDIDATE
        coords_candidate = SkyCoord(ra=candidate_coords_equatorial["RA"]*u.deg, dec=candidate_coords_equatorial["DEC"]*u.deg,
                        distance=candidate_coords_equatorial["dist"]*u.kpc, frame='icrs').transform_to("galactic")
        lon = coords_candidate.galactic.l
        lon.wrap_angle = 180 * u.deg
        lon = lon.radian
        lat = coords_candidate.galactic.b.radian
        ax_temp.scatter(-lon, lat, marker="P", color='red', s=0.1)#, label='Candidate', zorder=10,)
        ax_temp.text(-lon - 4*np.pi/180, lat + 5*np.pi/180, candidate_label, fontsize=3, color = 'red')
        ax_temp.set_axisbelow(True)

        #ADDING CIRCLES AROUND CANDIDATE
        circle_1deg = 1 * (np.pi/180)  # 1 degree in radians
        circle_3deg = 3 * (np.pi/180)  # 3 degrees in radians       
        circle_1deg_x = circle_1deg * np.cos(np.linspace(0, 2 * np.pi, 100))
        circle_1deg_y = circle_1deg * np.sin(np.linspace(0, 2 * np.pi, 100))
        circle_3deg_x = circle_3deg * np.cos(np.linspace(0, 2 * np.pi, 100))
        circle_3deg_y = circle_3deg * np.sin(np.linspace(0, 2 * np.pi, 100))
        ax_temp.plot(-lon + circle_1deg_x, lat + circle_1deg_y, color='red', linestyle='solid', label='1 Degree Circle',
                     linewidth=0.5)
        ax_temp.plot(-lon + circle_3deg_x, lat + circle_3deg_y, color='green', linestyle='--', label='3 Degree Circle',
                     linewidth=0.5)

        #ax_temp.grid(True)
        ax_temp.legend(loc='upper right')#, fontsize='small')

        #GENERAL TICKS
        x_tick_labels = ['', '']
        x_tick_positions = [-50*np.pi/180, -35*np.pi/180]
        ax_temp.set_xticks(x_tick_positions)
        ax_temp.set_xticklabels(x_tick_labels)

        y_tick_labels = ['', '']
        y_tick_positions = [-9*np.pi/180, 9*np.pi/180]
        ax_temp.set_yticks(y_tick_positions)
        ax_temp.set_yticklabels(y_tick_labels)

        #TICKS FOR CROP
        xticks_crop = [-50*np.pi/180 + 0.5*np.pi/180, -35*np.pi/180 - 2.8*np.pi/180] # To plot on left of lat lines we add 6 degrees
        xlabels_crop = ['50°', '35°']
        ax_temp.text(xticks_crop[0], -np.pi/20 - 2.8*np.pi/180, xlabels_crop[0], fontsize=4, c = 'white')
        ax_temp.text(xticks_crop[1], -np.pi/20 - 2.8*np.pi/180, xlabels_crop[1], fontsize=4, c = 'white')
        #for pos, label in zip(xticks_crop, xlabels_crop):
        #    ax_temp.text(pos, -np.pi/20 - 3*np.pi/180, label, fontsize=4)
        
        yticks_crop = [-9*np.pi/180 + 1*np.pi/180, 9*np.pi/180 - 2*np.pi/180] # To plot on top of lon lines we add 2 degrees
        ylabels_crop = ['-9°', '9°']
        ax_temp.text(-52.4*np.pi/180, yticks_crop[0], ylabels_crop[0], fontsize=4, c = 'white')#-9 deg
        ax_temp.text(-51.8*np.pi/180, yticks_crop[1], ylabels_crop[1], fontsize=4, c = 'white')#9 deg
        #for pos, label in zip(yticks_crop, ylabels_crop):
        #    ax_temp.text(-np.pi/3 - np.pi/24, pos, label, fontsize=10)

        ax_temp.grid(True)
        #ax_temp.legend(loc='upper right', fontsize='large')
        handles, labels = ax_temp.get_legend_handles_labels()
        
        scale_factor = 2.5  # Adjust this factor as needed
        fig_temp.savefig('temp.png', dpi=600*scale_factor, bbox_inches='tight')
        plt.close(fig_temp)
        gc.collect()

        #PLOTTING CROPPED IMAGE ON ORIGINAL AX
        map = Image.open('temp.png')
        #ax.imshow(map.crop((2280, 1600, 3300, 2600)), aspect="auto") # VALID FOR TRIPLET
        #ax.imshow(map.crop((2600, 1800, 3000, 2400)), aspect="auto") # VALID FOR TRIPLET
        ax.imshow(map.crop((2500*scale_factor, 1800*scale_factor, 3100*scale_factor, 2400*scale_factor)), aspect="auto") # VALID FOR TRIPLET (higher resolution for 2400 dpi)
        #ax.imshow(map.crop((4700, 700, 5300, 1300)), aspect="auto") #VALID FOR SHAPLEY TESTING
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{type_name}", fontsize=14)
        os.remove('temp.png')
        map.close()
        del map
        gc.collect()

        return handles, labels

def plot_combined_figure_each_seed(targets_names, simulations_directory, seeds,
                         target_coords_galactocentric, target_coords_equatorial,
                         crop = True, save_name = None):
    '''Plots a combined figure with 4 images each representing one of the types (candidate or particle type etc)'''

    for seed in tqdm(seeds):
        mag_field = 'JF12'
        particle = 'C'
        event_nums = [22, 23, 30]
        sim_num = 1000

        target_labels = {'sgr': 'SGR 1900+14', 'grs': 'GRS 1915+105', 'ngc': 'NGC 6760', 
                         'ss': 'SS 433', 'shapley': 'J125113.4-223227'}
        
        for event_num in event_nums:

            if not crop: fig, axes = plt.subplots(2, 2, figsize=(16, 9), subplot_kw={'projection': 'hammer'})
            if crop: fig, axes = plt.subplots(2, 2, figsize=(16, 9))
            # Flatten the 2x2 array into a 1D array (length 4) for easy looping
            axes_flat = axes.flatten()
            #Iterating through all objects to find and plot candidates
            
            for target_name in targets_names:
                #hit_stats = pd.read_csv(os.path.join(targets_directory, f"{target_name}/hit_statistics.csv"))
                #ADD HIT STATS EVALUATION IF ELSE
                #Getting simulated data and performing cut
                sim_types = ['base', 'striated', 'turbulent', 'striated+turbulent']
                save_name = os.path.join(save_directory, f"{mag_field}/{particle}/per_seed_plots/combined_cropped_maps_{event_num}_event_{target_name}_seed{seed}.jpeg")

                for i, sim_type in enumerate(sim_types):
                    sim_data = np.genfromtxt(f'{simulations_directory}/{mag_field}/{particle}/{sim_type}/traj_PA+TA_{particle}_{event_num}_event_{sim_num}sims_seed{seed}.txt', 
                                                        unpack=True, skip_footer=1)

                    #Galactocentric coords
                    data_cut = makeCut(sim_data, target_coords_galactocentric[target_name], rot=False)
                    # Transform galactocentric coordinates to galactic coordinates
                    lon, lat = transform_pandas_galactocentric_to_galactic(data_cut)
                    data_cut_galactic = pd.DataFrame({'Lon': lon, 'Lat': lat})
                    
                    #Plotting
                    handles, labels = plot_hammer_galactic_with_candidate(ax = axes_flat[i], type_name = sim_type,
                                                                        candidate_label = target_labels[target_name],
                                                        candidate_coords_equatorial = target_coords_equatorial[target_name],
                                                        data_cut_galactic = data_cut_galactic, crop = crop)

            plt.tight_layout()
            plt.suptitle(f"Galactic Coordinates Trajectory Distribution for {particle} nuclei {event_num} event", fontsize=20, y=1.02)
            fig.legend(handles=handles, labels=labels)
            if save_name: plt.savefig(save_name, bbox_inches='tight', dpi=300)
            plt.close()
            #plt.show()

def gather_data_for_kde(particle, mag_field, target_name, simulations_directory, seeds,
                        target_coords_galactocentric, save_directory):  
    '''Gathers all data into a single DataFrame for KDE plotting For one single particle type for one specific
    target'''

    csv_save_path = os.path.join(save_directory, f"{mag_field}/{particle}/combined_plots_and_artifacts/kde_data_{target_name}.csv")
    #Check if data has already been calculated and saved
    if os.path.exists(csv_save_path):
        print(f"Found existing data at {csv_save_path}. Loading...")
        master_df = pd.read_csv(csv_save_path)

        return master_df

    data_accumulator = []
    event_nums = [2, 40, 74]
    sim_num = 1000
    sim_types = ['base', 'striated', 'turbulent', 'striated+turbulent']

    for event_num in event_nums:
        for seed in tqdm(seeds, desc = "Seed", leave = True):          
            #Getting simulated data and performing cut
            for i, sim_type in enumerate(sim_types):
                sim_data = np.genfromtxt(f'{simulations_directory}/{mag_field}/{particle}/{sim_type}/traj_PA+TA_{particle}_{event_num}_event_{sim_num}sims_seed{seed}.txt', 
                                                    unpack=True, skip_footer=1)

                #Galactocentric coords
                data_cut = makeCut(sim_data, target_coords_galactocentric[target_name], rot=False)
                # Transform galactocentric coordinates to galactic coordinates
                lon, lat = transform_pandas_galactocentric_to_galactic(data_cut)

                batch_df = pd.DataFrame({
                    'Lon': lon,
                    'Lat': lat,
                    'SimType': sim_type,
                    'Seed': seed,
                    'EventNum': event_num  
                })
                
                data_accumulator.append(batch_df)

    # Concatenate all batches into a single DataFrame
    particle_df = pd.concat(data_accumulator, ignore_index=True)

    # Optional save
    particle_df.to_csv(csv_save_path, index=False)

    return particle_df


def plot_combined_figure_kde(targets_names, simulations_directory, seeds,
                             target_coords_galactocentric, target_coords_equatorial,
                             particle, mag_field, save_directory, crop = False):
    '''Plots a combined figure with 4 images each representing one of the types of mag field components using KDE maps
    for all seeds'''
    target_name = targets_names[0]  # Currently only supports one target at a time
    
    target_labels = {'sgr': 'SGR 1900+14', 'grs': 'GRS 1915+105', 'ngc': 'NGC 6760', 
                         'ss': 'SS 433', 'shapley': 'J125113.4-223227'}
    
    data = gather_data_for_kde(particle = particle, mag_field = mag_field, target_name = target_name,
                               simulations_directory = simulations_directory, seeds = seeds,
                               target_coords_galactocentric = target_coords_galactocentric, save_directory = save_directory)
    
    for event_num in [2, 40, 74]:
        #Plotting figure with 4 sim types for every event number
        if not crop: fig, axes = plt.subplots(2, 2, figsize=(16, 9), subplot_kw={'projection': 'hammer'})
        if crop: fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        # Flatten the 2x2 array into a 1D array
        axes_flat = axes.flatten()
        save_name = os.path.join(save_directory, f"{mag_field}/{particle}/combined_plots_and_artifacts/combined_cropped_maps_{particle}_{event_num}_event_{target_name}_kde.jpeg")

        sim_types = ['base', 'striated', 'turbulent', 'striated+turbulent']
        type_naming_for_maps = ["REGULAR FIELD", "STRIATED FIELD", "TURBULENT FIELD", "COMBINED FIELD"]
        event_num_subset = data[data['EventNum'] == event_num]

        #Calculating all KDE before plotting to have same color scale
        kdes = []
        for i, sim_type in tqdm(enumerate(sim_types), desc="Calculating KDE", leave=True):
            sim_type_subset = event_num_subset[event_num_subset['SimType'] == sim_type]
            xyz_kde = calculate_kde(-sim_type_subset["Lon"], sim_type_subset["Lat"], 
                                    x_min_shift = 0.3, x_max_shift = 0.5, y_min_shift = 0.3, y_max_shift = 0.3,
                                    bandwidth_degrees = 1, resolution = 1000)
            if i == 0:
                vmax_global = xyz_kde[2].max()
            else:
                if xyz_kde[2].max() > vmax_global:
                    vmax_global = xyz_kde[2].max()
            kdes.append(xyz_kde)

        print(f"Global vmax for KDE plots: {vmax_global}")
        #Plotting all sim types with same color scale
        cmap = 'turbo'
        for i, sim_type in tqdm(enumerate(sim_types), desc="Plotting KDE", leave=True):
            sim_type_subset = event_num_subset[event_num_subset['SimType'] == sim_type]

            handles, labels = plot_hammer_galactic_with_candidate(ax = axes_flat[i], type_name = type_naming_for_maps[i],
                                                                  candidate_label = target_labels[target_name],
                                                                  candidate_coords_equatorial = target_coords_equatorial[target_name],
                                                                  data_cut_galactic = sim_type_subset, crop = crop, 
                                                                  xyz_kde = kdes[i], vmax = vmax_global,
                                                                  kde = True, cmap = cmap)
        
        plt.tight_layout()

        #ADDING COLORBAR
        pos_top_left = axes[0, 0].get_position()
        pos_bottom_left = axes[1, 0].get_position()
        pos_top_right = axes[0, 1].get_position()
        pos_bottom_right = axes[1, 1].get_position()

        # Calculate exact bounds for the colorbar
        bottom = pos_bottom_right.y0  # Bottom of the lower plots
        top = pos_top_right.y1        # Top of the upper plots
        height = top - bottom

        plt.subplots_adjust(right=0.9) # Leave 10% of space on the right for the bar

        # Creating a dedicated axis for the colorbar
        # Coordinates are [left, bottom, width, height] in figure fraction (0 to 1)
        # left=0.92 puts it in the empty space which was just made
        cbar_ax = fig.add_axes([0.91, bottom, 0.02, height])

        #KDE SYNTHETIC COLORBAR AS WE KNOW MIN AND MAX VALUES FROM ALL PLOTS
        norm = mcolors.Normalize(vmin=0, vmax=vmax_global)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cbar_ax, 
                            orientation='vertical')
        cbar.ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        cbar.set_label(r'N$_{events}$ / $\it{deg}^2$', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        #plt.suptitle(f"Galactic Coordinates Trajectory Distribution for {particle} nuclei {event_num} event", fontsize=20, y=1.02)
        fig.legend(handles=handles, labels=labels, fontsize=12, markerscale=10, loc='upper right', 
                   bbox_to_anchor=(0.9, 0.95))#handlelength=10
        if save_name: plt.savefig(save_name, bbox_inches='tight', dpi=600)
        plt.close()
        gc.collect()
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
    targets_names = ['sgr']
    #targets_names = ['shapley']
    #save_name = os.path.join(save_directory, f"combined_cropped_maps_C_30_event_SGR_seed4200.png")
    seeds = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    '''
    plot_combined_figure_each_seed(targets_names = targets_names, 
                         simulations_directory = simulations_directory, 
                         seeds = seeds,
                         target_coords_galactocentric = target_coords_galactocentric, 
                         target_coords_equatorial = target_coords_equatorial, 
                         crop = True,)
                         #save_name = save_name)
    '''
    plot_combined_figure_kde(targets_names = targets_names, 
                         simulations_directory = simulations_directory, 
                         seeds = seeds,
                         target_coords_galactocentric = target_coords_galactocentric, 
                         target_coords_equatorial = target_coords_equatorial, 
                         particle = 'C', mag_field = 'JF12',
                         save_directory = save_directory, crop = True,)