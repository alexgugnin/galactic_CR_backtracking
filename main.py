import typing
import matplotlib.pyplot as plt
from run_sim import runSimulation
from visualizer import SimMap
from make_data import makeDF
from PIL import Image

def setupSimulation():
    '''
    This function setups current simulation
    '''
    # magnetic field setup
    B = JF12Field()
    seed = 42
    B.randomStriated(seed)
    B.randomTurbulent(seed)
    #B = PlanckJF12bField()
    #B = TF17Field()
    #B = PT11Field()

    # simulation setup
    sim = ModuleList()
    sim.add(PropagationCK(B, 1e-4, 0.1 * parsec, 100 * parsec))
    obs = Observer()
    obs.add(ObserverSurface( Sphere(Vector3d(0), 20 * kpc) ))
    sim.add(obs)
    #print(sim)

    return sim, obs

def cropper(image_path, save_path):
    '''MOVE TO ANOTHER FILE'''
    #Temp soluton for visualisation of only local void sources
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    map_img = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(map_img.crop((1400, 800, 4000, 2700)), aspect="auto")
    ax.axis('off')

    ax.set_title("Simulated trajectories for N(14,7) nuclei", fontsize=16)
    #ax.set_title("Events from Pierre Auger (PA) and Telescope Array (TA) observatories with E > 100 EeV in direction of LV", fontsize=16)
    legend_elements = [Line2D([0], [0], color='black', lw=1, label='Galaxy clusters'),
                        Line2D([0], [0], color='purple', lw=1, label='Local Void'),
                        #Line2D([0], [0], marker='*', color='orange', label='CRs E > 100 EeV', markerfacecolor='orange', linestyle='', markersize=8),
                        Line2D([0], [0], marker='*', color='orange', label='CRs from PAO', markerfacecolor='orange', linestyle='', markersize=8),
                        Line2D([0], [0], marker='*', color='gold', label='CRs from TA', markerfacecolor='gold', linestyle='', markersize=8),
                        #Line2D([0], [0], marker='o', color='blue', label='Simulated CRs', markerfacecolor='blue', linestyle='', markersize=8), #make variable colors
                        Line2D([0], [0], marker='p', color='pink', label='Magnetars', markerfacecolor='pink', linestyle='', markersize=8),
                        Line2D([0], [0], marker='D', color='turquoise', label='Starburst galaxies', markerfacecolor='turquoise', linestyle='', markersize=8),
                        Line2D([0], [0], marker='p', color='pink', label='SGR 1900+14', markerfacecolor='pink', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='red', label='GRS 1915+105', markerfacecolor='red', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='purple', label='NGC 6760', markerfacecolor='purple', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='cyan', label='SS 433', markerfacecolor='cyan', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='green', label='MGRO 1908', markerfacecolor='green', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='magenta', label='Cygnus OB2', markerfacecolor='magenta', linestyle='', markersize=8),
                        Line2D([0], [0], marker='p', color='pink', label='SGR 2013+34', markerfacecolor='pink', linestyle='', markersize=8),
                        Line2D([0], [0], marker='p', color='pink', label='SGR 1935+2154', markerfacecolor='pink', linestyle='', markersize=8),
                        Line2D([0], [0], marker='o', color='red', label='EHECR triplet', markerfacecolor='none', markeredgecolor='red', 
                                markeredgewidth=1.5, linestyle='', markersize=8)
                        ]
    
    legend_elements = legend_elements = [
            Line2D([0], [0], marker='X', color='#466BC7', label='JF12 base', markerfacecolor='none', markeredgecolor='#466BC7', 
                    markeredgewidth=1.5, linestyle='', markersize=8),
            Line2D([0], [0], marker='D', color='#B9C76B', label='UF23 base', markerfacecolor='none', markeredgecolor='#B9C76B', 
                    markeredgewidth=1.5, linestyle='', markersize=8)
                    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, framealpha=0.8)
    fig.savefig(save_path, dpi=600, bbox_inches='tight')

    map_img.close()
    del map_img

def combiner(image_paths):
    '''MOVE TO ANOTHER FILE
    '''
    #Temp solution for visualisation of only local void sources

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        img = Image.open(image_paths[i])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.axis('off')

    plt.tight_layout()
    plt.savefig('paper_results/full_maps_LV.jpeg', dpi=600, bbox_inches='tight')

if __name__ == '__main__':
    import math
    from crpropa import *
    import pandas as pd

    '''
    Getting data
    '''
    '''
    MAIN
    '''
    '''
    events = []
    with open('data/AugerApJS2022_highE.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[6]), float(temp_event[7]), float(temp_event[8])))
    '''
    '''
    events = []
    with open('data/TA2023_highE.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[6]), float(temp_event[7]), float(temp_event[8])))
    '''
    '''
    events = []
    with open('data/auger+TA_combined.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[6]), float(temp_event[7]), float(temp_event[8])))
    '''
    events = []
    with open('data/auger+TA_combined_old_data.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[6]), float(temp_event[7]), float(temp_event[8])))
    '''
    events = []
    with open('data/Auger_lowE_shapley.dat', 'r') as infile:
        for line in infile:
            if line.split()[0][0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[5]), float(temp_event[6]), float(temp_event[7])))
    '''
    '''
    Setupping simulation
    '''
    sim, obs = setupSimulation()

    '''
    Running simulation
    '''
    #TA energy from "An extremely energetic cosmic ray observed by a surface detector array", Auger from other article check Telegram
    initial_lats, initial_lons, all_events_lats, all_events_lons = runSimulation(sim, obs, events, seed=42, 
                                                                                 #sigma_energy = (0.07, 0.15), sigma_dir = (0.002, 0.003),
                                                                                 sigma_energy = (0.07, 0.07), sigma_dir = (0.002, 0.002),  
                                                                                 num_of_sims = 1)#, unique_event = 3)

    '''
    GATHERING DATA
    '''

    total_results = makeDF(all_events_lats, all_events_lons, num_events=len(events))#num_events=59)#num_events=28)
    #total_results.to_csv('results_100sims_all_events.csv')
    _ = [i for i in zip(initial_lats, initial_lons)]
    coords_df = pd.DataFrame(_, columns=['lats', 'lons'])
    #coords_df.to_csv('initial_cords_all_events.csv')

    '''
    Visualizing results achieved
    '''
    
    map = SimMap(total_results, initial_lats, initial_lons, particles=['H'])#['H', 'aH', 'He', 'C', 'Fe']
    map.setSaveName('paper_results/full_map_sources.jpeg')
    map.setTitle("Events from Pierre Auger (PA) and Telescope Array (TA) observatories with E > 100 EeV")
    map.setSourcesFlags({'mags': True, 'sbgs': True, 'clusts': True})
    map.plotMap(sim = True, transform=True, sgr=True, grs=True, ss=True, ngc=True, milagro=True, cygnus=True,
                shapley=False, aquila=True, sgr_2013=True, sgr_1935=True,
                legend=True, saving=True, custom_frame=False)
    
    #CROPPING IMAGE FOR PAPER FOR LV REGION ONLY

    image_path = "paper_results/full_map_sources.jpeg"
    save_path = "paper_results/full_map_sources_LV_crop_trajectories_He.jpeg"
    cropper(image_path, save_path)
    '''
    #Combining images for paper
    image_paths = ["paper_results/full_map_sources_LV_crop_no_trajectories.jpeg", 
                   "paper_results/full_map_sources_LV_crop_trajectories_H.jpeg", 
                   "paper_results/full_map_sources_LV_crop_trajectories_N.jpeg",
                   "paper_results/full_map_sources_LV_crop_trajectories_Si.jpeg"]
    combiner(image_paths)
    '''
    '''
    Saving results
    '''
    #total_results.to_csv('auger31_42.csv')
