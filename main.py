from useful_funcs import eqToGal
import typing
from run_sim import runSimulation
from visualizer import visualizeTotal, SimMap
from make_data import makeDF

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

    events = []
    with open('data/auger+TA_combined.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[6]), float(temp_event[7]), float(temp_event[8])))

    '''
    Setupping simulation
    '''
    sim, obs = setupSimulation()

    '''
    Running simulation
    '''
    #TA energy from "An extremely energetic cosmic ray observed by a surface detector array", Auger from other article check Telegram
    initial_lats, initial_lons, all_events_lats, all_events_lons = runSimulation(sim, obs, events, seed=42, sigma_energy = (0.07, 0.15), sigma_dir = (0.002, 0.003), num_of_sims = 1)#, unique_event = 3)

    '''
    GATHERING DATA
    '''

    total_results = makeDF(all_events_lats, all_events_lons, num_events=59)#num_events=28)
    total_results.to_csv('results_100sims_all_events.csv')
    _ = [i for i in zip(initial_lats, initial_lons)]
    coords_df = pd.DataFrame(_, columns=['lats', 'lons'])
    coords_df.to_csv('initial_cords_all_events.csv')

    '''
    Visualizing results achieved
    '''
    map = SimMap(total_results, initial_lats, initial_lons, particles=['H'])#['H', 'aH', 'He', 'C', 'Fe']
    map.setSaveName('results/conferences_2024/SGR_H.png')
    map.setTitle("Events from PA + TA observatories with E > 100 EeV")
    map.setSourcesFlags({'mags': False, 'sbgs': False, 'clusts': True})
    map.plotMap(sim = True, transform=True, sgr=True, legend=False, saving=False, custom_frame=False)

    '''
    Saving results
    '''
    #total_results.to_csv('auger31_42.csv')
