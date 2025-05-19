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
    #Brms = 0.1*nG FIRST MODEL
    Brms = 1e-4*nG # SECOND MODEL
    #lMin = 50*kpc FIRST MODEL
    #lMax = 5*Mpc FIRST MODEL
    lMin = 1*kpc
    lMax = 50*kpc
    sIndex = 5./3.
    turbSpectrum = SimpleTurbulenceSpectrum(Brms, lMin, lMax, sIndex)
    gridprops = GridProperties(Vector3d(0), 512, 0.1*kpc)
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

    lons0, lats0, dir_lons, dir_lats, dir_r, pos_x, pos_y, pos_z, res_energies = [], [], [], [], [], [], [], [], []
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
            res_energies.append(candidate.current.getRigidity() * 1 / 10**18)#FOR PROTON in EeV

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
    galactic_results["Res_Energy, EeV"] = res_energies

    return galactic_results


def innerGalacticVisualizer(data, fname, sim_color):
    '''
    for event_id in np.unique(data["Event_Id"]):
        plt.figure(figsize=(12,7))
        plt.subplot(111, projection = 'hammer')

        #plt.scatter(data["Init_Dir_Lon"], data["Init_Dir_Lat"], marker='+', c='black', s=25)
        #plt.scatter(data["Dir_Lon"], np.pi/2 - data["Dir_CoLat"], marker='o', c='blue', s=5, alpha=0.2)

        shapley_coords = pd.read_csv("shapley_with_radii.csv")

        #CHANGE TO VECTORIZED APPROACH IN FUTURE
        for index, clust in shapley_coords.iterrows():
            #shapley_cords = {"RA": 201.9934, "DEC": -31.5014, "z": 0.0487}
            cords = SkyCoord(ra=clust["RAJ2000"]*u.deg, dec=clust["DEJ2000"]*u.deg, 
                            frame='icrs').transform_to("galactic")
            lon = cords.galactic.l
            lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
            lon = lon.radian
            lat = cords.galactic.b.radian

            plt.scatter(lon, lat, marker='+', c='r', s=50)
        
        
        plt.scatter(data["Init_Dir_Lon"][data["Event_Id"] == event_id], data["Init_Dir_Lat"][data["Event_Id"] == event_id], marker='+', c='black', s=20)
        plt.scatter(data["Dir_Lon"][data["Event_Id"] == event_id], np.pi/2 - data["Dir_CoLat"][data["Event_Id"] == event_id], marker='o', c='blue', s=5, alpha=0.2)
        plt.grid(True)
        plt.title(event_id)
        plt.show()   
        plt.close()
    '''
    plt.figure(figsize=(12,7))
    plt.subplot(111, projection = 'hammer')

    plt.scatter(data["Dir_Lon"], np.pi/2 - data["Dir_CoLat"], marker='o', c=sim_color, s=1, alpha=0.3, label='Simulated particles')
    plt.scatter(data["Init_Dir_Lon"], data["Init_Dir_Lat"], marker='*', c='orange', s=10, label='Observed events')
    
    shapley_coords = pd.read_csv("shapley_with_radii.csv")

    #for index, clust in shapley_coords.iterrows():
    #shapley_cords = {"RA": 201.9934, "DEC": -31.5014, "z": 0.0487}
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
    plt.title("Map of simulated cosmic rays inside the Galaxy")
    plt.legend()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()   
    plt.close() 

def outerGalacticVisualizer(fname_final_directions, fname_detections, inner_gal_data):
    data_final_directions = np.genfromtxt(fname_final_directions, unpack=True, skip_footer=1)
    data_detections = np.genfromtxt(fname_detections, unpack=True, skip_footer=1)

    I, dir_lons, dir_colats = data_final_directions
    x, y, z, det_dir_lons, det_dir_colats = data_detections

    plt.figure(figsize=(12,7))
    plt.subplot(111, projection = 'hammer')

    '''
    #VISUALIZING POSITIONS
    cords = SkyCoord(x = x * u.Mpc, y = y * u.Mpc, z = z * u.Mpc,
                    frame='galactocentric').transform_to("galactic")
    
    lons = cords.galactic.l
    lons.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
    lons = lons.radian
    lats = cords.galactic.b.radian
        
    plt.scatter(lons, np.pi/2 - lats, marker='o', c='blue', s=5, alpha=0.2)
    '''
    #VISUALISING DIRECTIONS   
    plt.scatter(dir_lons, np.pi/2 - dir_colats, marker='o', c='blue', s=1, alpha=0.1, label='Simulated particles')

    #VISUALISING DETECTIONS
    #plt.scatter(det_dir_lons, np.pi/2 - det_dir_colats, marker='o', c='purple', s=20, alpha=1)

    plt.scatter(inner_gal_data["Init_Dir_Lon"], inner_gal_data["Init_Dir_Lat"], marker='*', c='orange', s=10, label='Observed events')

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
    plt.title("Map of simulated cosmic rays outside the Galaxy")
    plt.legend()
    plt.savefig("paper_plots/outer_gal_Lc1Mpc_B01nG.jpeg", dpi=300, bbox_inches='tight')
    plt.show()   
    plt.close() 


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
        self.final_directions.write('#i\tLon\tCoLat\n')
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
            self.final_directions.write('%i\t%.3f\t%.3f\n'%(self.i, c.current.getDirection().getPhi(), c.current.getDirection().getTheta()))
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
    
    inner_results = innerGalacticSimulator(sigma_energy=sigma_energy, sigma_dir=sigma_dir, events=events, particle = particles['Fe'])
    #innerGalacticVisualizer(inner_results)
    #print(inner_results.head())
    #visualize_3D_shapley('test_trajectories_inner.txt')
    #exit()
    #PERFORMING SIMULATIONS OUTSIDE THE GALAXY
    '''
    sim = setupOuterSimulation()
    output = ShapleyTrajectoryOutput('test_trajectories.txt', 'test_detections.txt', 'test_final_directions.txt')
    sim.add(output)

    
    for i in tqdm(range(len(inner_results))):
        position = Vector3d(inner_results["Pos_X, kpc"][i], inner_results["Pos_Y, kpc"][i], inner_results["Pos_Z, kpc"][i]) * kpc  # position
        direction = Vector3d()  
        direction.setRThetaPhi(1, inner_results["Dir_CoLat"][i], inner_results["Dir_Lon"][i])# direction
        candidate = Candidate(particles['p'], inner_results["Res_Energy, EeV"][i] * EeV, position, direction)#Energy in Joules
        sim.run(candidate, True)
    '''
    #visualize_3D_shapley('test_trajectories.txt')
    innerGalacticVisualizer(inner_results, fname="paper_plots/inner_gal_base_Fe.jpeg", sim_color='green')
    #outerGalacticVisualizer('test_final_directions_1model.txt', 'test_detections_1model.txt', inner_results)
    


         