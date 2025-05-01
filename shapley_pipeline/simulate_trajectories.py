import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from crpropa import *
from astropy import units as u
from astropy.coordinates import SkyCoord

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

class ShapleyTrajectoryOutput(Module):
    """
    Custom trajectory output: i, x, y, z
    where i is a running cosmic ray number
    and x,y,z are the Galactocentric coordinates in [kpc].
    Also returns velocities to measure time.
    """
    def __init__(self, fname):
        Module.__init__(self)
        self.fout = open(fname, 'w')
        self.fout.write('#i\tX\tY\tZ\n')
        self.i = 0
    def process(self, c):
        r = c.current.getPosition()
        v = c.current.getVelocity()
        v_mod = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        x = r.x / kpc
        y = r.y / kpc
        z = r.z / kpc
        #self.fout.write('%i\t%.3f\t%.3f\t%.3f\t%.3f\n'%(self.i, x, y, z, v_mod))
        self.fout.write('%i\t%.3f\t%.3f\t%.3f\n'%(self.i, x, y, z))
        if not(c.isActive()):
            self.i += 1
    def close(self):
        self.fout.close()


if __name__ == '__main__':
    seed = 42
    R = Random(seed)

    '''
    Uploading events from the Shapley quadrant
    '''
    '''
    events = []
    with open('Auger_lowE_shapley.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[5]), float(temp_event[6]), float(temp_event[7])))
    '''
    
    events = []
    with open('auger+TA_combined.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[6]), float(temp_event[7]), float(temp_event[8])))
    
    #particles = [- nucleusId(1,1), - nucleusId(4,2), - nucleusId(12,6), - nucleusId(52,26)]
    particles = {'p': - nucleusId(1,1), 
                 'Fe': - nucleusId(52,26)
                 }

    #INSTRUMENTAL FROM PA PAPER
    sigma_energy = 0.07
    sigma_dir = 0.002 #1 degree directional uncertainty

    lons0, lats0, lons, lats = [], [], [], []
    inner_sim, gal_obs = setupInnerSimulation(seed)
    for event in tqdm(events):
        NUM_OF_SIMS = 10
        particle_charge = particles['p']

        mean_energy = event[3] * EeV
        position = Vector3d(-8.2, 0, 0.0208) * kpc
        
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
            direction_lon, direction_lat = res.getDirection().getPhi(), res.getDirection().getTheta()
            lons.append(direction_lon)
            lats.append(direction_lat)

    lons, lats, lons0, lats0 = np.array(lons), np.array(lats), np.array(lons0), np.array(lats0)
    lats = np.pi/2 - lats # From CrPropa colatitude to matplotlib latitude

    '''
    n = 0
    for idx in range(len(lons0)):
        plt.figure(figsize=(12,7))
        plt.subplot(111, projection = 'hammer')
        plt.scatter(lons0[idx], lats0[idx], marker='+', c='black', s=100)
        for i in range(10):
            plt.scatter(lons[idx+i+n], lats[idx+i+n], marker='o', c='blue', s=5, alpha=0.2)
        n += 9
        plt.grid(True)
        #plt.savefig("compar.png", dpi=300)
        plt.show()
    '''
    plt.figure(figsize=(12,7))
    plt.subplot(111, projection = 'hammer')
    plt.scatter(lons0, lats0, marker='+', c='black', s=100)
    plt.scatter(lons, lats, marker='o', c='blue', s=5, alpha=0.2)
    plt.grid(True)
    plt.savefig("compar.png", dpi=300)
    plt.show()         