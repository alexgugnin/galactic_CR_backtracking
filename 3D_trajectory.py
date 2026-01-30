from crpropa import *
from useful_funcs import eqToGal
import numpy as np
from tqdm import tqdm
from astropy import units as u
from astropy.coordinates import SkyCoord
import gc


class MyTrajectoryOutput(Module):
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
        #v_mod = math.sqrt(v.x**2 + v.y**2 + v.z**2)
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
    '''
    Creating events list
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
    SHAPLEY events for testing turbulent magnetic field
    
    events = []
    with open('shapley_pipeline/Auger_lowE_shapley.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[5]), float(temp_event[6]), float(temp_event[7])))
    '''
    '''
    Sim for 4 particles for 1 event(third one)
    '''
    #particles = [- nucleusId(1,1), - nucleusId(4,2), - nucleusId(12,6), 
    #           - nucleusId(14,7), - nucleusId(16,8), - nucleusId(52,26)]
    particles = [- nucleusId(1,1)] 
    particle_alias = 'H'
    mag_model_alias = 'JF12'
    #events_in_void = [16, 18, 19, 20, 22, 23, 24, 25, 30] Not actual
    #triplet = [0]#[22, 23, 30]
    triplet = [2, 40, 74]

    #Uncertainties
    sigma_energy = (0.07, 0.15) #https://arxiv.org/pdf/2206.13492, https://www.science.org/doi/10.1126/science.abo5095
    alpha, beta = -0.15, 0.962 #https://lss.fnal.gov/archive/2025/conf/fermilab-conf-25-0486.pdf
    sigma_dir = (1*np.pi/180, 1.5*np.pi/180) #1, 1.5 degree directional uncertainty

    seeds = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for seed in tqdm(seeds):
        for mag_component_alias in tqdm(['base', 'striated', 'turbulent', 'striated+turbulent'], leave=False):
            R = Random(seed)

            #Mag component setup
            if mag_component_alias == 'base':
                B = JF12Field()
            elif mag_component_alias == 'striated':
                B = JF12Field()
                B.randomStriated(seed)
            elif mag_component_alias == 'turbulent':
                B = JF12Field()
                B.randomTurbulent(seed)
            elif mag_component_alias == 'striated+turbulent':
                B = JF12Field()
                B.randomStriated(seed)
                B.randomTurbulent(seed)

            for event_idx in triplet:
                # simulation setup
                sim = ModuleList()
                sim.add(PropagationCK(B, 1e-4, 0.1 * parsec, 100 * parsec))
                sim.add(SphericalBoundary(Vector3d(0), 20 * kpc))
                NUM_OF_SIMS = 1000
                output = MyTrajectoryOutput(f'trajectories_data/{mag_model_alias}/{particle_alias}/{mag_component_alias}/traj_PA+TA_{particle_alias}_{event_idx}_event_{NUM_OF_SIMS}sims_seed{seed}.txt')
                sim.add(output)

                event = events[event_idx]

                mean_energy = event[3] * EeV
                position = Vector3d(-8.122, 0, 0.0208) * kpc #Astropy in built params to match transformation

                #lon0,lat0 = eqToGal(event[1], event[2])        #RETURN WHEN NO TEST
                coords = SkyCoord(ra=event[1], dec=event[2], frame='icrs', unit='deg')
                #Here we have longtitudes [0, 2pi] and latitudes
                lon = coords.galactic.l
                #But for CRPROPA we need longitudes [-pi, pi] and colatitudes
                lon.wrap_angle = 180 * u.deg # longitude (phi) [-pi, pi] with 0 pointing in x-direction
                lon0 = lon.radian
                lat0 = coords.galactic.b.radian
                lat0 = np.pi/2 - lat0 #CrPropa uses colatitude, e.g. 90 - lat in degrees
                mean_dir = Vector3d()
                mean_dir.setRThetaPhi(1, lat0, lon0)

                for pid in particles:
                    for i in range(NUM_OF_SIMS):
                        if int(event[0]) < 72:
                            #TA EVENTS
                            #Harmonization according to https://lss.fnal.gov/archive/2025/conf/fermilab-conf-25-0486.pdf
                            mean_energy_harmonized = 10 * EeV * np.exp(alpha)*(mean_energy/(10 * EeV))**beta
                            energy = R.randNorm(mean_energy_harmonized, sigma_energy[1]*mean_energy_harmonized)
                            #direction = R.randVectorAroundMean(mean_dir, sigma_dir[1])
                            direction = R.randFisherVector(mean_dir, 2.278/(sigma_dir[1]**2))
                        else:
                            #AUGER EVENTS
                            energy = R.randNorm(mean_energy, sigma_energy[0]*mean_energy)
                            #direction = R.randVectorAroundMean(mean_dir, sigma_dir[0])
                            direction = R.randFisherVector(mean_dir, 2.278/(sigma_dir[0]**2))

                        candidate = Candidate(ParticleState(pid, energy, position, direction))
                        sim.run(candidate)
                output.close()
                del output
                gc.collect()
    '''
    for event in events:
        if int(event[0]) != 31: continue

        mean_energy = event[3] * EeV
        sigma_energy = (0.07, 0.15)
        position = Vector3d(-8.2, 0, 0) * kpc

        lon0,lat0 = eqToGal(event[1], event[2])        #RETURN WHEN NO TEST
        lat0 = math.pi/2 - lat0 #CrPropa uses colatitude, e.g. 90 - lat in degrees
        mean_dir = Vector3d()
        mean_dir.setRThetaPhi(1, lat0, lon0)
        sigma_dir = (0.002, 0.003) #1, 1.5 degree directional uncertainty

        for pid in particles:
            for i in range(NUM_OF_SIMS):
                if int(event[0]) < 28:
                    energy = R.randNorm(mean_energy, sigma_energy[1])
                    direction = R.randVectorAroundMean(mean_dir, sigma_dir[1])
                else:
                    energy = R.randNorm(mean_energy, sigma_energy[0])
                    direction = R.randVectorAroundMean(mean_dir, sigma_dir[0])

                candidate = Candidate(ParticleState(pid, energy, position, direction))
                sim.run(candidate)
    output.close()
    '''
