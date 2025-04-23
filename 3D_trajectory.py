from crpropa import *
from useful_funcs import eqToGal
import math
from tqdm import tqdm


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
    # magnetic field setup
    seed = 42
    R = Random(seed)
    B = JF12Field()
    B.randomStriated(seed)
    B.randomTurbulent(seed)

    # simulation setup
    #sim = ModuleList()
    #sim.add(PropagationCK(B, 1e-4, 0.1 * parsec, 100 * parsec))
    #sim.add(SphericalBoundary(Vector3d(0), 20 * kpc))
    #output = MyTrajectoryOutput(f'trajectories/trajectories_1e4/traj_PA+TA_H_{event_idx}_event.txt')
    #sim.add(output)
    #NUM_OF_SIMS = 10000

    '''
    Creating events list
    '''
    events = []
    with open('data/auger+TA_combined.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[6]), float(temp_event[7]), float(temp_event[8])))

    '''
    Sim for 4 particles for 1 event(third one)
    '''
    #particles = [- nucleusId(1,1), - nucleusId(4,2), - nucleusId(12,6), - nucleusId(52,26)]
    particles = [- nucleusId(12,6)]
    events_in_void = [16, 18, 19, 20, 22, 23, 24, 25, 30]
    triplet = [22, 23, 30]
    sigma_energy = (0.07, 0.15)
    sigma_dir = (0.002, 0.003) #1, 1.5 degree directional uncertainty

    for event_idx in tqdm(events_in_void):
        # simulation setup
        sim = ModuleList()
        sim.add(PropagationCK(B, 1e-4, 0.1 * parsec, 100 * parsec))
        sim.add(SphericalBoundary(Vector3d(0), 20 * kpc))
        NUM_OF_SIMS = 10000
        output = MyTrajectoryOutput(f'trajectories_data/C/traj_PA+TA_C_{event_idx}_event_{NUM_OF_SIMS}sims.txt')
        sim.add(output)

        event = events[event_idx]

        mean_energy = event[3] * EeV
        position = Vector3d(-8.2, 0, 0.0208) * kpc

        lon0,lat0 = eqToGal(event[1], event[2])        #RETURN WHEN NO TEST
        lat0 = math.pi/2 - lat0 #CrPropa uses colatitude, e.g. 90 - lat in degrees
        mean_dir = Vector3d()
        mean_dir.setRThetaPhi(1, lat0, lon0)

        for pid in particles:
            for i in tqdm(range(NUM_OF_SIMS)):
                if int(event[0]) < 28:
                    energy = R.randNorm(mean_energy, sigma_energy[1])
                    direction = R.randVectorAroundMean(mean_dir, sigma_dir[1])
                    #energy = mean_energy
                    #direction = mean_dir
                else:
                    energy = R.randNorm(mean_energy, sigma_energy[0])
                    direction = R.randVectorAroundMean(mean_dir, sigma_dir[0])
                    #energy = mean_energy
                    #direction = mean_dir

                candidate = Candidate(ParticleState(pid, energy, position, direction))
                sim.run(candidate)
        output.close()
        del output
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
