from useful_funcs import eqToGal

def visualizeTrajectory(id, lat0, lon0, lats_tot, lons_tot) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    lat0 = np.pi/2 - lat0
    plt.figure(figsize=(12,7))
    plt.subplot(111, projection = 'hammer')
    plt.title(f"{id} EVENT")
    plt.scatter(lon0, lat0, marker='+', c='black', s=100)#initial state
    H_patch = mpatches.Patch(color='blue', label='H')
    He_patch = mpatches.Patch(color='orange', label='He')
    N_patch = mpatches.Patch(color='green', label='N')
    plt.legend(handles=[H_patch, He_patch, N_patch])
    if id == '3':
        plt.scatter(0.751, 0.0135, marker='+', c='red', s=100)#AIM SGR 1900+14


    for i in range(3):
        lats = lats_tot[i][0]
        lons = lons_tot[i][0]
        lats = np.pi/2 - np.array(lats)
        plt.scatter(lons, lats, marker='o', linewidths=0, alpha=0.2)
        plt.grid(True)

    plt.show()

def visualizeMap(mag_data, lats_tot, lons_tot) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    plt.figure(figsize=(12,7))
    plt.subplot(111, projection = 'hammer')
    plt.title(f"Magnetars and CR distribution")

    mag_lons, mag_lats = [], []
    for magn in mag_data:
        mag_lons.append(magn[0]*np.pi/180)
        mag_lats.append(magn[1]*np.pi/180)
    plt.scatter(mag_lons, mag_lats, marker='+')
    lats_tot = np.pi/2 - np.array(lats_tot)
    plt.scatter(lons_tot, lats_tot, marker='o', linewidths=0, alpha=0.2)
    plt.scatter(0.751, 0.0135, marker='+', c='red', s=100)
    plt.grid(True)
    plt.savefig('temp_map.png')
    #plt.show()

def setupSimulation():
    '''
    This function setups current simulation
    '''
    # magnetic field setup
    B = JF12Field()
    #seed = 691342
    #B.randomStriated(seed)
    #B.randomTurbulent(seed)

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

    events = []
    with open('data/AugerApJS2022_highE.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            events.append((temp_event[0], float(temp_event[6]), float(temp_event[7]), float(temp_event[8])))

    magnetars = []
    with open('source_maps/magnetars.csv', 'r') as infile:
        for line in infile:
            temp_source = line.split(',')
            if temp_source[0] == 'Name': continue
            magnetars.append((float(temp_source[7]), float(temp_source[8])))

    sim, obs = setupSimulation()

    R = Random()  # CRPropa random number generator
    H_id = '-1000010010'
    He_id = '-1000020040'
    N_id = '-1000070140'

    lons_tot, lats_tot = [], []
    particles = [- nucleusId(1,1), - nucleusId(4,2), - nucleusId(14,7)]
    for event in events:
        meanEnergy = event[3] * EeV
        sigmaEnergy = 0.1 * meanEnergy  # 10% energy uncertainty
        position = Vector3d(-8.5, 0, 0) * kpc
        lon0,lat0 = eqToGal(event[1], event[2])
        lat0 = math.pi/2 - lat0 #CrPropa uses colatitude, e.g. 90 - lat in degrees
        meanDir = Vector3d()
        meanDir.setRThetaPhi(1, lat0, lon0)
        sigmaDir = 0.002  # 1 degree directional uncertainty
        for pid in particles:
            lons, lats = [], []
            for i in range(10):
                energy = R.randNorm(meanEnergy, sigmaEnergy)
                direction = R.randVectorAroundMean(meanDir, sigmaDir)

                c = Candidate(ParticleState(pid, energy, position, direction))
                sim.run(c)

                d1 = c.current.getDirection()
                lons.append(d1.getPhi())
                lats.append(d1.getTheta())
            if str(pid) == He_id:
                for elem in lons:
                    lons_tot.append(elem)
                for elem in lats:
                    lats_tot.append(elem)
            elif str(pid) == H_id:
                pass
                #lons_tot.append((lons, 'He'))
                #lats_tot.append((lats, 'He'))
            else:
                pass
                #lons_tot.append((lons, 'N'))
                #lats_tot.append((lats, 'N'))
    visualizeMap(magnetars, lats_tot, lons_tot)
    #visualizeTrajectory(event[0], lat0, lon0, lats_tot, lons_tot)
