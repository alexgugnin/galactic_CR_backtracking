import numpy as np
from crpropa import kpc
import pandas as pd
import time
from tqdm import tqdm

def calcPerpPlane(target_cords, earth_cords = [0, -8.2, 0, 0.0208]):
    '''Calculates the A, B, C, D for the plane'''
    norm = np.array([target_cords[1] - earth_cords[1], target_cords[2] - earth_cords[2], target_cords[3] - earth_cords[3]])
    D_plane = -(norm[0]*target_cords[1] + norm[1]*target_cords[2] + norm[2]*target_cords[3])
    #xx, yy = np.meshgrid(np.linspace(-20,20,160), np.linspace(-20,20,160))
    #z = (-norm[0] * xx - norm[1] * yy + D_plane)/norm[2]

    return norm, D_plane


def calcEdge(x, y, z, norm, D_plane):
    ''' Finds the Last point which crosses observational surface'''
    old_distance = 1e5
    for idx in range(len(x)):
        new_distance = abs(norm[0]*x[idx] + norm[1]*y[idx] + norm[2]*z[idx] + D_plane)/np.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
        if ((new_distance > old_distance) and (new_distance <= 0.1)):
            x = x[:idx]
            y = y[:idx]
            z = z[:idx]
            return x, y, z
        else:
            old_distance = new_distance


def makeCut(data, target_cords):
    '''Returns the point of intersection of the trajectory with the object surface.
    Makes ortogonal transformation to match yOz plane. Need to be merged with calcEdge?
    Now returns also a transformed object cords as [y, z]'''
    
    print("---STARTING TO MAKE A CUT---")
    start_time = time.time()
    I,X,Y,Z,V = data
    x_nonrot, y_nonrot, z_nonrot = [], [], []
    for i in tqdm(np.unique(I)):
        norm, D_plane = calcPerpPlane(target_cords)
        try:
            _x, _y, _z = calcEdge(X[I == i], Y[I == i], Z[I == i], norm, D_plane)
        except:
            continue
        x_nonrot.append(_x)
        y_nonrot.append(_y)
        z_nonrot.append(_z)

    print(f"Cut DONE in {time.time() - start_time}")
    
    return pd.DataFrame({'X':np.array(x_nonrot), 'Y':np.array(y_nonrot), 'Z':np.array(z_nonrot)})


if __name__ == "__main__":
    data = np.genfromtxt(f'traj_PA+TA_C_30_event_100sims.txt', unpack=True, skip_footer=1)
    object_name = 'sgr'
    d_list = {
        "sgr": 12.5, #2.9+-0.2, 8.1+-0.5 https://arxiv.org/pdf/2308.03484, 12.5
        "grs": 8.6, #+2-1.6
        "ss": 5.5, #+-0.2
        "ngc": 7.4
    }
    objects_list = {
        "sgr": [0, d_list['sgr']*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.2, d_list['sgr']*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), d_list['sgr']*np.sin(0.77*np.pi/180) + 0.0208],
        "grs": [0, d_list['grs']*np.cos(45.37*np.pi/180)*np.cos(-0.22*np.pi/180) - 8.2, d_list['grs']*np.sin(45.37*np.pi/180)*np.cos(-0.22*np.pi/180), d_list['grs']*np.sin(-0.22*np.pi/180) + 0.0208],
        "ss": [0, d_list['ss']*np.cos(39.69*np.pi/180)*np.cos(-2.24*np.pi/180) - 8.2, d_list['ss']*np.sin(39.69*np.pi/180)*np.cos(-2.24*np.pi/180), d_list['ss']*np.sin(-2.24*np.pi/180) + 0.0208],
        "ngc": [0, d_list['ngc']*np.cos(36.11*np.pi/180)*np.cos(-3.9*np.pi/180) - 8.2, d_list['ngc']*np.sin(36.11*np.pi/180)*np.cos(-3.9*np.pi/180), d_list['ngc']*np.sin(-3.9*np.pi/180) + 0.0208]
    }
    earth_cords = [0, -8.2, 0, 0.0208]

    obj_cords = objects_list[object_name]

    I, X, Y, Z, velocities = data #All VELOCITIES ARE THE SAME!!!
    velocity = velocities[0]
    data_cut = makeCut(data, obj_cords)

    for i in range(len(data_cut)):
        X, Y, Z = data_cut.loc[i]['X'], data_cut.loc[i]['Y'], data_cut.loc[i]['Z']
        displacements = []
        for j in range(len(X)):
            if j == 0: 
                displacements.append(0)
                continue
            displacement = np.sqrt((X[j] - X[j-1])**2 + (Y[j] - Y[j-1])**2 + (Z[j] - Z[j-1])**2)
            displacements.append(displacement)
        
        displacements = np.array(displacements) * kpc # To be in m/s
        times = displacements / velocity
        print(f"C time: {np.sum(times)/60/60/24/365}, y")
        photon_direct = (np.sqrt(np.sum((np.array(obj_cords[1:]) - np.array(earth_cords[1:]))**2)) * kpc / velocity)/60/60/24/365
        print(f"Direct way: {photon_direct}, y")
        print(f"Difference: {np.sum(times)/60/60/24/365 - photon_direct}, y")

        break    

    exit()


    for i in range(len(data_cut)):
        if i == 0: 
            displacements.append(0)
            continue
        displacement = np.sqrt((X[i] - X[i-1])**2 + (Y[i] - Y[i-1])**2 + (Z[i] - Z[i-1])**2)
        displacements.append(displacement)
    
    displacements = np.array(displacements) * kpc # To be in m/s
    times = displacements / velocities
    print(np.sum(times))