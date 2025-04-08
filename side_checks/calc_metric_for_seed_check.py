import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import glob
import time

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

def p_mat(angle):
    '''Rotate over X'''
    p_matrix = np.array([
        [1,           0,                 0],
        [0, np.cos(angle).item(),   -np.sin(angle).item()],
        [0, np.sin(angle).item(),  np.cos(angle).item()]
    ])
    return p_matrix

def q_mat(angle):
    '''Rotate over Y'''
    q_matrix = np.array([
        [np.cos(angle).item(), 0, np.sin(angle).item()],
        [0,             1,              0],
        [-np.sin(angle).item(), 0,  np.cos(angle).item()]
    ])
    return q_matrix

def r_mat(angle):
    '''Rotate over Z'''
    r_matrix = np.array([
        [np.cos(angle).item(), -np.sin(angle).item(), 0],
        [np.sin(angle).item(), np.cos(angle).item(),   0],
        [0,               0,           1]
    ])
    return r_matrix


def rotate_yz(x, y, z, norm):
    '''Ortogonal transformation of a plane'''
    abs_norm = np.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    cosa, cosb, cosg = np.arccos(norm[0]/abs_norm), np.arccos(norm[1]/abs_norm), np.arccos(norm[2]/abs_norm) 

    data_before_rot = np.vstack((x,y,z))
    
    data_Z_rot = np.matmul(r_mat(cosb), data_before_rot) #rotating over Z
    data_after_rot = np.matmul(p_mat(np.absolute(np.pi/2 - cosg)), data_Z_rot) #rotating over X

    return data_after_rot[:1].ravel(), data_after_rot[1:2].ravel(), data_after_rot[2:].ravel()

def makeCut(data, target_cords, rot=True):
    '''Returns the point of intersection of the trajectory with the object surface.
    Makes ortogonal transformation to match yOz plane. Need to be merged with calcEdge?
    Now returns also a transformed object cords as [y, z]'''
    
    print("---STARTING TO MAKE A CUT---")
    start_time = time.time()
    I,X,Y,Z = data
    x_nonrot, y_nonrot, z_nonrot = [], [], []
    for i in tqdm(np.unique(I)):
        norm, D_plane = calcPerpPlane(target_cords)
        try:
            _x, _y, _z = calcEdge(X[I == i], Y[I == i], Z[I == i], norm, D_plane)
        except:
            continue
        x_nonrot.append(_x[-1])
        y_nonrot.append(_y[-1])
        z_nonrot.append(_z[-1])

    print(f"Cut DONE in {time.time() - start_time}")
    if rot:
        norms = []
        norm = norm.reshape(-1,1)
        norms.append(norm)
        print("---PERFORMING ROTATION---")
        x_rot, y_rot, z_rot = rotate_yz(x_nonrot, y_nonrot, z_nonrot, norm)
        obj_trans = np.array(target_cords[1:]).reshape(-1,1) 
        abs_norm = np.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
        alpha = np.arccos(norm[0]/abs_norm)
        beta = np.arccos(norm[1]/abs_norm)
        gamma = np.arccos(norm[2]/abs_norm) 
        
        obj_Z_rot = np.matmul(r_mat(beta), obj_trans) #rotating over Z
        norm_Z_rot = np.matmul(r_mat(beta), norm)
        norms.append(np.matmul(r_mat(beta), norm))

        obj_rot = np.matmul(p_mat(np.absolute(np.pi/2 - gamma)), obj_Z_rot) #rotating over X
        norm_X_rot = np.matmul(p_mat(np.absolute(np.pi/2 - gamma)), norm_Z_rot)
        norms.append(norm_X_rot)

        #obj_cords_transformed = np.array([obj_rot[1], obj_rot[2], obj_rot[0]]) #Y, Z, X
        obj_cords_transformed = np.array([obj_rot[0], obj_rot[1], obj_rot[2]]) # X, Y, Z

        return pd.DataFrame({'X':np.array(x_rot), 'Y':np.array(y_rot), 'Z':np.array(z_rot)}), obj_cords_transformed, norms
    else:
        return pd.DataFrame({'X':np.array(x_nonrot), 'Y':np.array(y_nonrot), 'Z':np.array(z_nonrot)}), target_cords[:2]

def calculate_kde(data, object_cords) -> float:
    '''Calculates pdf using kde with bandwith from the gridsearch
    and returns the denstiy value of needed object for this pdf divided by
    the max density value for this pdf
    https://gist.github.com/daleroberts/7a13afed55f3e2388865b0ec94cd80d2
    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/'''
    xz = np.vstack([data['X'], data['Z']])
    d = xz.shape[0]
    n = xz.shape[1]

    #Creating grid for search and finding best estimator in terms of bandwidth
    print('---STARTING KDE GRIDSEARCH---')
    start_time = time.time()
    grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.01, 1.0, 100)},
                    cv=20) # 20-fold cross-validation with 100 bandwidths
    grid.fit(xz.T)
    kde = grid.best_estimator_
    print(f"Gridsearch FINISHED in {time.time() - start_time} s.")

    point = np.array([object_cords[0], object_cords[2]])

    xmin = data['X'].min()
    xmax = data['X'].max()
    ymin = data['Z'].min()
    ymax = data['Z'].max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(np.exp(kde.score_samples(positions.T)), X.shape)

    return (np.exp(kde.score_samples(point.T))/Z.max())[0] #score_samples returns the log density, so exp is needed. Also prob density can be more than 1

def calculate_hit(data, circle_center, r) -> float:
    '''
    This function calculates the number of events hitting the area of r degree around
    the potential source divided by the number of all simulated events.
    Equasion as simple as : (x - cc_x)^2 + (y - cc_y)^2 = r^2
    Returns simple float32 value.
    '''

    print("---CALCULATING HIT---")
    # Calculate squared distances from the circle center
    data['distance_squared'] = (data['X'] - circle_center[0]) ** 2 + (data['Z'] - circle_center[2]) ** 2

    # Determine how many points are inside the circle
    inside_circle_count = data[data['distance_squared'] <= r**2].shape[0]

    return inside_circle_count, inside_circle_count/len(data['X'])

def calculate_mahalanobis(data, object_cords):
    '''Calculates Mahalanobis distance from a given object to the data distribution'''
    from scipy.spatial import distance

    data = np.array([data['X'], data['Z']])
    # Step 1: Compute the mean of the data points
    mean = np.mean(data, axis=0)

    # Step 2: Compute the covariance matrix of the data
    cov_matrix = np.cov(data, rowvar=False)
    print(cov_matrix)
    # Step 3: Inverse of the covariance matrix
    cov_inv = np.linalg.inv(cov_matrix)

    # Step 4: Compute Mahalanobis distance for a point (e.g., the first point)
    point = object_cords
    print(point)
    mah_dist = distance.mahalanobis(point, mean, cov_inv)

def calculate_hotellings(data, object_cords):
    pass


if __name__ == '__main__':
    path = 'trajectories_1000_rand_seeds/C/'
    file_list = glob.glob(f'{path}*.txt')
    d_list = {
        "sgr": 8.1,
        "grs": 8.6,
        "ss": 5.5,
        "ngc_cords": 7.4
    }
    objects_list = {
        "sgr": [0, 8.1*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.5, 8.1*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), 8.1*np.sin(0.77*np.pi/180)],
        "grs": [0, 8.6*np.cos(45.37*np.pi/180)*np.cos(-0.22*np.pi/180) - 8.5, 8.6*np.sin(45.37*np.pi/180)*np.cos(-0.22*np.pi/180), 8.6*np.sin(-0.22*np.pi/180)],
        "ss": [0, 5.5*np.cos(39.69*np.pi/180)*np.cos(-2.24*np.pi/180) - 8.5, 5.5*np.sin(39.69*np.pi/180)*np.cos(-2.24*np.pi/180), 5.5*np.sin(-2.24*np.pi/180)],
        "ngc_cords": [0, 7.4*np.cos(36.11*np.pi/180)*np.cos(-3.9*np.pi/180) - 8.5, 7.4*np.sin(36.11*np.pi/180)*np.cos(-3.9*np.pi/180), 7.4*np.sin(-3.9*np.pi/180)]
    }
    #triplet = [22, 23, 30]

    result = pd.DataFrame(columns=['Event', 'Object', 'Seed', 'Hit'])
    for file in tqdm(file_list):
        filename = file.split('/')[2].split('_')
        event, seed = filename[3], filename[5]
        data = np.genfromtxt(file, unpack=True, skip_footer=1)

        #Creating cut for every potential source and finding score for them
        for obj_name, obj_cords in objects_list.items():
            cut_data, obj_cords_tf, _ = makeCut(data, obj_cords)
            hit = calculate_hit(cut_data, obj_cords_tf, np.pi*d_list[obj_name]/180)
            #score = calculate_kde(cut_data, obj_cords_tf)
            temp_dict = {'Event': [event],
                    'Object': [obj_name],
                    'Seed': [seed],
                    'Hit': [hit]
                    #'Score': [score]
                   }
            temp_df = pd.DataFrame(temp_dict)
            result = pd.concat([result, temp_df], ignore_index = True)
            result.reset_index()
    result.to_csv(f"{path}results_for_{path.split('/')[1]}_rotated_hitMetric.csv")
