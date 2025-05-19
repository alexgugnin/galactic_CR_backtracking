from crpropa import *
from useful_funcs import eqToGal
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from side_checks.calc_metric_for_seed_check import makeCut, calcPerpPlane, calcEdge, r_mat, calculate_kde, calculate_hit, calculate_mahalanobis
import astropy.units as u
from astropy.coordinates import SkyCoord

import glob


def get_objects_list():
    d_list = {
        "sgr": 12.5, #2.9+-0.2, 8.1+-0.5 https://arxiv.org/pdf/2308.03484, 12.5, 3.8
        "grs": 8.6, #+2-1.6  https://arxiv.org/pdf/1409.2453
        "ss": 5.5, #+-0.2   https://www.aanda.org/articles/aa/full_html/2018/09/aa32488-17/aa32488-17.html
        "ngc": 7.58#       https://doi.org/10.1093/mnras/stab1475
    }
    object_coords = {"sgr": {"RA": 286.8097083, "DEC": 9.3222500, "dist": d_list["sgr"]}, #https://arxiv.org/pdf/2412.20050
                     "grs": {"RA": 288.798, "DEC": 10.946, "dist": d_list["grs"]},  #https://swift.gsfc.nasa.gov/results/transients/GRS1915p105/ 
                     "ss": {"RA": 287.956, "DEC": 4.99, "dist": d_list["ss"]},      #https://swift.gsfc.nasa.gov/results/transients/SS433/
                     "ngc": {"RA": 287.800, "DEC": 1.030, "dist": d_list["ngc"]}, #https://doi.org/10.1093/mnras/stab1475
                     }
    
    '''
    objects_list_old = {
        "sgr": [0, d_list['sgr']*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.2, d_list['sgr']*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), d_list['sgr']*np.sin(0.77*np.pi/180) + 0.0208],
        "grs": [0, d_list['grs']*np.cos(45.37*np.pi/180)*np.cos(-0.22*np.pi/180) - 8.2, d_list['grs']*np.sin(45.37*np.pi/180)*np.cos(-0.22*np.pi/180), d_list['grs']*np.sin(-0.22*np.pi/180) + 0.0208],
        "ss": [0, d_list['ss']*np.cos(39.69*np.pi/180)*np.cos(-2.24*np.pi/180) - 8.2, d_list['ss']*np.sin(39.69*np.pi/180)*np.cos(-2.24*np.pi/180), d_list['ss']*np.sin(-2.24*np.pi/180) + 0.0208],
        "ngc": [0, d_list['ngc']*np.cos(36.11*np.pi/180)*np.cos(-3.9*np.pi/180) - 8.2, d_list['ngc']*np.sin(36.11*np.pi/180)*np.cos(-3.9*np.pi/180), d_list['ngc']*np.sin(-3.9*np.pi/180) + 0.0208]
    }
    '''
    coords_sgr = SkyCoord(ra=object_coords["sgr"]["RA"]*u.deg, dec=object_coords["sgr"]["DEC"]*u.deg,
                      distance=object_coords["sgr"]["dist"]*u.kpc, frame='icrs')
    coords_grs = SkyCoord(ra=object_coords["grs"]["RA"]*u.deg, dec=object_coords["grs"]["DEC"]*u.deg,
                      distance=object_coords["grs"]["dist"]*u.kpc, frame='icrs')
    coords_ss = SkyCoord(ra=object_coords["ss"]["RA"]*u.deg, dec=object_coords["ss"]["DEC"]*u.deg,
                      distance=object_coords["ss"]["dist"]*u.kpc, frame='icrs')
    coords_ngc = SkyCoord(ra=object_coords["ngc"]["RA"]*u.deg, dec=object_coords["ngc"]["DEC"]*u.deg,
                      distance=object_coords["ngc"]["dist"]*u.kpc, frame='icrs')
    
    g_sgr = coords_sgr.transform_to('galactocentric') 
    g_grs = coords_grs.transform_to('galactocentric') 
    g_ss = coords_ss.transform_to('galactocentric') 
    g_ngc = coords_ngc.transform_to('galactocentric') 

    objects_list = {
        "sgr": [0, g_sgr.x.value, g_sgr.y.value, g_sgr.z.value],
        "grs": [0, g_grs.x.value, g_grs.y.value, g_grs.z.value],
        "ss": [0, g_ss.x.value, g_ss.y.value, g_ss.z.value],
        "ngc": [0, g_ngc.x.value, g_ngc.y.value, g_ngc.z.value]
    }

    return objects_list, d_list

def plot3D(data, objects) -> None:
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111, projection='3d')

    # plot trajectories
    I,X,Y,Z = data
    for i in np.unique(I):
        if i > 50: break
        ax.plot(X[I == i], Y[I == i], Z[I == i], lw=1, alpha=1, c='g')

    # plot Galactic border
    r = 20
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100))
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    ax.plot_surface(x, y, z, rstride=2, cstride=2, color='r', alpha=0.1, lw=0)
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='k', alpha=0.5, lw=0.1)

    # plot Galactic center
    ax.scatter(0,0,0, marker='o', color='r')
    # plot Earth
    ax.scatter(-8.2,0,0.0208, marker='P', color='b')

    #Plotting potential sources
    #Plot SGR 1900+14
    #for d in [2.9-0.2, 2.9, 2.9+0.2]:

    '''
    for d in [8.1-0.5, 8.1-0.3, 8.1-0.1, 8.1+0.1, 8.1+0.3, 8.1+0.5]:
        sgr_cords=[0, d*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.2, d*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), d*np.sin(0.77*np.pi/180)]
        ax.scatter(sgr_cords[1], sgr_cords[2], sgr_cords[3], marker='+', c='r', s=70) #43.02 0.77 12.5±1.7
    '''
    ax.scatter(objects['sgr'][1], objects['sgr'][2], objects['sgr'][3], marker='+', c='red', s=70)
    #plt.text(0.751-2*0.751 - 15*np.pi/180, 0.0135+7*np.pi/180, 'SGR 1900+14', fontsize=8, fontweight='bold')
    #Plot GRS 1915+105
    ax.scatter(objects['grs'][1], objects['grs'][2], objects['grs'][3], marker='+', c='green', s=70) #45.37 -0.22 8.6+2.0-1.6
    #Plot SS 433 Мікроквазар 39.69 -2.24 5.5±0.2
    ax.scatter(objects['ss'][1], objects['ss'][2], objects['ss'][3], marker='+', c='purple', s=70) #39.69 -2.24 5.5±0.2
    #Plot NGC 6760 Кулясте скупчення 36.11 -3.9 7.4±0.4
    ax.scatter(objects['ngc'][1], objects['ngc'][2], objects['ngc'][3], marker='+', c='magenta', s=70) #36.11 -3.9 7.4±0.4

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=1, label='Simulated CRs'),
                        Line2D([0], [0], marker='+', color='red', label='SGR 1900+14', markerfacecolor='red', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='green', label='GRS 1915+105', markerfacecolor='green', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='purple', label='SS 433', markerfacecolor='purple', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='magenta', label='NGC 6760', markerfacecolor='magenta', linestyle='', markersize=8)
                        ]
    fig.legend(handles=legend_elements, loc='upper right')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlabel('x / kpc', fontsize=18)
    ax.set_ylabel('y / kpc', fontsize=18)
    ax.set_zlabel('z / kpc', fontsize=18)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    ax.set_zlim((-20, 20))
    ax.xaxis.set_ticks((-20,-10,0,10,20))
    ax.yaxis.set_ticks((-20,-10,0,10,20))
    ax.zaxis.set_ticks((-20,-10,0,10,20))
    plt.show()

def plot3D_from_pandas(data, data_cut, objects, target_transformed = None, norms=None, save_file = None):
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111, projection='3d')

    # plot trajectories
    I,X,Y,Z = data
    for i in np.unique(I):
        if i > 50: break
        ax.plot(X[I == i], Y[I == i], Z[I == i], lw=1, alpha=0.2, c='g')
    #Plot cut surface
    ax.scatter(data_cut['X'], data_cut['Y'], data_cut['Z'], lw=1, alpha=1, c='r', s=10)

    # plot Galactic border
    r = 20
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100))
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    ax.plot_surface(x, y, z, rstride=2, cstride=2, color='r', alpha=0.1, lw=0)
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='k', alpha=0.5, lw=0.1)

    # plot Galactic center
    ax.scatter(0,0,0, marker='o', color='yellow')
    # plot Earth
    ax.scatter(-8.2,0,0.0208, marker='P', color='b')
    
    if norms:
        earth = [-8.2,0,0.0208]
        #Plot Normale
        ax.quiver(
                earth[0], earth[1], earth[2], 
                norms[0][0], norms[0][1], norms[0][2], color='r', label='Base norm'
            )
        ax.quiver(
                earth[0], earth[1], earth[2], 
                norms[1][0], norms[1][1], norms[1][2], color='g', label='Base norm'
            )
        ax.quiver(
                earth[0], earth[1], earth[2], 
                norms[2][0], norms[2][1], norms[2][2], color='b', label='Base norm'
            )

    #Plotting potential sources

    #plt.text(0.751-2*0.751 - 15*np.pi/180, 0.0135+7*np.pi/180, 'SGR 1900+14', fontsize=8, fontweight='bold')
    #Plot SGR 1900+14
    ax.scatter(objects['sgr'][1], objects['sgr'][2], objects['sgr'][3], marker='+', c='r', s=70) #OLD CORDS 43.02 0.77 12.5±1.7
    #Plot GRS 1915+105
    ax.scatter(objects['grs'][1], objects['grs'][2], objects['grs'][3], marker='+', c='green', s=70) #45.37 -0.22 8.6+2.0-1.6
    #Plot SS 433 Мікроквазар 39.69 -2.24 5.5±0.2
    ax.scatter(objects['ss'][1], objects['ss'][2], objects['ss'][3], marker='+', c='purple', s=70) #39.69 -2.24 5.5±0.2
    #Plot NGC 6760 Кулясте скупчення 36.11 -3.9 7.4±0.4
    ax.scatter(objects['ngc'][1], objects['ngc'][2], objects['ngc'][3], marker='+', c='magenta', s=70) #36.11 -3.9 7.4±0.4
    if target_transformed is not None:
        ax.scatter(target_transformed[0], target_transformed[1], target_transformed[2], marker='+', c='black', s=80) #36.11 -3.9 7.4±0.4

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=1, label='Simulated CRs'),
                        Line2D([0], [0], marker='+', color='red', label='SGR 1900+14', markerfacecolor='red', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='green', label='GRS 1915+105', markerfacecolor='green', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='purple', label='SS 433', markerfacecolor='purple', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='magenta', label='NGC 6760', markerfacecolor='magenta', linestyle='', markersize=8)
                        ]
    fig.legend(handles=legend_elements, loc='upper right')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlabel('x / kpc', fontsize=18)
    ax.set_ylabel('y / kpc', fontsize=18)
    ax.set_zlabel('z / kpc', fontsize=18)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    ax.set_zlim((-20, 20))
    ax.xaxis.set_ticks((-20,-10,0,10,20))
    ax.yaxis.set_ticks((-20,-10,0,10,20))
    ax.zaxis.set_ticks((-20,-10,0,10,20))
    if save_file is not None: plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot2D_projection(x, z, target, radius, xyz_kde, save_name=None) -> None:
    import seaborn as sns
    '''Func for plotting XZ projection with target object and 1 degree circle around it'''

    theta = np.linspace(0, 2 * np.pi, 100)  # Angles from 0 to 2*pi
    x_circle = target[0] + radius * np.cos(theta)  # X coordinates of the circle
    z_circle = target[2] + radius * np.sin(theta)  # Y coordinates of the circle

    # Plot the circle
    plt.figure(figsize=(12, 8))
    plt.plot(x_circle, z_circle, label=f'Circle (r={radius})', c='r')

    '''Simple 2D scatterplot'''
    
    plt.scatter(x, z, s = 5, label="Trajectories crossing the object's plane")
    plt.scatter(target[0], target[2], c='r', label='Target object')
    plt.text(target[0]-0.15, target[2]+0.06, 'SGR 1900+14', fontsize=10, fontweight='bold', c = 'red')
    plt.xlabel("Cartesian coordinate X, kPc")
    plt.ylabel("Cartesian coordinate Z, kPc")
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()
    

    '''KDE colormesh with levels'''
    '''
    confidence_levels = [0.6827, 0.9545, 0.9973]
    Z_flat = xyz_kde[2].flatten()
    Z_sorted = np.sort(Z_flat)[::-1]
    cumsum = np.cumsum(Z_sorted)
    cumsum /= cumsum[-1]
    levels = sorted([Z_sorted[np.searchsorted(cumsum, cl)] for cl in confidence_levels])

    plt.contour(xyz_kde[0], xyz_kde[1], xyz_kde[2], levels=levels, colors=['blue', 'green', 'red'])
    plt.pcolormesh(xyz_kde[0], xyz_kde[1], xyz_kde[2], cmap='plasma')
    plt.scatter(target[0], target[2], marker='+', c='red', s=80, label = 'SGR 1900+14')
    plt.text(target[0]-0.15, target[2]+0.06, 'SGR 1900+14', fontsize=10, fontweight='bold', c = 'red')
    #plt.legend()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    '''
    '''Histplot'''
    '''
    plt.hist2d(x, z, bins=30, cmap='binary')
    cb = plt.colorbar()
    cb.set_label('counts in bin')
    plt.scatter(target[0], target[2], marker='+', c='red', s=80, label = "Target object")
    plt.xlabel('X, kpc')
    plt.ylabel('Z, kpc')
    #plt.text(8.2, 0.3, 'SGR 1900+14', fontsize=10, fontweight='bold', c = 'red')
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()
    '''

    '''Joint histplot'''
    '''
    nbins = int(np.ceil(np.log2(len(x))) + 1)
    g0 = sns.jointplot(x=np.array(x), y=np.array(z), kind="hist", height=8, ratio=6,
                       marginal_ticks=True,
                       marginal_kws=dict(bins=nbins, fill=True))
    #g0.plot_joint(sns.kdeplot, color="grey", zorder=0, levels = [0.003, 0.05, 0.32])#levels=[0.68, 0.95, 0.997])
    g0.ax_joint.scatter(target[0], target[2], marker='+', c='red', s=80)
    sns.kdeplot(x = x, y = z, color="grey", zorder=0, levels = [0.003, 0.05, 0.32], ax=g0.ax_joint)
    g0.ax_joint.text(target[0], target[2], 'SGR 1900+14', fontsize=9, fontweight='bold', c = 'red')

    g0.set_axis_labels('X, kpc', 'Z, kpc', fontsize=10)
    g0.figure.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    '''

if __name__ == '__main__':
    particle = 'H'
    event_num = 23
    object_name = 'sgr'
    data = np.genfromtxt(f'trajectories_data/{particle}/traj_PA+TA_{particle}_{event_num}_event_1000sims_CORRECTED.txt', unpack=True, skip_footer=1)
    objects_list, d_list = get_objects_list()
    
    obj_cords = objects_list[object_name]
    '''
    2D SURFACE APPROACH
    '''
    data_cut, obj_cords_transformed, norms = makeCut(data, obj_cords, rot=True)
    plot3D(data, objects_list)
    #print(data_cut['Y'])
    #print(norms)
    #data_cut_unrot, _ = makeCut(data, obj_cords, rot=False)
    #plot3D(data, objects_list)
    #plot3D_from_pandas(data, data_cut_unrot, objects_list)
    #plot3D_from_pandas(data, data_cut, objects_list, target_transformed=obj_cords_transformed, norms=norms)#, save_file=f'paper_results/trajectories/event_{event_num}_{object_name}_{particle}_3Dmap.jpeg')
    xyz_colormesh_kde, score = 0,0#calculate_kde(data_cut, obj_cords_transformed)
    count, hit = calculate_hit(data_cut, obj_cords_transformed, np.pi*d_list[object_name]/180)
    #mah_dist = calculate_mahalanobis(data_cut, obj_cords_transformed)
    print(f"\n Num of trajectories: {count}, Hit is :{hit}, KDE score is : {score}")
    plot2D_projection(data_cut['X'], data_cut['Z'], obj_cords_transformed, np.pi*d_list[object_name]/180, xyz_colormesh_kde,)
                      #f'paper_results/projections/eng_pres_scatter.jpeg')

    '''3D VOLUME APPROACH'''
    '''
    I,X,Y,Z = data
    volume_radii = 0.5#kpc
    impact_radii = np.pi*d_list[object_name]/180
    print(impact_radii)
    sphere_cords = pd.DataFrame({'x': np.empty(1), 
                          'y': np.empty(1), 
                          'z': np.empty(1)})
    for i in tqdm(np.unique(I)):
        temp_data = pd.DataFrame({'x': np.array(X[I == i]), 
                                  'y': np.array(Y[I == i]), 
                                  'z': np.array(Z[I == i])})
        temp_data = temp_data[(temp_data['x'].between(obj_cords[1]-volume_radii, obj_cords[1]+volume_radii, inclusive=True)) & 
                              (temp_data['y'].between(obj_cords[2]-volume_radii, obj_cords[2]+volume_radii, inclusive=True)) & 
                              (temp_data['z'].between(obj_cords[3]-volume_radii, obj_cords[3]+volume_radii, inclusive=True))]
        sphere_cords = pd.concat([sphere_cords, temp_data]).reset_index().drop(['index'], axis=1)
    
    #sphere_cords = sphere_cords.drop([0]).reset_index() RAZOBRATSIA
    obj_volume = sphere_cords[(sphere_cords['x'].between(obj_cords[1]-impact_radii, obj_cords[1]+impact_radii, inclusive=True)) & 
                           (sphere_cords['y'].between(obj_cords[2]-impact_radii, obj_cords[2]+impact_radii, inclusive=True)) & 
                           (sphere_cords['z'].between(obj_cords[3]-impact_radii, obj_cords[3]+impact_radii, inclusive=True))]
    print(obj_volume.shape, sphere_cords.shape)
    print(f"SCORE IS {obj_volume.shape[1]/sphere_cords.shape[1]}")
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111, projection='3d')
    ax.scatter(sphere_cords['x'], sphere_cords['y'], sphere_cords['z'], s=30)
    ax.scatter(obj_cords[1], obj_cords[2], obj_cords[3], marker='+', c='r', s=70)
    plt.show()
    '''
    