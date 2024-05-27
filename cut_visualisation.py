from crpropa import *
from useful_funcs import eqToGal
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from side_checks.calc_metric_for_seed_check import makeCut, calcPerpPlane, calcEdge, r_mat, calculate_kde

import glob

def plot3D(data) -> None:
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
    ax.scatter(-8.5,0,0, marker='P', color='b')

    #Plotting potential sources
    #Plot SGR 1900+14
    ax.scatter(12.5*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.5, 12.5*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), 12.5*np.sin(0.77*np.pi/180), marker='+', c='red', s=70) #43.02 0.77 12.5±1.7
    #plt.text(0.751-2*0.751 - 15*np.pi/180, 0.0135+7*np.pi/180, 'SGR 1900+14', fontsize=8, fontweight='bold')
    #Plot GRS 1915+105
    ax.scatter(8.6*np.cos(45.37*np.pi/180)*np.cos(-0.22*np.pi/180) - 8.5, 8.6*np.sin(45.37*np.pi/180)*np.cos(-0.22*np.pi/180), 8.6*np.sin(-0.22*np.pi/180), marker='+', c='green', s=70) #45.37 -0.22 8.6+2.0-1.6
    #Plot SS 433 Мікроквазар 39.69 -2.24 5.5±0.2
    ax.scatter(5.5*np.cos(39.69*np.pi/180)*np.cos(-2.24*np.pi/180) - 8.5, 5.5*np.sin(39.69*np.pi/180)*np.cos(-2.24*np.pi/180), 5.5*np.sin(-2.24*np.pi/180), marker='+', c='purple', s=70) #39.69 -2.24 5.5±0.2
    #Plot NGC 6760 Кулясте скупчення 36.11 -3.9 7.4±0.4
    ax.scatter(7.4*np.cos(36.11*np.pi/180)*np.cos(-3.9*np.pi/180) - 8.5, 7.4*np.sin(36.11*np.pi/180)*np.cos(-3.9*np.pi/180), 7.4*np.sin(-3.9*np.pi/180), marker='+', c='magenta', s=70) #36.11 -3.9 7.4±0.4

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

def plot3D_from_pandas(data, data_cut):
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
    ax.scatter(0,0,0, marker='o', color='r')
    # plot Earth
    ax.scatter(-8.5,0,0, marker='P', color='b')

    #Plotting potential sources
    #Plot SGR 1900+14
    sgr_cords = [0, 12.5*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.5, 12.5*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), 12.5*np.sin(0.77*np.pi/180)]
    ax.scatter(12.5*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.5, 12.5*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), 12.5*np.sin(0.77*np.pi/180), marker='+', c='b', s=70) #43.02 0.77 12.5±1.7
    #Transformed SGR 1900+14
    norm, plane = calcPerpPlane(sgr_cords)
    sgr_trans = np.array(sgr_cords[1:]).reshape(-1,1)
    abs_norm = np.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    alpha = np.arccos(norm[0]/abs_norm)
    sgr_rot = np.matmul(r_mat(-alpha), sgr_trans)
    ax.scatter(sgr_rot[0], sgr_rot[1], sgr_rot[2], marker='+', c='b', s=70)
    #plt.text(0.751-2*0.751 - 15*np.pi/180, 0.0135+7*np.pi/180, 'SGR 1900+14', fontsize=8, fontweight='bold')
    #Plot GRS 1915+105
    ax.scatter(8.6*np.cos(45.37*np.pi/180)*np.cos(-0.22*np.pi/180) - 8.5, 8.6*np.sin(45.37*np.pi/180)*np.cos(-0.22*np.pi/180), 8.6*np.sin(-0.22*np.pi/180), marker='+', c='green', s=70) #45.37 -0.22 8.6+2.0-1.6
    #Plot SS 433 Мікроквазар 39.69 -2.24 5.5±0.2
    ax.scatter(5.5*np.cos(39.69*np.pi/180)*np.cos(-2.24*np.pi/180) - 8.5, 5.5*np.sin(39.69*np.pi/180)*np.cos(-2.24*np.pi/180), 5.5*np.sin(-2.24*np.pi/180), marker='+', c='purple', s=70) #39.69 -2.24 5.5±0.2
    #Plot NGC 6760 Кулясте скупчення 36.11 -3.9 7.4±0.4
    ax.scatter(7.4*np.cos(36.11*np.pi/180)*np.cos(-3.9*np.pi/180) - 8.5, 7.4*np.sin(36.11*np.pi/180)*np.cos(-3.9*np.pi/180), 7.4*np.sin(-3.9*np.pi/180), marker='+', c='magenta', s=70) #36.11 -3.9 7.4±0.4

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

if __name__ == '__main__':
    data = np.genfromtxt('traj_PA+TA_Fe_23_event_10737418_seed.txt', unpack=True, skip_footer=1)
    sgr_cords = [0, 12.5*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.5, 12.5*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), 12.5*np.sin(0.77*np.pi/180)]
    data_cut = makeCut(data, sgr_cords)
    plot3D_from_pandas(data, data_cut)
    score = calculate_kde(data_cut, sgr_cords)
    plt.scatter(data_cut['Y'], data_cut['Z'], s = 5)
    norm, plane = calcPerpPlane(sgr_cords)
    sgr_trans = np.array(sgr_cords[1:]).reshape(-1,1)
    abs_norm = np.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    alpha = np.arccos(norm[0]/abs_norm)
    sgr_rot = np.matmul(r_mat(-alpha), sgr_trans)
    print(score)
    plt.scatter(sgr_rot[1], sgr_rot[2])
    plt.show()


    #print(data_cut)
