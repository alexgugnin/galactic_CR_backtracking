from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt

def plot2DProjection(data, axis) -> None:
    plt.figure(figsize=(12,12))

    I,X,Y,Z = data
    for i in np.unique(I):
        plt.plot([I,X,Y,Z][axis[0]][I == i], [I,X,Y,Z][axis[1]][I == i], lw=1, alpha=1)

    # plot Galactic border
    #r = 20
    #u, v = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100))
    #x = r * np.cos(u) * np.sin(v)
    #y = r * np.sin(u) * np.sin(v)
    #z = r * np.cos(v)
    #plt.plot_surface(x, y, rstride=2, cstride=2, color='r', alpha=0.1, lw=0)
    #plt.plot_wireframe(x, y, rstride=10, cstride=10, color='k', alpha=0.5, lw=0.3)

    # plot Galactic center
    plt.scatter(0,0, marker='o', color='r')
    # plot Earth
    earth_cords = [-8.5, 0, 0]
    plt.scatter(earth_cords[axis[0] - 1], earth_cords[axis[1] - 1], marker='P', color='b')
    #Plot SGR 1900+14
    sgr_cords = [12.5*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.5, 12.5*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), 12.5*np.sin(0.77*np.pi/180)]
    plt.scatter(sgr_cords[axis[0] - 1], sgr_cords[axis[1] - 1], marker='+', c='red', s=70) #43.02 0.77 12.5±1.7
    labels = ['I', 'X', 'Y', 'Z']
    plt.xlabel(f"{labels[axis[0]]} / kpc")
    plt.ylabel(f"{labels[axis[1]]} / kpc")

    plt.show()

def plotAll2DProjections(data, title:str = '', fname:str = '') -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    fig.suptitle(title, y=1)
    I,X,Y,Z = data
    projections = [ax1, ax2, ax3]
    for projection, axis in enumerate([(1,2), (2,3), (1,3)]):
        for i in np.unique(I):
            projections[projection].plot([I,X,Y,Z][axis[0]][I == i], [I,X,Y,Z][axis[1]][I == i], lw=0.1, alpha=0.3, color='green')
        # plot Galactic center
        projections[projection].scatter(0,0, marker='o', color='black', alpha=0.5)
        # plot Earth
        earth_cords = [0, -8.5, 0, 0] #first 0 stands for I in data for generalising formulas (previously in scatter I've written earth_cords[axis[0] - 1])
        projections[projection].scatter(earth_cords[axis[0]], earth_cords[axis[1]], marker='P', color='blue')
        #Plot SGR 1900+14
        sgr_cords = [0, 12.5*np.cos(43.02*np.pi/180)*np.cos(0.77*np.pi/180) - 8.5, 12.5*np.sin(43.02*np.pi/180)*np.cos(0.77*np.pi/180), 12.5*np.sin(0.77*np.pi/180)]
        projections[projection].scatter(sgr_cords[axis[0]], sgr_cords[axis[1]], marker='+', c='red', s=80) #43.02 0.77 12.5±1.7
        #Plot GRS 1915+105
        grs_cords = [0, 8.6*np.cos(45.37*np.pi/180)*np.cos(-0.22*np.pi/180) - 8.5, 8.6*np.sin(45.37*np.pi/180)*np.cos(-0.22*np.pi/180), 8.6*np.sin(-0.22*np.pi/180)]
        projections[projection].scatter(grs_cords[axis[0]], grs_cords[axis[1]], marker='+', c='green', s=70) #45.37 -0.22 8.6+2.0-1.6
        #Plot SS 433 Мікроквазар 39.69 -2.24 5.5±0.2
        ss_cords = [0, 5.5*np.cos(39.69*np.pi/180)*np.cos(-2.24*np.pi/180) - 8.5, 5.5*np.sin(39.69*np.pi/180)*np.cos(-2.24*np.pi/180), 5.5*np.sin(-2.24*np.pi/180)]
        projections[projection].scatter(ss_cords[axis[0]], ss_cords[axis[1]], marker='+', c='purple', s=70) #39.69 -2.24 5.5±0.2
        #Plot NGC 6760 Кулясте скупчення 36.11 -3.9 7.4±0.4
        ngc_cords = [0, 7.4*np.cos(36.11*np.pi/180)*np.cos(-3.9*np.pi/180) - 8.5, 7.4*np.sin(36.11*np.pi/180)*np.cos(-3.9*np.pi/180), 7.4*np.sin(-3.9*np.pi/180)]
        projections[projection].scatter(ngc_cords[axis[0]], ngc_cords[axis[1]], marker='+', c='magenta', s=70) #36.11 -3.9 7.4±0.4

        labels = ['I', 'X', 'Y', 'Z']
        projections[projection].set(xlabel=f"{labels[axis[0]]} / kpc", ylabel=f"{labels[axis[1]]} / kpc")
        projections[projection].set_title(f"{labels[axis[0]]}O{labels[axis[1]]} flat surface")

        #Legend
        from matplotlib.lines import Line2D
        legend_elements = [
                          Line2D([0], [0], marker='+', color='red', label='SGR 1900+14', markerfacecolor='red', linestyle='', markersize=8),
                          Line2D([0], [0], marker='+', color='green', label='GRS 1915+105', markerfacecolor='green', linestyle='', markersize=8),
                          Line2D([0], [0], marker='+', color='purple', label='SS 433', markerfacecolor='purple', linestyle='', markersize=8),
                          Line2D([0], [0], marker='+', color='magenta', label='NGC 6760', markerfacecolor='magenta', linestyle='', markersize=8),
                          Line2D([0], [0], marker='P', color='blue', label='Earth', markerfacecolor='blue', linestyle='', markersize=8),
                          Line2D([0], [0], marker='o', color='black', label='Galaxy Center', markerfacecolor='black', linestyle='', markersize=8),
                          #Line2D([0], [0], lw=1, color='blue', label='H'),
                          #Line2D([0], [0], lw=1, color='blue', label='$\overline{H}$'),
                          #Line2D([0], [0], lw=1, color='orange', label='He'),
                          #Line2D([0], [0], lw=1, color='green', label='H'),
                          #Line2D([0], [0], lw=1, color='red', label='F'),
                          ]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()

def plot3D(data) -> None:
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111, projection='3d')

    # plot trajectories
    I,X,Y,Z = data
    print(I.shape, X[I == 0].shape, np.unique(I))
    for i in np.unique(I):
        ax.plot(X[I == i], Y[I == i], Z[I == i], lw=1, alpha=1)

    # plot Galactic border
    r = 20
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100))
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    ax.plot_surface(x, y, z, rstride=2, cstride=2, color='r', alpha=0.1, lw=0)
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='k', alpha=0.5, lw=0.3)

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

if __name__ == '__main__':
    # plot 3D
    data_22 = np.genfromtxt('traj_PA+TA_Fe_22_event_10000sims.txt', unpack=True, skip_footer=1)
    data_23 = np.genfromtxt('traj_PA+TA_Fe_23_event_10000sims.txt', unpack=True, skip_footer=1)
    data_30 = np.genfromtxt('traj_PA+TA_Fe_30_event_10000sims.txt', unpack=True, skip_footer=1)
    #plot3D(np.genfromtxt('galactic_trajectories_with_uncert_with_random_4types.txt', unpack=True, skip_footer=1))
    #plot2DProjection(np.genfromtxt('one_particle_galactic_trajectories_with_uncert_with_random_C.txt', unpack=True, skip_footer=1), (2,3))
    plotAll2DProjections(data_22, title = "All three projections for the simulated CR with Z = 26 and random striated+turbulent field for the TA event (top)", fname = "Fe_comb_22event_3projections.png")
    plotAll2DProjections(data_23, title = "All three projections for the simulated CR with Z = 26 and random striated+turbulent field for the TA event (bottom)", fname = "Fe_comb_23event_3projections.png")
    plotAll2DProjections(data_30, title = "All three projections for the simulated CR with Z = 26 and random striated+turbulent field for the PA event", fname = "Fe_comb_30event_3projections.png")
