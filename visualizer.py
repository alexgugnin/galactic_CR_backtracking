def sim_transform(arr):
    arr_transf = []
    for lon in arr:
        if lon >= 0:
            lon = lon-2*lon
            arr_transf.append(lon)
            continue
        if lon < 0:
            lon = lon+2*(-lon)
            arr_transf.append(lon)

    return arr_transf

def source_transform(lons):
    import numpy as np
    lons -= np.pi
    temp_lons = []
    for lon in lons:
        if lon >= 0: lon = np.pi - lon
        if lon < 0: lon = -np.pi - lon
        temp_lons.append(lon)

    return temp_lons

def inits_transform(initial_lons, initial_lats) -> tuple:
    import numpy as np
    initial_lats = np.array([np.pi/2 - lat for lat in initial_lats])
    '''
    TRANSFORMATION for inits
    '''
    temp = []
    initial_lons = np.array(initial_lons)
    for lon in initial_lons:
        if lon >= np.pi: lon -= np.pi*2
        temp.append(lon)
    temp_lons = []
    for lon in temp:
        if lon >= 0:
            lon = lon-2*lon
            temp_lons.append(lon)
            continue
        if lon < 0:
            lon = lon+2*(-lon)
            temp_lons.append(lon)

    return temp_lons, initial_lats


class SimMap(object):
    def __init__(self, total_results, initial_lats:list, initial_lons:list,
                 particles:list = ['H', 'He', 'C', 'Fe'],
                 colors:list = ['blue', 'orange', 'green', 'red']
                 ):

        self.total_results = total_results
        self.initial_lats:list = initial_lats
        self.initial_lons:list = initial_lons
        self.particles:list = particles
        self.colors:list = colors
        #Plot params
        self.figsize:tuple = (12, 7)
        self.projection:str = 'hammer'
        self.is_grid:bool = True
        self.title:str = 'Title'
        self.save_name:str = 'fname'
        #Potential sources
        self.sources_flags:dict = {
            'mags': False,
            'sbgs': False,
            'clusts': False
        }

    def getFigsize(self) -> tuple:
        return self.figsize

    def setFigsize(self, new_figsize) -> None:
        self.figsize = new_figsize

    def getProjection(self) -> str:
        return self.projection

    def setProjection(self, new_projection) -> None:
        self.projection = new_projection

    def getIsGrid(self) -> bool:
        return self.is_grid

    def setIsGrid(self, new_is_grid) -> None:
        self.is_grid = new_is_grid

    def getTitle(self) -> str:
        return self.title

    def setTitle(self, new_title) -> None:
        self.title = new_title

    def getSaveName(self) -> str:
        return self.save_name

    def setSaveName(self, new_save_name) -> None:
        self.save_name = new_save_name

    def getSourcesFlags(self) -> dict:
        return self.sources_flags

    def setSourcesFlags(self, new_flags) -> None:
        for key in self.sources_flags.keys():
            try:
                self.sources_flags[key] = new_flags[key]
            except: continue

    def gatherSources(self, data_path:str) -> tuple:
        '''Make this func when will be able to generalise data structure'''
        return (0,0)

    def gatherMags(self, data_path:str) -> tuple:
        '''Hard coded params of the data structure, should be changed to more generalised in future'''
        import numpy as np
        mags_lons, mags_lats = [], []
        with open(data_path, 'r') as mags:
            for mag in mags:
                if mag.split()[0][0] == 'N': continue
                mags_lons.append(float(mag.split(',')[7]))
                mags_lats.append(float(mag.split(',')[8]))
        mags_lons, mags_lats = (np.array(mags_lons)/180)*np.pi, (np.array(mags_lats)/180)*np.pi

        return mags_lons, mags_lats

    def gatherSbgs(self, data_path:str) -> tuple:
        '''Hard coded params of the data structure, should be changed to more generalised in future'''
        import numpy as np
        sbgs_lons, sbgs_lats = [], []
        with open(data_path, 'r') as sbgs:
            for sbg in sbgs:
                if sbg.split()[0][0] == 'N': continue
                sbgs_lons.append(float(sbg.split(',')[1]))
                sbgs_lats.append(float(sbg.split(',')[2]))
        sbgs_lons, sbgs_lats = (np.array(sbgs_lons)/180)*np.pi, (np.array(sbgs_lats)/180)*np.pi

        return sbgs_lons, sbgs_lats

    def gatherClusts(self, data_path:str) -> tuple:
        '''Hard coded params of the data structure, should be changed to more generalised in future'''
        import numpy as np
        clust_names, clust_lons, clust_lats, clust_radii = [], [], [], []
        with open(data_path, 'r') as clusts:
            for clust in clusts:
                if clust.split()[0][0] == 'I': continue
                clust_names.append(clust.split(',')[0])
                clust_lons.append(float(clust.split(',')[3]))
                clust_lats.append(float(clust.split(',')[4]))
                clust_radii.append(float(clust.split(',')[5]))
        clust_lons, clust_lats = (np.array(clust_lons)/180)*np.pi, (np.array(clust_lats)/180)*np.pi

        return clust_lons, clust_lats, clust_names, clust_radii

    def createHandles(self) -> list:
        '''
        DEPRECATED, new legend is built
        '''
        import matplotlib.patches as mpatches

        handles = []
        for n, particle in enumerate(self.particles):
            handles.append(mpatches.Patch(color=self.colors[n], label=particle))

        return handles

    def plotClustNames(self, lons, lats, clust_names) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.text(lons[0]-8*np.pi/180, lats[0], clust_names[0], fontsize=10, fontweight='bold')#Centaurus
        plt.text(lons[1]-8*np.pi/180, lats[1], clust_names[1], fontsize=10, fontweight='bold')#Hya
        plt.text(lons[2]-2*np.pi/180, lats[2], clust_names[2], fontsize=10, fontweight='bold')#Norm
        plt.text(lons[3]-4*np.pi/180, lats[3], clust_names[3], fontsize=10, fontweight='bold')#PP
        plt.text(lons[4]-4*np.pi/180, lats[4], clust_names[4], fontsize=10, fontweight='bold')#PI
        plt.text(lons[5]+10*np.pi/180, lats[5]-10*np.pi/180, clust_names[5], fontsize=10, fontweight='bold')#Coma
        plt.text(lons[6]+10*np.pi/180, lats[6], clust_names[6], fontsize=10, fontweight='bold')#Virgo
        plt.text(lons[7]+1*np.pi/180, lats[7], clust_names[7], fontsize=10, fontweight='bold')#F
        plt.text(lons[8]+1*np.pi/180, lats[8], clust_names[8], fontsize=10, fontweight='bold')#E
        plt.text(47.66190266*np.pi/180-2*47.66190266*np.pi/180-15*np.pi/180, 10.98251055*np.pi/180 + 15*np.pi/180, 'Local Void', fontsize=10, fontweight='bold')#Local Void

    def makeLegend(self) -> list:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='black', lw=1, label='Galaxy clusters'),
                            Line2D([0], [0], color='purple', lw=1, label='Local Void'),
                            #Line2D([0], [0], marker='*', color='orange', label='CRs E > 100 EeV', markerfacecolor='orange', linestyle='', markersize=8),
                            Line2D([0], [0], marker='*', color='orange', label='CRs from PA', markerfacecolor='orange', linestyle='', markersize=8),
                            Line2D([0], [0], marker='*', color='gold', label='CRs from TA', markerfacecolor='gold', linestyle='', markersize=8),
                            #Line2D([0], [0], marker='o', color='blue', label='Simulated CRs', markerfacecolor='blue', linestyle='', markersize=8), #make variable colors
                            Line2D([0], [0], marker='p', color='pink', label='Magnetars', markerfacecolor='pink', linestyle='', markersize=8),
                            Line2D([0], [0], marker='D', color='turquoise', label='starburst galaxies', markerfacecolor='turquoise', linestyle='', markersize=8),
                            Line2D([0], [0], marker='+', color='red', label='SGR 1900+14', markerfacecolor='red', linestyle='', markersize=8)
                            ]
        return legend_elements

    def plotMap(self, sim=True, transform=None, saving=True, sgr=True, grs=False,
                ss=False, ngc=False, legend=True, custom_frame=False):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import numpy as np
        from astropy.wcs import WCS
        from astropy.visualization.wcsaxes.frame import EllipticalFrame
        '''Description'''

        #General parameters
        plt.figure(figsize=self.figsize)
        plt.subplot(111, projection = self.projection)
        plt.grid(True)
        plt.title(self.title, y=1.05)

        #Plotting simulations
        if sim:
            for particle, color in zip(self.particles, self.colors):
                #Change lons and lats to be [0], [1] in future
                if transform:
                    x, y = np.array([_[1] for _ in np.array(self.total_results[particle])]), np.pi/2 - np.array([_[0] for _ in np.array(self.total_results[particle])])
                    plt.scatter(sim_transform(x), y, marker='o', linewidths=0, s = 10, c=color, alpha=0.5)
                else:
                    x, y = np.array([_[1] for _ in np.array(self.total_results[particle])]), np.pi/2 - np.array([_[0] for _ in np.array(self.total_results[particle])])
                    plt.scatter(x, y, marker='o', linewidths=0, s = 10, c=color, alpha=0.5)

        #Plotting events
        if transform:
            lons, lats = inits_transform(self.initial_lons, self.initial_lats)
            ta_lons, ta_lats = [lons[16], lons[18], lons[19], lons[20], lons[22], lons[23], lons[24], lons[25]], [lats[16], lats[18], lats[19], lats[20], lats[22], lats[23], lats[24], lats[25]]
            pa_lons, pa_lats = [lons[30]], [lats[30]]
            #ta_lons, ta_lats = lons[:28], lats[:28]
            #pa_lons, pa_lats = lons[28:], lats[28:]
            plt.scatter(pa_lons, pa_lats, marker='*', c='orange', s=50)
            plt.scatter(ta_lons, ta_lats, marker='*', c='gold', s=50)
        else:
            plt.scatter(self.initial_lons, self.initial_lats, marker='*', c='orange', s=50)
        #Plotting sources

        #Ideal version, when data generalisation will be done
        #for source in self.sources_flags.keys():
        #    if self.sources_flags[source]:
        #        lons, lats = self.gatherSourcesData(path)
        if self.sources_flags['mags']:
            lons, lats = self.gatherMags("potential_sources/magnetars.csv")
            if transform:
                plt.scatter(source_transform(lons), lats, marker='p', c='pink', s=25)
            else:
                plt.scatter(lons, lats, marker='p', c='pink', s=25)
        if self.sources_flags['sbgs']:
            lons, lats = self.gatherSbgs("potential_sources/SBGs_under50Mpc.csv")
            if transform:
                plt.scatter(source_transform(lons), lats, marker='D', c='turquoise', s=15)
            else:
                plt.scatter(lons, lats, marker='D', c='turquoise', s=15)
        if self.sources_flags['clusts']:
            lons, lats, clust_names, clust_radii = self.gatherClusts("potential_sources/Clust_circle_ICRC.csv")
            x = np.linspace(-np.pi, np.pi, 10000)
            y = np.linspace(-np.pi/2, np.pi/2, 10000)
            X, Y = np.meshgrid(x,y)
            if transform:
                lons = source_transform(lons)
                for i in range(len(lons)):
                    if clust_names[i] == "Coma": continue
                    F = (X-lons[i])**2 + (Y-lats[i])**2 - (clust_radii[i]*np.pi/180)**2
                    plt.contour(X,Y,F,[0],colors='black',linewidths=0.75)
                F = (X-(47.66190266*np.pi/180-2*47.66190266*np.pi/180))**2 + (Y-10.98251055*np.pi/180)**2 - (40*np.pi/180)**2#Local Void
                plt.contour(X,Y,F,[0],colors='purple',linewidths=0.75)
                self.plotClustNames(lons, lats, clust_names)
            else:
                for i in range(len(lons)):
                    if clust_names[i] == "Coma": continue
                    F = (X-lons[i])**2 + (Y-lats[i])**2 - (clust_radii[i]*np.pi/180)**2
                    plt.contour(X,Y,F,[0],colors='black',linewidths=0.75)
                F = (X-(47.66190266*np.pi/180-2*47.66190266*np.pi/180))**2 + (Y-10.98251055*np.pi/180)**2 - (40*np.pi/180)**2#Local Void
                plt.contour(X,Y,F,[0],colors='purple',linewidths=0.75)
                self.plotClustNames(lons, lats, clust_names)

        #SGR
        if sgr:
            sgr = plt.scatter(0.751-2*0.751, 0.0135, marker='+', c='red', s=70)#SGR 1900+14
            plt.text(0.751-2*0.751 - 15*np.pi/180, 0.0135+7*np.pi/180, 'SGR 1900+14', fontsize=8, fontweight='bold')#SGR 1900+14
        #Plot GRS 1915+105
        if grs:
            grs_cords = [0, 8.6*np.cos(45.37*np.pi/180)*np.cos(-0.22*np.pi/180) - 8.5, 8.6*np.sin(45.37*np.pi/180)*np.cos(-0.22*np.pi/180), 8.6*np.sin(-0.22*np.pi/180)]
            projections[projection].scatter(grs_cords[axis[0]], grs_cords[axis[1]], marker='+', c='green', s=70) #45.37 -0.22 8.6+2.0-1.6
        #Plot SS 433 Мікроквазар 39.69 -2.24 5.5±0.2
        if ss:
            ss_cords = [0, 5.5*np.cos(39.69*np.pi/180)*np.cos(-2.24*np.pi/180) - 8.5, 5.5*np.sin(39.69*np.pi/180)*np.cos(-2.24*np.pi/180), 5.5*np.sin(-2.24*np.pi/180)]
            projections[projection].scatter(ss_cords[axis[0]], ss_cords[axis[1]], marker='+', c='purple', s=70) #39.69 -2.24 5.5±0.2
        #Plot NGC 6760 Кулясте скупчення 36.11 -3.9 7.4±0.4
        if ngc:
            ngc_cords = [0, 7.4*np.cos(36.11*np.pi/180)*np.cos(-3.9*np.pi/180) - 8.5, 7.4*np.sin(36.11*np.pi/180)*np.cos(-3.9*np.pi/180), 7.4*np.sin(-3.9*np.pi/180)]
            projections[projection].scatter(ngc_cords[axis[0]], ngc_cords[axis[1]], marker='+', c='magenta', s=70) #36.11 -3.9 7.4±0.4
        #Legend
        if legend: plt.legend(handles=self.makeLegend(), loc='upper right')
        #Ticks
        x_tick_labels = ['150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°']
        x_tick_positions = [-5*np.pi/6, -2*np.pi/3, -np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]

        plt.xticks(x_tick_positions, labels=x_tick_labels)

        if saving: plt.savefig(self.save_name, dpi=300, bbox_inches='tight')
        plt.show()


def visualizeTotal(total_results, initial_lats:list, initial_lons:list) -> None:
    '''
    Graph builder for DataFrame generated by makeDF
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    import numpy as np

    #Graph customization
    plt.figure(figsize=(12,7))
    plt.subplot(111, projection = 'hammer')
    plt.grid(True)
    plt.title(f"Simulated CR with Z = 1.")
    '''
    Patches for multi Z
    '''
    H_patch = mpatches.Patch(color='blue', label='H')
    He_patch = mpatches.Patch(color='orange', label='He')
    N_patch = mpatches.Patch(color='green', label='N')
    Fe_patch = mpatches.Patch(color='red', label='Fe')
    plt.legend(handles=[H_patch, He_patch, N_patch, Fe_patch])

    #Getting coordinates
    hx, hy = np.array([_[1] for _ in np.array(total_results['H'])]), np.pi/2 - np.array([_[0] for _ in np.array(total_results['H'])])
    hex, hey = np.array([_[1] for _ in np.array(total_results['He'])]), np.pi/2 - np.array([_[0] for _ in np.array(total_results['He'])])
    nx, ny = np.array([_[1] for _ in np.array(total_results['C'])]), np.pi/2 - np.array([_[0] for _ in np.array(total_results['C'])])
    fex, fey = np.array([_[1] for _ in np.array(total_results['Fe'])]), np.pi/2 -  np.array([_[0] for _ in np.array(total_results['Fe'])])


    #Sources
    mags_lons, mags_lats = [], []
    with open("potential_sources/magnetars.csv", 'r') as mags:
        for mag in mags:
            if mag.split()[0][0] == 'N': continue
            mags_lons.append(float(mag.split(',')[7]))
            mags_lats.append(float(mag.split(',')[8]))
    mags_lons, mags_lats = (np.array(mags_lons)/180)*np.pi, (np.array(mags_lats)/180)*np.pi

    sbgs_lons, sbgs_lats = [], []
    with open("potential_sources/SBGs_under50Mpc.csv", 'r') as sbgs:
        for sbg in sbgs:
            if sbg.split()[0][0] == 'N': continue
            sbgs_lons.append(float(sbg.split(',')[1]))
            sbgs_lats.append(float(sbg.split(',')[2]))
    sbgs_lons, sbgs_lats = (np.array(sbgs_lons)/180)*np.pi, (np.array(sbgs_lats)/180)*np.pi

    clust_names, clust_lons, clust_lats, clust_radii = [], [], [], []
    with open("potential_sources/Clust_circle_ICRC.csv", 'r') as clusts:
        for clust in clusts:
            if clust.split()[0][0] == 'I': continue
            clust_names.append(clust.split(',')[0])
            clust_lons.append(float(clust.split(',')[3]))
            clust_lats.append(float(clust.split(',')[4]))
            clust_radii.append(float(clust.split(',')[5]))
    clust_lons, clust_lats = (np.array(clust_lons)/180)*np.pi, (np.array(clust_lats)/180)*np.pi

    #Plotting
    '''
    TRANSFORMATION for sim
    '''
    plt.scatter(sim_transform(hx), hy, marker='o', linewidths=0, s = 15, c='blue', alpha=0.5)
    plt.scatter(sim_transform(hex), hey, marker='o', linewidths=0, s = 15, c = 'orange', alpha=0.5)
    plt.scatter(sim_transform(nx), ny, marker='o', linewidths=0, s = 15, c = 'green', alpha=0.5)
    plt.scatter(sim_transform(fex), fey, marker='o', linewidths=0, s = 15, c = 'red', alpha=0.5)

    initial_lats = np.array([np.pi/2 - lat for lat in initial_lats])
    '''
    TRANSFORMATION for inits
    '''
    temp = []
    initial_lons = np.array(initial_lons)
    for lon in initial_lons:
        if lon >= np.pi: lon -= np.pi*2
        temp.append(lon)
    temp_lons = []
    for lon in temp:
        if lon >= 0:
            lon = lon-2*lon
            temp_lons.append(lon)
            continue
        if lon < 0:
            lon = lon+2*(-lon)
            temp_lons.append(lon)
    plt.scatter(temp_lons, initial_lats, marker='*', c='orange', s=50)#initial state

    '''
    TRANSFORMATION mags
    '''
    mags_lons -= np.pi
    mtemp_lons = []
    for lon in mags_lons:
        if lon >= 0: lon = np.pi - lon
        if lon < 0: lon = -np.pi - lon
        mtemp_lons.append(lon)

    plt.scatter(mtemp_lons, mags_lats, marker='p', c='pink', s=25)#magnetars

    '''
    TRANSFORMATION sbg
    '''
    sbgs_lons -= np.pi
    sbgtemp_lons = []
    for lon in sbgs_lons:
        if lon >= 0: lon = np.pi - lon
        if lon < 0: lon = -np.pi - lon
        sbgtemp_lons.append(lon)
    plt.scatter(sbgtemp_lons, sbgs_lats, marker='D', c='turquoise', s=15)#starburst galaxies

    sgr = plt.scatter(0.751-2*0.751, 0.0135, marker='+', c='red', s=70)#SGR 1900+14
    plt.text(0.751-2*0.751 - 15*np.pi/180, 0.0135+7*np.pi/180, 'SGR 1900+14', fontsize=8, fontweight='bold')#SGR 1900+14

    '''
    CLUSTERS
    '''
    clust_lons -= np.pi
    clusttemp_lons = []
    for lon in clust_lons:
        if lon >= 0: lon = np.pi - lon
        if lon < 0: lon = -np.pi - lon
        clusttemp_lons.append(lon)

    x = np.linspace(-np.pi, np.pi, 10000)
    y = np.linspace(-np.pi/2, np.pi/2, 10000)
    X, Y = np.meshgrid(x,y)
    for i in range(len(clusttemp_lons)):
        if clust_names[i] == "Coma": continue
        F = (X-clusttemp_lons[i])**2 + (Y-clust_lats[i])**2 - (clust_radii[i]*np.pi/180)**2
        plt.contour(X,Y,F,[0],colors='black',linewidths=0.75)
    F = (X-(47.66190266*np.pi/180-2*47.66190266*np.pi/180))**2 + (Y-10.98251055*np.pi/180)**2 - (40*np.pi/180)**2#Local Void
    plt.contour(X,Y,F,[0],colors='purple',linewidths=0.75)
    '''
    CLUSTER TEXT
    '''
    plt.text(clusttemp_lons[0]-8*np.pi/180, clust_lats[0], clust_names[0], fontsize=10, fontweight='bold')#Centaurus
    plt.text(clusttemp_lons[1]-8*np.pi/180, clust_lats[1], clust_names[1], fontsize=10, fontweight='bold')#Hya
    plt.text(clusttemp_lons[2]-2*np.pi/180, clust_lats[2], clust_names[2], fontsize=10, fontweight='bold')#Norm
    plt.text(clusttemp_lons[3]-4*np.pi/180, clust_lats[3], clust_names[3], fontsize=10, fontweight='bold')#PP
    plt.text(clusttemp_lons[4]-4*np.pi/180, clust_lats[4], clust_names[4], fontsize=10, fontweight='bold')#PI
    plt.text(clusttemp_lons[5]+10*np.pi/180, clust_lats[5]-10*np.pi/180, clust_names[5], fontsize=10, fontweight='bold')#Coma
    plt.text(clusttemp_lons[6]+10*np.pi/180, clust_lats[6], clust_names[6], fontsize=10, fontweight='bold')#Virgo
    plt.text(clusttemp_lons[7]+1*np.pi/180, clust_lats[7], clust_names[7], fontsize=10, fontweight='bold')#F
    plt.text(clusttemp_lons[8]+1*np.pi/180, clust_lats[8], clust_names[8], fontsize=10, fontweight='bold')#E
    plt.text(47.66190266*np.pi/180-2*47.66190266*np.pi/180-15*np.pi/180, 10.98251055*np.pi/180 + 15*np.pi/180, 'Local Void', fontsize=10, fontweight='bold')#Local Void

    '''
    LEGEND
    '''
    legend_elements = [Line2D([0], [0], color='black', lw=1, label='Galaxy clusters'),
                        Line2D([0], [0], color='purple', lw=1, label='Local Void'),
                        Line2D([0], [0], marker='*', color='orange', label='CRs E(EeV) > 100', markerfacecolor='orange', linestyle='', markersize=8),
                        Line2D([0], [0], marker='o', color='blue', label='Simulated CRs', markerfacecolor='blue', linestyle='', markersize=8),
                        Line2D([0], [0], marker='p', color='pink', label='Magnetars', markerfacecolor='pink', linestyle='', markersize=8),
                        Line2D([0], [0], marker='D', color='turquoise', label='starburst galaxies', markerfacecolor='turquoise', linestyle='', markersize=8),
                        Line2D([0], [0], marker='+', color='red', label='SGR 1900+14', markerfacecolor='red', linestyle='', markersize=8)
                        ]
    plt.legend(handles=legend_elements, loc='upper right')

    x_tick_labels = ['180°', '150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°']
    x_tick_positions = [0, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi, np.pi, 3*np.pi/4, np.pi/2, np.pi/4]
    x_tick_positions = [-np.pi, -5*np.pi/6, -2*np.pi/3, -np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]

    plt.xticks(x_tick_positions, labels=x_tick_labels)
    plt.savefig('base_test_sim.png', dpi=300)
    plt.show()
