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
                #if mag.split(',')[0] == 'SGR 1935+2154' or mag.split(',')[0] == 'SGR 2013+34' or mag.split(',')[0] == 'GRB1900+14': continue
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
        #plt.text(lons[5]+10*np.pi/180, lats[5]-10*np.pi/180, clust_names[5], fontsize=10, fontweight='bold')#Coma
        #plt.text(lons[6]+10*np.pi/180, lats[6], clust_names[6], fontsize=10, fontweight='bold')#Virgo
        plt.text(lons[7]+1*np.pi/180, lats[7], clust_names[7], fontsize=10, fontweight='bold')#F
        plt.text(lons[8]+1*np.pi/180, lats[8], clust_names[8], fontsize=10, fontweight='bold')#E
        plt.text(47.66190266*np.pi/180-2*47.66190266*np.pi/180-15*np.pi/180, 10.98251055*np.pi/180 + 15*np.pi/180, 'Local Void', fontsize=10, fontweight='bold')#Local Void

    def makeLegend(self) -> list:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='black', lw=1, label='Galaxy clusters'),
                            Line2D([0], [0], color='purple', lw=1, label='Local Void'),
                            #Line2D([0], [0], marker='*', color='orange', label='CRs E > 100 EeV', markerfacecolor='orange', linestyle='', markersize=8),
                            Line2D([0], [0], marker='*', color='orange', label='CRs from PAO', markerfacecolor='orange', linestyle='', markersize=8),
                            Line2D([0], [0], marker='*', color='gold', label='CRs from TA', markerfacecolor='gold', linestyle='', markersize=8),
                            #Line2D([0], [0], marker='o', color='blue', label='Simulated CRs', markerfacecolor='blue', linestyle='', markersize=8), #make variable colors
                            Line2D([0], [0], marker='p', color='pink', label='Magnetars', markerfacecolor='pink', linestyle='', markersize=8),
                            Line2D([0], [0], marker='D', color='turquoise', label='Starburst galaxies', markerfacecolor='turquoise', linestyle='', markersize=8),
                            Line2D([0], [0], marker='p', color='pink', label='SGR 1900+14', markerfacecolor='pink', linestyle='', markersize=8),
                            Line2D([0], [0], marker='+', color='red', label='GRS 1915+105', markerfacecolor='red', linestyle='', markersize=8),
                            Line2D([0], [0], marker='+', color='purple', label='NGC 6760', markerfacecolor='purple', linestyle='', markersize=8),
                            Line2D([0], [0], marker='+', color='cyan', label='SS 433', markerfacecolor='cyan', linestyle='', markersize=8),
                            Line2D([0], [0], marker='+', color='green', label='MGRO 1908', markerfacecolor='green', linestyle='', markersize=8),
                            Line2D([0], [0], marker='+', color='magenta', label='Cygnus OB2', markerfacecolor='magenta', linestyle='', markersize=8),
                            Line2D([0], [0], marker='p', color='pink', label='SGR 2013+34', markerfacecolor='pink', linestyle='', markersize=8),
                            Line2D([0], [0], marker='p', color='pink', label='SGR 1935+2154', markerfacecolor='pink', linestyle='', markersize=8),
                            Line2D([0], [0], marker='o', color='red', label='EHECR triplet', markerfacecolor='none', markeredgecolor='red', 
                                   markeredgewidth=1.5, linestyle='', markersize=8)
                            ]
        return legend_elements

    def plotMap(self, sim=True, transform=None, saving=True, sgr=True, grs=False, sgr_2013=False, sgr_1935=False,
                ss=False, ngc=False, milagro=False, cygnus=False, aquila = False, shapley=False, legend=True, custom_frame=False):
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import numpy as np
        from astropy.wcs import WCS
        from astropy.visualization.wcsaxes.frame import EllipticalFrame
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from cut_visualisation import transform_pandas_galactocentric_to_galactic
        '''Description'''

        #General parameters
        plt.figure(figsize=self.figsize)
        plt.subplot(111, projection = self.projection)
        plt.grid(True)
        plt.title(self.title, fontsize = 12, y=1.01)

        #Plotting simulations
        if sim:
            events_in_void = [2, 4, 6, 21, 39, 40, 53, 56, 72, 75, 80] 
            model_configs = [
                ('JF12', 'X', '#466BC7'),
                ('UF23', 'D', '#B9C76B')
            ]
            
            particle = 'He'
            for event in events_in_void: 
                for model_name, marker_shape, color in model_configs:
                    
                    file_path = f'trajectories_data/traj_for_hammer_init_plot/traj_PA+TA_{model_name}_{particle}_{event}_event_1sims.txt'
                    data = np.genfromtxt(file_path, unpack=True, skip_footer=1)

                    I, X, Y, Z = data
                    pd_data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

                    # Dropping the initial injection points (the observer singularity)
                    pd_data = pd_data.iloc[10:].reset_index(drop=True)

                    lon, lat = transform_pandas_galactocentric_to_galactic(pd_data)

                    plt.plot(-lon, lat, color=color, linewidth=1, alpha=0.7)
                    
                    #Plot the hollow end marker
                    plt.plot(-lon[-1], lat[-1], marker=marker_shape, markersize=4, markeredgewidth = 1,
                            markerfacecolor='none', markeredgecolor=color, linestyle='None', alpha=0.8)

            '''
            for particle, color in zip(self.particles, self.colors):
                #Change lons and lats to be [0], [1] in future
                if transform:
                    x, y = np.array([_[1] for _ in np.array(self.total_results[particle])]), np.pi/2 - np.array([_[0] for _ in np.array(self.total_results[particle])])
                    plt.scatter(sim_transform(x), y, marker='o', linewidths=0, s = 10, c=color, alpha=0.5)
                else:
                    x, y = np.array([_[1] for _ in np.array(self.total_results[particle])]), np.pi/2 - np.array([_[0] for _ in np.array(self.total_results[particle])])
                    plt.scatter(x, y, marker='o', linewidths=0, s = 10, c=color, alpha=0.5)
            '''

        #Plotting events
        if transform:
            lons, lats = inits_transform(self.initial_lons, self.initial_lats)
            #ta_lons, ta_lats = lons[:73], lats[:73] 
            #ta_lons, ta_lats = [lons[2], lons[40]], [lats[2], lats[40]]
            #pa_lons, pa_lats = lons[73:], lats[723:]
            #pa_lons, pa_lats = [lons[74]], [lats[74]]

            #LV SOURCES
            ta_lons = [lons[2], lons[4], lons[6], lons[21], lons[39], lons[40], lons[53], lons[56], lons[72]]
            ta_lats = [lats[2], lats[4], lats[6], lats[21], lats[39], lats[40], lats[53], lats[56], lats[72]]
            pa_lons, pa_lats = [lons[75], lons[80]],  [lats[75], lats[80]]

            #CYG OB2 SOURCES
            #ta_lons, ta_lats = [lons[21], lons[39], lons[6], lons[56]], [lats[21], lats[39], lats[6], lats[56]]
            #pa_lons, pa_lats = [],  []

            #SGR 2013 SOURCE
            #ta_lons, ta_lats = [lons[4]], [lats[4]]
            #pa_lons, pa_lats = [],  []

            plt.scatter(pa_lons, pa_lats, marker='*', c='orange', s=50)
            plt.scatter(ta_lons, ta_lats, marker='*', c='gold', s=50)

            annotate_indices = False
            if annotate_indices:
                # Annotate TA points (indices 0-71)
                for i, (lon, lat) in enumerate(zip(ta_lons, ta_lats)):
                    plt.annotate(str(i), 
                                (lon, lat), 
                                textcoords="offset points", 
                                xytext=(5, 5), # Offsets the text slightly from the star
                                ha='center', 
                                fontsize=8)

                # Annotate PA points (indices 72 and above)
                for i, (lon, lat) in enumerate(zip(pa_lons, pa_lats)):
                    original_index = i + 72
                    plt.annotate(str(original_index), 
                                (lon, lat), 
                                textcoords="offset points", 
                                xytext=(5, 5), # Offsets the text slightly from the star
                                ha='center', 
                                fontsize=8)
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
                    if clust_names[i] == "Virgo": continue
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

        #TRIPLET CONTOUR
        x = np.linspace(-np.pi, np.pi, 10000)
        y = np.linspace(-np.pi/2, np.pi/2, 10000)
        X, Y = np.meshgrid(x,y)

        F = (X-(0.751-2*0.751 + 8*np.pi/180))**2 + (Y - (0.0135 - 4*np.pi/180))**2 - (5*np.pi/180)**2
        plt.contour(X,Y,F,[0],colors='red',linewidths=0.75)
        #plt.text(0.751-2*0.751 + 5*np.pi/180, 0.0135 - 6.5*np.pi/180, 'EHECR triplet', fontsize=8
        #         , fontweight='bold', color='red')
        
        from cut_visualisation import get_objects_params
        
        '''
        if shapley:
            shapley_cords = {"RA": 201.9934, "DEC": -31.5014, "z": 0.0487}
            cords = SkyCoord(ra=shapley_cords["RA"]*u.deg, dec=shapley_cords["DEC"]*u.deg, frame='icrs').transform_to("galactic")
            plt.scatter((2*np.pi*u.rad - cords.l.to(u.rad)).value, cords.b.to(u.rad).value, marker='+', c='magenta', s=70)
            plt.text((2*np.pi*u.rad - cords.l.to(u.rad)).value - np.pi*12/180, (cords.b.to(u.rad)).value + 5*np.pi/180, 
                     'Shapley Center', fontsize=8, fontweight='bold')
        '''
        sources_config = {
            'sgr':     {'plot': sgr,     'color': 'pink',     'label': 'SGR 1900+14', 'offset': (-15, 5), 'arrow': True, 'alpha': 0.0},
            'grs':     {'plot': grs,     'color': 'red',    'label': 'GRS 1915',    'offset': (-10, -6), 'arrow': True, 'alpha': 1.0},
            'ss':      {'plot': ss,      'color': 'cyan',    'label': 'SS 433',      'offset': (-10, -10), 'arrow': True, 'alpha': 1.0},
            'ngc':     {'plot': ngc,     'color': 'purple',  'label': 'NGC 6760',    'offset': (-3, -10), 'arrow': True, 'alpha': 1.0},
            'milagro': {'plot': milagro, 'color': 'green',   'label': 'MGRO 1908',   'offset': (2, 7), 'arrow': True, 'alpha': 1.0},
            'cygnus':  {'plot': cygnus,  'color': 'magenta',  	'label':'Cyg OB2',    	'offset' :(-5,-3),	'arrow' :False,	'alpha' :1.0},
            'aquila':  {'plot': aquila,  'color': 'brown',   'label': 'Aquila X-1',  'offset': (8, 0), 'arrow': True, 'alpha': 1.0},
            'sgr_2013': {'plot': sgr_2013,'color': 'pink',   'label': 'SGR 2013+34', 'offset': (-5, 5), 'arrow': True, 'alpha': 0.0},
            'sgr_1935': {'plot': sgr_1935,'color': 'pink',   'label': 'SGR 1935+2154', 'offset': (-20, -8), 'arrow': True, 'alpha': 0.0},
        }

        target_coords_galactocentric, distances, target_coords_equatorial = get_objects_params()
        
        for key, config in sources_config.items():
            if not config['plot']: 
                continue # Skip if the flag (e.g., sgr=False) is false

            color = config['color']
            alpha = config['alpha']
            
            # Coordinate transformations
            candidate_coords_equatorial = target_coords_equatorial[key]
            coords_candidate = SkyCoord(
                ra=candidate_coords_equatorial["RA"]*u.deg, 
                dec=candidate_coords_equatorial["DEC"]*u.deg,
                distance=candidate_coords_equatorial["dist"]*u.kpc, 
                frame='icrs'
            ).transform_to("galactic")
            
            can_lon = coords_candidate.galactic.l
            can_lon.wrap_angle = 180 * u.deg
            lon = can_lon.radian
            lat = coords_candidate.galactic.b.radian
            
            # Plot the scatter point
            plt.scatter(-lon, lat, marker="+", color=color, s=20, alpha=alpha)
            
            # Calculate text position
            text_lon = -lon + (config['offset'][0] * np.pi / 180)
            text_lat = lat + (config['offset'][1] * np.pi / 180)

            if config['arrow']:
                arrow_props = dict(arrowstyle="->", color=color, lw=0.8, shrinkA=2, shrinkB=3)
            else:
                arrow_props = None

            # Text + arrows
            annotate_arrows = False
            if annotate_arrows:
                plt.annotate(
                    config['label'],
                    xy=(-lon, lat),          # The exact point the arrow should point to
                    xytext=(text_lon, text_lat), # The coordinates where the text sits
                    color=color,
                    fontsize=6,
                    arrowprops=arrow_props
                )        
        #Legend
        if legend: plt.legend(handles=self.makeLegend(), loc='upper right')
        #Ticks
        x_tick_labels = ['150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°']
        x_tick_positions = [-5*np.pi/6, -2*np.pi/3, -np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]

        plt.xticks(x_tick_positions, labels=x_tick_labels, fontsize=10)

        '''ONLY FOR ELLIPSE'''

        y_tick_labels = ['', '', '', '', '', '', '', '', '', '', '']
        y_tick_positions = [-75*np.pi/180, -60*np.pi/180, -45*np.pi/180, -30*np.pi/180, -15*np.pi/180, 
                            0,  15*np.pi/180,  30*np.pi/180,  45*np.pi/180,  60*np.pi/180, 75*np.pi/180,]
        
        plt.yticks(y_tick_positions, labels=y_tick_labels)

        #TO PLOT WITH LABELS INCIDE CIRCLE
        yticks_crop = [-np.pi*75/180 , -np.pi*60/180, -np.pi*45/180, -np.pi*30/180, -np.pi*15/180, 0 + np.pi*1/180,
                        np.pi*15/180, np.pi*30/180, np.pi*45/180, np.pi*60/180, np.pi*75/180] 
        ylabels_crop = ['-75°', '-60°', '-45°', '-30°', '-15°', '0°',
                        '15°', '30°', '45°', '60°', '75°']
        x_cords_adjusted = [-np.pi*174/180 , -np.pi*175/180, -np.pi*176/180, -np.pi*176/180, -np.pi*177/180, -np.pi*177/180,
                            -np.pi*175/180, -np.pi*174/180, -np.pi*169/180, -np.pi*164/180, -np.pi*140/180]

        for pos, label, x_adj in zip(yticks_crop, ylabels_crop, x_cords_adjusted):
            plt.text(x_adj, pos, label, fontsize=10)

        #Hiding all the spines (the lines that form the box)
        ax_gca = plt.gca()
        #for spine in ax_gca.spines.values():
        #    spine.set_visible(False)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if saving: plt.savefig(self.save_name, dpi=600, bbox_inches='tight',
                               pad_inches=0, transparent=False)
        plt.show()


