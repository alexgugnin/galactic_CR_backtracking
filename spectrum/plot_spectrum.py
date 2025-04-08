import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    energies = []
    with open('../data/AugerApJS2022_highE.dat', 'r') as infile:
        for line in infile:
            if line.split()[0] == '#': continue
            temp_event = line.split()
            energies.append(float(temp_event[8]))
    energies.append(244.0) #AMATERASU event
    energies = np.array(sorted(energies))*1e18 #In EeV
    
    #x = np.logspace(19, 20.5, 100) #binsize 3.3 EeV?
    x = np.logspace(20, 20.5, 20)
    counts = []
    for i in range(len(x) - 1):
        lower_bound = x[i]
        upper_bound = x[i + 1]
        count = np.sum((energies >= lower_bound) & (energies < upper_bound))
        counts.append(count)
    
    print(len(energies), sum(counts))
    results = pd.DataFrame({'x': x[:-1],
                            'counts': counts})
    
    #GZK threshold
    #results = results[results['x'] >= 5e19]
    results = results[results['counts'] != 0.0]
    
    plt.tight_layout()
    plt.hist(results['counts'])
    plt.savefig("spectrum_over_gzk_hist.jpeg", dpi=300, bbox_inches='tight')
    exit()
    plt.scatter(x = results['x'], y = results['counts'], c='#ff7f00', s=10)
    #GZK LIMIT
    #plt.plot(np.full(100, 5e19), np.linspace(0, max(results['counts']), 100), c='#984ea3')
    #plt.text(5.1e19,150,"GZK Limit",fontsize=14, rotation=270)
    
    plt.xscale('log')
    plt.xlabel("E, eV", fontsize=16)
    plt.ylabel("Num of events per energy bin", fontsize=16)
    plt.savefig("spectrum_under_gzk.jpeg", dpi=300, bbox_inches='tight')
    