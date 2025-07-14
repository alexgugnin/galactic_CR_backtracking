import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    '''PLotting energy spectrum from https://arxiv.org/pdf/2008.06486 Fig.9'''
    data = pd.read_csv('data_from_image.csv')
    
    data["Flux"] /= (1e10 * 365*24*60*60 * 1e18)  #cm^2 sr s GeV**2
    data["Err_low"] /= (1e10 * 365*24*60*60 * 1e18) #cm^2 sr s
    data["Err_up"] /= (1e10 * 365*24*60*60 * 1e18) #cm^2 sr s
    data["E"] /= 1e9 #GeV
    
    plt.errorbar(data["E"], data["Flux"], [data["Flux"] - data["Err_low"], data["Err_up"] - data["Flux"]], 
                 fmt="o", )#label='vertical sample 0$^{\circ}$- 60$^{\circ}$')
    
    plt.xlabel('E [GeV]')
    plt.ylabel(r'J(E) x E$^{3}$ [cm$^{-2}$ $\times$ sr$^{-1}$ s$^{-1}$ GeV$^{2}$]')
    #plt.ylabel(r'J(E) x E$^{3}$ [km$^{-2}$ $\times$ sr$^{-1}$ yr$^{-1}$ eV$^{2}$]')
    plt.xscale("log")
    plt.yscale("log")
    #plt.xlim(2e18, 2e20)
    plt.ylim(1.45e1, 2e2)
    plt.show()

    data.to_csv("data_from_image_rescaled.csv")