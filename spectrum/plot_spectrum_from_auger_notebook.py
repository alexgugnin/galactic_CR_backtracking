import pandas as pd
import numpy as np
import scipy.stats
from scipy import stats
import matplotlib.pyplot as plt

def plot_figure(cut_nz, flux_E3, flux_E3_lower, flux_E3_upper, cut_z, FC_CL_E3, bin_energy18, 
                cut_nzi, flux_E3_i, flux_E3_lower_i, flux_E3_upper_i, bin_energy18_i,
                #bin_energy18_LE, cut_nzLE, flux_LE, flux_lower_LE, flux_upper_LE, cut_zLE, FC_CL_LE, 
                ):
    Y_0val2 = FC_CL_E3 * 0.6
    #Y_0val_LE = FC_CL_LE * 0.9
    #Y_0val2_i = FC_CL_E3_i * 0.6


    plt.title(r"Spectrum with flux")
    #plt.errorbar(bin_energy18_LE[cut_nzLE], flux_LE, [flux_lower_LE, flux_upper_LE], fmt="o", label='SD-750', color="orange")
    #plt.errorbar(bin_energy18_LE[cut_zLE], FC_CL_LE, Y_0val_LE, uplims=True, marker="None", color="orange", 
    #            markeredgecolor="r", markerfacecolor="r", linewidth=2.0, linestyle="None", capsize=5)
    
    plt.errorbar(bin_energy18[cut_nz], flux_E3, [flux_E3_lower, flux_E3_upper], fmt="o", label='vertical sample 0$^{\circ}$- 60$^{\circ}$')
    plt.errorbar(bin_energy18[cut_z], FC_CL_E3, Y_0val2, uplims=True, barsabove=True, marker="None", color="steelblue",
                markeredgecolor="r", markerfacecolor="r", linewidth=2.0, linestyle="None", capsize=5)

    plt.errorbar(bin_energy18_i[cut_nzi], flux_E3_i, [flux_E3_lower_i, flux_E3_upper_i], fmt="o", label='inclined sample 60$^{\circ}$- 80$^{\circ}$', color="orange")

    #plt.xlim(2.3e18, 1.5e20)
    #plt.ylim(1e36, 1e39)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('E [eV]')
    plt.ylabel(r'E$^{3} $J$^{Raw}$(E) [cm$^{-2}$ $\times$ sr$^{-1}$ s$^{-1}$ GeV$^{2}$]')

    plt.legend()
    plt.plot()
    plt.show()  


if __name__ == "__main__":
    #Extracting data
    data = pd.read_csv("/home/alexgugnin/galactic_CR_backtracking/data/summary_release_3/dataSummarySD1500.csv")
    data_i = pd.read_csv("/home/alexgugnin/galactic_CR_backtracking/data/summary_release_3/dataSummaryInclined.csv")

    #Taking data with energie more than 2.5 EeV and zenith angle < 60 to have the shower detection efficiency > 97%
    data = data[data["sd_exposure"] > 0]
    data = data[(data.sd_energy.notna()) & (data.sd_energy > 2.5)]
    data = data.sort_values(by = 'sdid')
    exposure = data["sd_exposure"].iat[-1] #km^2 sr year
    exposure *= 1e10 * 365*24*60*60 #cm^2 sr s
    energy = data.drop_duplicates('id')["sd_energy"] #Number of events 21564 so 10 % of all PA data [https://arxiv.org/pdf/1909.09073 Table 1]
    
    #Define the energy bins, selected to be of constant width in the decimal logarithm of the energy 
    # \Delta lg_(E) = 0.1
    log_E_min = 0.4
    E_bins = 20
    E_bin_size = 0.1
    log_E_max = log_E_min + E_bins * E_bin_size

    log_bins = np.linspace(log_E_min, log_E_max, E_bins + 1)
    log_bin_centers = log_bins[:-1] + 0.05
    bins = pow(10, log_bins);
    bin_energy = pow(10, log_bin_centers)
    bin_width = bins[1:] - bins[:-1]

    #Fill the histogram to get the number of events in each energy bin.
    h = np.histogram(energy, bins)[0]
    
    #Calculating lower and upper limits for the 68% confidence interval to account statistical uncertainties
    #1 - alpha - beta = 0.68
    alpha = 0.16
    beta = 0.16
    lim_low = (h - np.nan_to_num(0.5 * scipy.stats.chi2.ppf(alpha, 2 * h)) )
    lim_up = ( 0.5 * scipy.stats.chi2.ppf(1 - beta, 2 * (h + 1)) - h)
    
    #Identify the bins with events and those with no events. In bins without events we calculate upper limits.
    cut_nz = h > 0
    cut_z = h == 0

    #Calculating raw flux without correcting on detector effect (refer to notebook)
    normalization = exposure * bin_width * 1e9 # to be in eV, 1e9 to be in GeV like in https://arxiv.org/pdf/2504.15272
    flux = h[cut_nz] / normalization[cut_nz]
    flux_lower = lim_low[cut_nz] / normalization[cut_nz]
    flux_upper = lim_up[cut_nz] / normalization[cut_nz]
    
    #Rescaling by E**3
    bin_energy18 = bin_energy * 1e9#1e18
    bin_energy18_3 = bin_energy18**3
    flux_E3 = flux * bin_energy18_3[cut_nz]
    flux_E3_lower = flux_lower * bin_energy18_3[cut_nz]
    flux_E3_upper = flux_upper * bin_energy18_3[cut_nz]

    #For bins with 0 events we calculate 90 % confidence level (C.L.) from https://arxiv.org/pdf/physics/9711021
    #2.44 is the limit for Poissonian distribution
    FC_90CL_0 = 2.44

    FC_CL    = FC_90CL_0 / normalization[cut_z]
    FC_CL_E3 = FC_CL * bin_energy18_3[cut_z]
    FC_CLt    = FC_90CL_0 / normalization[cut_z]
    FC_CL_E3t = FC_CLt * bin_energy18_3[cut_z]

    '''
    Same procedure for the inclined events
    '''

    data_i = data_i[data_i["sd_exposure"]>0]
    data_i = data_i[(data_i.sd_energy.notna())&(data_i.sd_energy>4)]
    data_i = data_i.sort_values(by='sdid')
    exposure_i = data_i["sd_exposure"].iat[-1]
    exposure_i *= 1e10 * 365*24*60*60 #cm^2 sr s

    energy_i = data_i.drop_duplicates('id')["sd_energy"]

    log_E_min_i = 0.6
    E_bins_i = 18
    log_E_max_i = log_E_min_i + E_bins_i * E_bin_size

    log_bins_i = np.linspace(log_E_min_i, log_E_max_i, E_bins_i + 1)
    log_bin_centers_i = log_bins_i[:-1] + 0.05
    bins_i = pow(10, log_bins_i);
    bin_energy_i = pow(10, log_bin_centers_i)
    bin_width_i = bins_i[1:] - bins_i[:-1]

    h_i = np.histogram(energy_i, bins_i)[0]


    lim_low_i = (h_i - np.nan_to_num(0.5 * scipy.stats.chi2.ppf(alpha, 2 * h_i)) )
    lim_up_i = ( 0.5 * scipy.stats.chi2.ppf(1 - beta, 2 * (h_i + 1)) - h_i)

    cut_nzi = h_i > 0
    cut_zi = h_i == 0


    normalization_i = exposure_i * bin_width_i * 1e9 #1e18
    flux_i = h_i[cut_nzi] / normalization_i[cut_nzi]
    flux_lower_i = lim_low_i[cut_nzi] / normalization_i[cut_nzi]
    flux_upper_i = lim_up_i[cut_nzi] / normalization_i[cut_nzi]


    bin_energy18_i = bin_energy_i * 1e9 #1e18
    bin_energy18_3_i = bin_energy18_i**3
    flux_E3_i = flux_i * bin_energy18_3_i[cut_nzi]
    flux_E3_lower_i = flux_lower_i * bin_energy18_3_i[cut_nzi]
    flux_E3_upper_i = flux_upper_i * bin_energy18_3_i[cut_nzi]


    FC_CL_i    = FC_90CL_0 / normalization_i[cut_zi]
    FC_CL_E3_i = FC_CL_i * bin_energy18_3_i[cut_zi]

    plot_figure(cut_nz, flux_E3, flux_E3_lower, flux_E3_upper, cut_z, FC_CL_E3, bin_energy18, 
                cut_nzi, flux_E3_i, flux_E3_lower_i, flux_E3_upper_i, bin_energy18_i)
    exit()
    '''
    Same procedure for SD750
    '''

    dataLE = pd.read_csv("/home/alexgugnin/galactic_CR_backtracking/data/summary_release_3/dataSummarySD750.csv")
    dataLE = dataLE[(dataLE.sd_energy.notna())&(dataLE.sd_energy>=0.1)]

    dataLE = dataLE.sort_values(by='sdid')
    exposureLE = dataLE["sd_exposure"].iat[-1]

    energyLE = dataLE.drop_duplicates('id')["sd_energy"] 

    #Bins are defined, according to the reference paper, to be of constant width in the decimal 
    # logarithm of the energy ($ \Delta \mathrm{log}_{10}(E) = 0.1$ below 10 EeV and 0.3 above).
    binslogP = np.array([17., 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18., 18.1, 18.2, 18.3,
                        18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 19.3, 19.6]) 
    log_bin_centersLE = binslogP[:-1] + (binslogP[1:] - binslogP[:-1])/2 
    binsP = pow(10, binslogP-18);
    bin_energyLE = pow(10, log_bin_centersLE)

    bin_widthLE = binsP[1:] - binsP[:-1] 

    hLE = np.histogram(energyLE, binsP)[0]

    lim_lowLE = (hLE - np.nan_to_num(0.5 * scipy.stats.chi2.ppf(alpha, 2 * hLE)) )
    lim_upLE = ( 0.5 * scipy.stats.chi2.ppf(1 - beta, 2 * (hLE + 1)) - hLE)

    cut_nzLE = hLE > 0
    cut_zLE = hLE == 0

    '''
    normalization_LE = exposureLE * bin_widthLE * 1e18
    flux_LE = hLE[cut_nzLE] / normalization_LE[cut_nzLE]
    flux_lower_LE = lim_lowLE[cut_nzLE] / normalization_LE[cut_nzLE]
    flux_upper_LE = lim_upLE[cut_nzLE] / normalization_LE[cut_nzLE]

    bin_energy18_LE = bin_energyLE * 1e18
    bin_energy18_3_LE = bin_energy18_LE**3
    flux_E3_LE = flux_LE * bin_energy18_3_LE[cut_nzLE]
    flux_E3_lower_LE = flux_lower_LE * bin_energy18_3_LE[cut_nzLE]
    flux_E3_upper_LE = flux_upper_LE * bin_energy18_3_LE[cut_nzLE]

    FC_CL_LE = FC_90CL_0 / normalization_LE[cut_zLE]
    FC_CL_LE_3 = FC_CL_LE * bin_energy18_3_LE[cut_zLE]
    print(flux_E3_LE)
    plot_figure(cut_nz, flux_E3, flux_E3_lower, flux_E3_upper, cut_z, FC_CL_E3, bin_energy18, 
                cut_nzi, flux_E3_i, flux_E3_lower_i, flux_E3_upper_i, bin_energy18_i,
                bin_energy18_3_LE, cut_nzLE, flux_E3_LE, flux_E3_lower_LE, flux_E3_upper_LE, cut_zLE, FC_CL_LE_3, 
                )
    '''
    normalization_LE = exposureLE * bin_widthLE * 1e18
    flux_LE = hLE[cut_nzLE] / normalization_LE[cut_nzLE]
    flux_lower_LE = lim_lowLE[cut_nzLE] / normalization_LE[cut_nzLE]
    flux_upper_LE = lim_upLE[cut_nzLE] / normalization_LE[cut_nzLE]

    bin_energy18_LE = bin_energyLE

    FC_CL_LE    = FC_90CL_0 / normalization_LE[cut_zLE]

    plot_figure(cut_nz, flux_E3, flux_E3_lower, flux_E3_upper, cut_z, FC_CL_E3, bin_energy18, 
                cut_nzi, flux_E3_i, flux_E3_lower_i, flux_E3_upper_i, bin_energy18_i,
                bin_energy18_LE, cut_nzLE, flux_LE, flux_lower_LE, flux_upper_LE, cut_zLE, FC_CL_LE, 
                )