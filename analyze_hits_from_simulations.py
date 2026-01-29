import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def hit_count_vs_event_num(data, target, particle):
    """
    Create a box plot showing the distribution of hit counts for each event number.
    """
    plt.style.use('seaborn-muted')
    degrees = ['1degree', '2degree', '3degree']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    # Building boxplot for every axis
    events = sorted(data['event_num'].unique()) # returns [22, 23, 30] for 3 events in Void
    events_labels = ["TA-LV-1", "TA-LV-2", "PA-LV"]
    colors = ["#7eb0d5", "#b2e061", "#bd7ebe"]

    for i, degree in enumerate(degrees):
        ax = axes[i]
        column_name = f'hit_count_{degree}'
        data_to_plot = [data[data['event_num'] == event][f'hit_count_{degree}'] for event in events]

        bp = ax.boxplot(data_to_plot, 
                        labels=events_labels, 
                        patch_artist=True,
                        notch=False, # Set to True if you want to see confidence intervals
                        widths=0.6,
                        showfliers=False)  # Hides outliers for clarity
        
        # Styling the boxes
        for patch in bp['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
            patch.set_edgecolor('#444444')

        # Styling medians and whiskers
        plt.setp(bp['medians'], color='red', linewidth=1.5)
        plt.setp(bp['whiskers'], color='#444444', linestyle='--')

        # DYNAMIC SCALING: Instead of fixed 1000, use data max + 10% padding
        max_val = data[column_name].max()
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.1)

        # Aesthetics
        ax.set_title(f"Aperture: {degree}", loc='left', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylabel('Hit Count', fontsize=10)

    # Label only the bottom X axis
    axes[-1].set_xlabel('Event', fontsize=12)

    # Main Title - using constrained_layout or specific y position to avoid overlap
    plt.suptitle(f"Hit Count Distribution: {particle} Nuclei (Target: {target.upper()})", 
                 fontsize=16, fontweight='bold', y=0.935)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle space

    plt.savefig(f'paper_results/projections_and_statistics/boxplots/{target}_{particle}.jpeg', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Load the simulation hits data
    particles = ['H', 'He', 'C', 'N', 'O', 'Fe']
    targets = ['grs', 'ss', 'ngc']#['sgr']#

    #BOXPLOTS
    for target in tqdm(targets):
        hits_df = pd.read_csv(f"paper_results/projections_and_statistics/{target}/hit_statistics_1000.csv")
        for particle in particles:
            # Group by particle type 
            particle_info = hits_df[hits_df['particle'] == particle]  
            #calling boxplot func 
            hit_count_vs_event_num(particle_info, target, particle)