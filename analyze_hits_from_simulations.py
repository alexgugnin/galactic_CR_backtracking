import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def hit_count_vs_event_num(data, target, particle):
    """
    Create a box plot showing the distribution of hit counts for each event number.
    """
    degrees = ['1degree', '2degree', '3degree']
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    axes_flat = axes.flatten()

    # Building boxplot for every axis
    events = sorted(data['event_num'].unique()) # returns [22, 23, 30] for 3 events in Void
    for i, degree in enumerate(degrees):
        data_to_plot = [data[data['event_num'] == event][f'hit_count_{degree}'] for event in events]

        axes_flat[i].boxplot(data_to_plot, labels=events, patch_artist=True, 
                    boxprops=dict(facecolor="lightblue", color="blue"),
                    medianprops=dict(color="red"))
        axes_flat[i].set_ylim([0, 1000])

    plt.tight_layout()
    plt.suptitle(f"Distribution of Hit Count by Event Number for {particle} nuclei, target {target}", fontsize=20, y=1.02)
    plt.xlabel('Event Number')
    plt.ylabel('Hit Count')
    plt.savefig(f'paper_results/projections_and_statistics/boxplots/{target}_{particle}.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Load the simulation hits data
    particles = ['H', 'He', 'C', 'N', 'O', 'Fe']
    targets = ['grs', 'ss', 'ngc']#['sgr']

    #BOXPLOTS
    for target in tqdm(targets):
        hits_df = pd.read_csv(f"paper_results/projections_and_statistics/{target}/hit_statistics_1000.csv")
        for particle in particles:
            # Group by particle type 
            particle_info = hits_df[hits_df['particle'] == particle]  
            #calling boxplot func 
            hit_count_vs_event_num(particle_info, target, particle)