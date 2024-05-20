def makeDF(all_events_lats:list, all_events_lons:list, num_events:int):
    '''
    Helper function that converts all the Python lists of tuples generated with
    runSimulation() to one pandas DataFrame
    '''
    import numpy as np
    import pandas as pd

    NUM_OF_SIMS = num_events #31 for Auger, 28 for TA

    all_lats = list(np.split(np.array(all_events_lats, dtype='object'),NUM_OF_SIMS))
    all_lons = list(np.split(np.array(all_events_lons, dtype='object'),NUM_OF_SIMS))

    total_results = pd.DataFrame()
    for (event_lats, event_lons) in zip(all_lats, all_lons):
        events_temp = {}
        elements = event_lats[:,1:].T.squeeze()
        for i in range(len(event_lats)):
            events_temp[elements[i]] = [(x,y) for x,y in zip(event_lats[i][0], event_lons[i][0])]
        temp_df = pd.DataFrame.from_dict(events_temp)
        total_results = pd.concat([total_results, temp_df], ignore_index=True)

    return total_results
