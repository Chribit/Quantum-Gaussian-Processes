import numpy as np
import pandas as pd
from plot import plot_ttc




classic_data = pd.read_csv("data/old_conversions_classic.csv")

unique_days = classic_data.days.unique()
classic_averages = []

for days in unique_days:
    
    data_subset = classic_data.loc[classic_data["days"] == days]
    cycles = data_subset["cycles"]
    average_cycle_count = np.average(cycles)
    classic_averages.append(average_cycle_count)
    
plot_ttc(unique_days, classic_averages, True, "evaluation/time_to_conversion/ttc")

