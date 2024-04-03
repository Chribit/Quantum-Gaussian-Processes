import numpy as np
import pandas as pd
from plot import plot_ttc




classic_data = pd.read_csv("data/conversions_classic.csv")
quantum_data = pd.read_csv("data/conversions_quantum.csv")

unique_days = classic_data.days.unique()
classic_averages = []
quantum_averages = []

for days in unique_days:
    
    data_subset = classic_data.loc[classic_data["days"] == days]
    cycles = data_subset["cycles"]
    average_cycle_count = np.average(cycles)
    classic_averages.append(average_cycle_count)
    
    q_data_subset = quantum_data.loc[quantum_data["days"] == days]
    q_cycles = q_data_subset["cycles"]
    q_average_cycle_count = np.average(q_cycles)
    quantum_averages.append(q_average_cycle_count)
    
plot_ttc(unique_days, classic_averages, quantum_averages, True, "evaluation/time_to_conversion/ttc")

