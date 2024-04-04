import numpy as np
import pandas as pd
from plot import plot_distances, plot_trends




classic_data = pd.read_csv("data/accuracy_classic.csv")
quantum_data = pd.read_csv("data/accuracy_quantum.csv")

predicted_days = np.arange(1, 11, 1)

plot_distances(predicted_days, classic_data["distance"], quantum_data["distance"], True, "evaluation/accuracy/average_distance")
plot_trends(predicted_days, classic_data["trend"], quantum_data["trend"], True, "evaluation/accuracy/average_trends")
