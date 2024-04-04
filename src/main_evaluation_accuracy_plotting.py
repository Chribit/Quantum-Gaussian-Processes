import numpy as np
import pandas as pd
from plot import plot_distances, plot_trends




classic_data = pd.read_csv("data/accuracy_classic.csv")
quantum_data = pd.read_csv("data/accuracy_quantum.csv")

predicted_days = np.arange(1, 11, 1)

classic_accuracies = classic_data["distance"]
quantum_accuracies = quantum_data["distance"]

plot_distances(predicted_days, classic_accuracies, True, "evaluation/accuracy/average_distance_classic")
plot_distances(predicted_days, quantum_accuracies, True, "evaluation/accuracy/average_distance_quantum")

classic_trends = classic_data["trend"]
quantum_trends = quantum_data["trend"]

plot_trends(predicted_days, classic_trends, True, "evaluation/accuracy/average_trends_classic")
plot_trends(predicted_days, quantum_trends, True, "evaluation/accuracy/average_trends_quantum")
