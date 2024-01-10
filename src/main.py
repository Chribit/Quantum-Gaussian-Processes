from data import generate_data
from plot import plot_dataset




data = generate_data(365, 100, "stein")
plot_dataset(data, "Generated Dataset (Seed: 'stein')", True, "dataset_stein")