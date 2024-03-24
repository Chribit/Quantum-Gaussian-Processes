from data import generate_data
from plot import plot_aapl, plot_dataset

days = 365
seeds = [
    "adu38d1ed0k21kand",
    "7878uhe8450jffw3t5g5ggsvns3",
    "vi5u30ß021eijdaw3",
    "ß0ojn5btg7rfw8du3jhfnrwa",
    "90ßpüö.,l3mkrnjfugeeff",
    "7589rgjivmp3re2d3f49t+raük2jhrqf",
    "567gt8fhivnwvs0rikpslrt4g3ßa22",
    "47tg89t0bzoidjvsnjabzegu2d",
    "58z9gujsnfiyw",
    "vn96e5t40wr3aijis"
]

for index, seed in enumerate(seeds):
    data = generate_data(days, 69, seed)
    plot_dataset(data, "Generated Dataset (Seed: '" + seed + "')", True, "generation/dataset_" + str(index))

plot_aapl("generation/apple")