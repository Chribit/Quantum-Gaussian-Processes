import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



image_folder = "images/"




def __plot_init (figsize, save):
    

    sns.set_style(
        style = "darkgrid", 
        rc = {
            "axes.facecolor": "0.9",
            "grid.color": "0.8"
        }
    )
    sns.set_context(
        context = "paper",
        font_scale = 2
    )
    sns.set_palette(palette = "deep")
    
    colours = sns.color_palette(palette = "colorblind")
    
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = 100
    
    return colours

def plot_dataset (data, title, save = False, filename = "new_plot"):

    colours = __plot_init([10, 6], save)
    
    fig, ax = plt.subplots()
    
    sns.lineplot(x = "time", y = "value", data = data, color = colours[2])
    plt.xlabel("")
    plt.ylabel("")
    
    # fig.text(0.525, 0.02, "Day", ha = "center")
    # fig.text(0.01, 0.5, "Value", va = "center", rotation = "vertical")
    # plt.subplots_adjust(left = 0.1, right = 0.98, top = 0.98, bottom = 0.12)
    plt.subplots_adjust(left = 0.07, right = 0.98, top = 0.98, bottom = 0.07)
    
    # fig.suptitle(title)
    
    if save:
        fig.savefig(image_folder + filename + ".png")
    else:
        plt.show()
   
def plot_aapl ():
    
    data = pd.read_csv("data/aapl_full_data.csv")
    data.drop(columns = ['date', 'open', 'high', 'low', 'volume'], inplace = True)
    data.rename(columns = {'close': 'value'}, inplace = True)
    data = data.tail(365)
    data.insert(0, "time", np.arange(365))
    data.index = pd.RangeIndex(0, len(data))
    
    plot_dataset(data, "Apple Stock Prices (last 365 days)", True, "apple")
        
def plot_prediction (title, x_train, y_train, x_test, y_test, x_pred, y_pred, sigmas, x_limits = False, y_limits = False, save = False, filename = "new_plot"):
    
    colours = __plot_init([12, 6], save)
    
    x_test_line = np.append(np.array(x_train[-1]), x_test)
    y_test_line = np.append(np.array(y_train[-1]), y_test)
    
    fig, ax = plt.subplots(1, 1)
    
    fig.text(0.525, 0.02, "Time", ha = "center")
    fig.text(0.025, 0.5, "Value", va = "center", rotation = "vertical")
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.9, bottom = 0.15)

    plt.suptitle(title)
    
    if (x_limits != False):
        plt.xlim(x_limits)
        
    if (y_limits != False):
        plt.ylim(y_limits)
    
    plt.fill_between(
        x_pred,
        np.subtract(y_pred, sigmas),
        np.add(y_pred, sigmas),
        color = colours[2],
        alpha = 0.33
    )
    ax.plot(x_pred, y_pred, color = colours[2], label = "predictions")
    ax.plot(x_train, y_train, color = colours[-1], alpha = 0.33)
    ax.plot(x_test_line, y_test_line, color = colours[1], alpha = 0.33)
    ax.plot(x_train, y_train, 'o', color = colours[-1], label = "training data")
    ax.plot(x_test, y_test, 'o', color = colours[1], label = "testing data")
    
    legend = plt.legend(loc = "lower left", frameon = False)
    frame = legend.get_frame()
    frame.set_facecolor("#000000")
    
    if save:
        fig.savefig(image_folder + filename + ".png", transparent = True)
    else:
        plt.show()

def plot_fitness (title, x_train, y_train, fitness_x, fitness_y, x_pred, y_pred, sigmas, x_limits = False, y_limits = False, save = False, filename = "new_plot"):
    
    colours = __plot_init([12, 6], save)
    
    fig, ax = plt.subplots(1, 1)
    
    fig.text(0.525, 0.02, "Time", ha = "center")
    fig.text(0.025, 0.5, "Value", va = "center", rotation = "vertical")
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.9, bottom = 0.15)

    plt.suptitle(title)
    
    if (x_limits != False):
        plt.xlim(x_limits)
        
    if (y_limits != False):
        plt.ylim(y_limits)
        
    prediction_colour = colours[-1]
    target_colour = colours[3]
    
    plt.fill_between(
        x_pred,
        np.subtract(y_pred, sigmas),
        np.add(y_pred, sigmas),
        color = prediction_colour,
        alpha = 0.33
    )
    ax.plot(x_pred, y_pred, color = prediction_colour, label = "predictions")
    ax.plot(x_train, y_train, 'o', color = target_colour, label = "training data")
    ax.plot(fitness_x, fitness_y, color = target_colour, alpha = 0.75, label = "fitness target")
    
    legend = plt.legend(loc = "lower left", frameon = False)
    frame = legend.get_frame()
    frame.set_facecolor("#000000")
    
    if save:
        fig.savefig(image_folder + filename + ".png", transparent = True)
    else:
        plt.show()
        
def plot_circuit (title, quantum_model, save = False, filename = "new_circuit_plot"):
    
    if not quantum_model.is_quantum:
        return
    
    __plot_init([12, 6], save)
    
    fig = quantum_model.plot_quantum_circuit()
    
    plt.suptitle(title)
    
    if save:
        fig.savefig(image_folder + filename + ".png", transparent = True)
    else:
        plt.show()

def plot_evolution (title, fitnesses, save = False, filename = "new_evolution_plot"):

    colours = __plot_init([12, 6], save)
    
    fig, ax = plt.subplots(1, 1)
    
    fig.text(0.525, 0.02, "Generations", ha = "center")
    fig.text(0.025, 0.5, "Fitness", va = "center", rotation = "vertical")
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.9, bottom = 0.15)

    plt.suptitle(title)
    ax.set_ylim([-0.05, 1.05])
    
    averages = []
    weights = np.arange(len(fitnesses[0]), 0, -1)
    
    for generation in range(len(fitnesses)):
        averages.append(np.average(fitnesses[generation], 0, weights))
        
    generations_x = np.arange(1, len(averages) + 1)
    
    ax.plot(np.repeat(generations_x, len(fitnesses[0])), fitnesses.flatten(), 'o', color = colours[1], alpha = 0.2, label = "individual fitnesses")
    ax.plot(generations_x, averages, color = colours[-1], label = "weighted average fitness")
    
    legend = plt.legend(loc = "lower right", frameon = False)
    frame = legend.get_frame()
    frame.set_facecolor("#000000")
    
    if save:
        fig.savefig(image_folder + filename + ".png", transparent = True)
    else:
        plt.show()

def plot_covariance_matrix (covariance_matrix, title, save = False, filename = "new_matrix_plot"):
        
    fig, ax = plt.subplots(1, 1)
    
    plt.suptitle(title)
    ax.matshow(covariance_matrix, cmap = "Wistia")
    
    for i in range(len(covariance_matrix[0])):
        for j in range(len(covariance_matrix[0])):
            cell_value = "%.2f" % covariance_matrix[j, i]
            ax.text(i, j, str(cell_value), va = 'center', ha = 'center', fontsize = 5.0, color = 'black')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if save:
        fig.savefig(image_folder + filename + ".png", transparent = True)
    else:
        plt.show()
