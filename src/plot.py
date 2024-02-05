import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



image_folder = "images/"




def __plot_init (figsize, save):
    
    if save:
        sns.set_style(
            style = "darkgrid", 
            rc = {
                "figure.facecolor": "black",
                "axes.facecolor": "0.9",
                "text.color": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "grid.color": "0.5",
                "patch.edgecolor": "none"
            }
        )
        sns.set_context(
            context = "talk",
            font_scale = 0.8
        )
    else:
        sns.set_style(
            style = "darkgrid", 
            rc = {
                "axes.facecolor": "0.9",
                "grid.color": "0.8"
            }
        )
        sns.set_context(
            context = "paper",
            font_scale = 0.8
        )
    
    sns.set_palette(palette = "deep")
    
    colours = sns.color_palette(palette = "colorblind")
    if save:
        colours = sns.color_palette(palette = "bright")
    
    plt.rcParams["figure.figsize"] = figsize
    
    if save:
        plt.rcParams["figure.dpi"] = 250
    else:
        plt.rcParams["figure.dpi"] = 100
    
    return colours

def plot_dataset (data, title, save = False, filename = "new_plot"):

    colours = __plot_init([12, 6], save)
    
    fig, ax = plt.subplots()
    
    sns.lineplot(x = "time", y = "value", data = data, color = colours[4])
    plt.xlabel("")
    plt.ylabel("")
    
    fig.text(0.525, 0.02, "Day", ha = "center")
    fig.text(0.025, 0.5, "Value", va = "center", rotation = "vertical")
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.9, bottom = 0.15)
    
    fig.suptitle(title)
    
    if save:
        fig.savefig(image_folder + filename + ".png", transparent = True)
    else:
        plt.show()
        
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
