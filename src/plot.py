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