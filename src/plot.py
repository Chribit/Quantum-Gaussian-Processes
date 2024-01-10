import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd




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

def plot_dataset (data, title):
    
    colours = __plot_init([12, 6])
    
    fig, ax = plt.subplots()
    sns.lineplot(x = "time", y = "value", data = data, color = colours[0], ax = ax) 
    ax.set(title = title, xlabel = "day", ylabel = "value")
    
    plt.show()
    
def plot_samples (x_train, samples, title, data = pd.DataFrame()):
    
    colours = __plot_init([12, 6])
    
    fig, ax = plt.subplots()
    
    for sample_index in range(0, len(samples[0])):
        sns.lineplot(x = x_train[..., 0], y = samples[:, sample_index], color = colours[1], alpha = 0.5, ax = ax)
    
    if not data.empty:
        sns.lineplot(x = "time", y = "value", data = data, color = colours[0], ax = ax)
        
    ax.set(title = title, xlabel = "day", ylabel = "value")
    
    plt.show()

def plot_prediction (predictions, training_cutoff, title):
    
    colours = __plot_init([12, 6])
    
    fig, ax = plt.subplots()
    
    ax.fill_between(
        x = predictions["time"],
        y1 = predictions["prediction_lower"], 
        y2 = predictions["prediction_upper"], 
        color = colours[2], 
        alpha = 0.25
    )
    sns.lineplot(x = "time", y = "prediction_mean", data = predictions, color = colours[2], ax = ax)
    sns.lineplot(x = "time", y = "value", data = predictions[: training_cutoff + 1], color = colours[0], ax = ax)
    
    ax.axvline(training_cutoff, color = colours[3], linestyle='--')

    ax.set(title = title, xlabel = "day", ylabel = "value")
    
    plt.show()

def plot_window_comparison (data, title, save = False):
    
    colours = __plot_init([12, 6], save)
    
    fig, ax = plt.subplots()
    
    sns.barplot(x = "window_size", y = "average_difference", data = data, color = colours[4])
    plt.xlabel("")
    plt.ylabel("")
    
    fig.text(0.525, 0.02, 'Window Size', ha='center')
    fig.text(0.025, 0.5, 'Average Deviation', va='center', rotation='vertical')
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.9, bottom = 0.15)
    
    fig.suptitle(title)
    
    if save:
        fig.savefig('window_size_comparison.png', transparent = True)
    else:
        plt.show()

def plot_model_samples (data, samples, title, y_limits = False, plot_data = True):
    
    colours = __plot_init([12, 6])
    
    x_data = data["time"].to_numpy()
    y_data = data["value"].to_numpy()
    
    for index, sample in enumerate(samples):
        colour = colours[(index + 1) % len(colours)]
        plt.plot(x_data, sample, color = colour)
    
    if (plot_data):
        plt.plot(x_data, y_data, color = colours[0], linewidth = 2.0)
    
    plt.title(title)
    plt.xlabel("day")
    plt.ylabel("value")
    if (y_limits != False):
        plt.ylim(y_limits)
        
    plt.show()
    
def plot_model_samples_dual (data, prior_samples, posterior_samples, title, y_limits = False, save = False):
    
    colours = __plot_init([14, 6], save)
    
    x_data = data["time"].to_numpy()
    y_data = data["value"].to_numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    for index, sample in enumerate(prior_samples):
        
        colour = colours[(index + 1) % len(colours)]
        ax1.plot(x_data, sample, color = colour, alpha = 1.0)
        
    for index, sample in enumerate(posterior_samples):
        
        colour = colours[(index + 1) % len(colours)]
        ax2.plot(x_data, sample, color = colour, alpha = 1.0)
    
    fig.text(0.5, 0.02, 'Time', ha='center')
    fig.text(0.025, 0.5, 'Value', va='center', rotation='vertical')
    plt.subplots_adjust(left = 0.08, right = 0.92, top = 0.9, bottom = 0.15)
    
    ax1.set_title("Prior Samples")
    ax2.set_title("Posterior Samples")
    
    fig.suptitle(title)
    
    if (y_limits != False):
        ax1.set_ylim(y_limits)
        ax2.set_ylim(y_limits)
    
    if save:
        fig.savefig('dual_samples_plot.png', transparent = True)
    else:
        plt.show()

def plot_model_performance (training_data, testing_data, predictions_x, predictions_y, sigmas, title, y_limits = False, save = False):
    
    colours = __plot_init([14, 6], save)
    
    x_train = training_data["time"].to_numpy()
    y_train = training_data["value"].to_numpy()
    x_test  = testing_data["time"].to_numpy()
    y_test  = testing_data["value"].to_numpy()
    x_pred = predictions_x
    y_pred = predictions_y
    
    x_test_line = np.append(np.array(x_train[-1]), x_test)
    y_test_line = np.append(np.array(y_train[-1]), y_test)
    
    fig, ax = plt.subplots(1, 1)
    
    fig.text(0.525, 0.02, 'Time', ha='center')
    fig.text(0.025, 0.5, 'Value', va='center', rotation='vertical')
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.9, bottom = 0.15)

    plt.suptitle(title)
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
        fig.savefig('classical_model_performance_plot.png', transparent = True)
    else:
        plt.show()

def plot_covariance_matrix (covariance_matrix, title, save = False):
        
    fig, ax = plt.subplots(1, 1)
    
    plt.suptitle(title)
    ax.matshow(covariance_matrix, cmap = "Wistia")
    
    for i in range(len(covariance_matrix[0])):
        for j in range(len(covariance_matrix[0])):
            cell_value = "%.2f" % covariance_matrix[j, i]
            ax.text(i, j, str(cell_value), va = 'center', ha = 'center', fontsize = 'x-small', color = 'black')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    if save:
        fig.savefig(title + '.png', transparent = True)
    else:
        plt.show()
