"""
Some useful functions for all the project

"""

import os
import matplotlib.pyplot as plt


def plot(X,y,xlabel,ylabel,title):
    plt.plot(X,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    title_fig = "_".join(title.split(" "))+".png"
    plt.grid()
    directory = "plots"
    filepath = os.path.join(directory, title_fig)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the figure to the specified path
    plt.savefig(filepath)
    plt.show()
    plt.close() # Close the plot to release memory