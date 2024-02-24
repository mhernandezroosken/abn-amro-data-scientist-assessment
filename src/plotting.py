# general imports
import os

# conda packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np

def plot_nr_credict_applications(nr_applications, title):
    """
    Make a plot of the number of applications.
    Arguments:
        months: np.array of the months on the x-axis
        nr_applications: np.array of the nubmer of applications on the y-axis
    """
    fix, ax = plt.subplots()
    nr_applications.plot()
    plt.title(title)
    ax.set_xlabel('')
    ax.set_ylabel('Applications')
    # ax.xaxis.set_major_locator(
    #     md.MonthLocator() # show one tick per quarter
    # )
    # ax.xaxis.set_major_formatter(
    #     md.DateFormatter('%m-%Y')
    # )
    plt.show()
