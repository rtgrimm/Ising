from itertools import cycle, islice

import seaborn as sb
from matplotlib import rcParams
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def save(name):
    plt.savefig(f"figures/{name}.pdf")




def pal(name, count, skip=0):
    def iter_f():
        palette = sb.color_palette(name, count)

        for color in palette:
            yield list(color)

    return islice(cycle(iter_f()), skip, None)

def set_style(palette = "muted"):
    sb.set(context="paper", style="ticks", font_scale=3.5, font="Arial", rc={
        "lines.linewidth": 3
    })

    matplotlib.rc('axes', edgecolor='black')

    rcParams['axes.linewidth'] = 3.0

    sb.set_palette(palette)

    matplotlib.rcParams['axes.spines.left'] = True
    matplotlib.rcParams['axes.spines.bottom'] = True
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False


    matplotlib.rcParams['xtick.major.size'] = 20
    matplotlib.rcParams['xtick.major.width'] = 4

    matplotlib.rcParams['xtick.minor.size'] = 10
    matplotlib.rcParams['xtick.minor.width'] = 2

    matplotlib.rcParams['ytick.major.size'] = 20
    matplotlib.rcParams['ytick.major.width'] = 4

    matplotlib.rcParams['ytick.minor.size'] = 10
    matplotlib.rcParams['ytick.minor.width'] = 2