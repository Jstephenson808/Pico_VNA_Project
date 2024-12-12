import os

from matplotlib import pyplot as plt

from graphs import plot_magnitude_from_touchstone, subplot_params
from skrf.io import touchstone

touchstone_folder_path = r"C:\Users\2573758S\OneDrive - University of Glasgow\PhD\Experiments\Glove Gesture Experiment\Touchstones\Pre Experiment Touchstones"

touchstone_folder = os.listdir(touchstone_folder_path)

for touchstone_fname in touchstone_folder:
    touchstone_path = os.path.join(touchstone_folder_path, touchstone_fname)
    plot_magnitude_from_touchstone(
        touchstone_path,
        f'{os.path.basename(touchstone_fname).split(".")[0].replace("_", " ").title()}',
    )
    subplot_params(touchstone.hfss_touchstone_2_network(touchstone_path))

plt.show()
