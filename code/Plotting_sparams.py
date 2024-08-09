# Plots single freq vs time, single s-param, phase / mag
from VNA_data import VnaData
from VNA_utils import ghz_to_hz
from VNA_enums import SParam, DataFrameCols

data = VnaData(r"C:\Users\mww19a\PycharmProjects\Pico_VNA_Project\results\data\single_Test_dipole1_xx\single_Test_dipole1_xx_2024_08_09_14_52_50_S11_S21_S12_S22_20_secs.csv")
data.single_freq_plotter(ghz_to_hz(1.2), plot_s_param=SParam.S11, data_frame_column_to_plot=DataFrameCols.PHASE)

