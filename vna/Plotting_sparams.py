# Plots single freq vs time, single s-param, phase / mag
from VNA_data import VnaData
from VNA_utils import ghz_to_hz
from VNA_enums import SParam, DataFrameCols

data = VnaData(r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\results\data\single_Stretch_Test\single_Stretch_Test_2024_08_09_15_52_14_S11_S21_S12_S22_1200_secs.csv")
data.single_freq_plotter(ghz_to_hz(1.0), plot_s_param=SParam.S11, data_frame_column_to_plot=DataFrameCols.MAGNITUDE)

