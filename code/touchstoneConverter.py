import os.path

from skrf import Frequency, Network
from datetime import datetime

def get_time_recorded_from_touchstone(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        if len(lines) >= 2:
            line = lines[1].strip()
            time_recorded = line.split(' ', maxsplit=1)[1]
            return datetime.fromisoformat(time_recorded)
        else:
             pass


path = r'D:\James\documents\OneDrive - University of Glasgow\Glasgow\Year 4\Project\Pico_VNA_Project\results\touchstones\17_09_patent_exp\2024_09_17_18_09_29_401pts_A_test_1-0.s4p'
fname = os.path.basename(path)

split_fname = fname.split('_')
experiment_id = ('_').join(split_fname[:6])

time_touchstone_created = get_time_recorded_from_touchstone(path)


label = split_fname[7]

test_network = Network(path)
frequency_array = test_network.f #.tolist()

db_values_array = test_network.s.real
phase_values_array = test_network.s.imag

s_params = [f'S{i}{j}' for i in range(1,5) for j in range(1,5)]

def s_params_as_list(network: Network, s_param, phase_flag=False):
    i = int(s_param[1])
    j = int(s_param[2])
    assert(1<=i<=4)&(1<=j<=4)
    if phase_flag:
        return [matrix[i-1][j-1].imag for matrix in network.s]
    else:
        return [matrix[i-1][j-1].real for matrix in network.s]