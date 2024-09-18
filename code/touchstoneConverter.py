import os.path

from skrf import Frequency, Network

def get_time_recorded_from_touchstone(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        if len(lines) >= 2:
            line = lines[1].strip()
            time_recorded = line.split(' ', maxsplit=1)[1]
            return time_recorded
        else:
             pass


path = r'D:\James\documents\OneDrive - University of Glasgow\Glasgow\Year 4\Project\Pico_VNA_Project\results\touchstones\17_09_patent_exp\2024_09_17_18_09_29_401pts_A_test_1-0.s4p'
fname = os.path.basename(path)

split_fname = fname.split('_')
date_created = ('_').join(split_fname[:6])


label = split_fname[7]

test_network = Network(path)
frequency_array = test_network.f #.tolist()

