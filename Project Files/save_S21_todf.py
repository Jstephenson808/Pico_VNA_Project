#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Pico Technology Ltd. See LICENSE file for terms.
#
#

import win32com.client
import numpy as np
import pandas as pd
from datetime import datetime

N_RUNS = 10
MEASURE = 'S21'
picoVNACOMObj = win32com.client.gencache.EnsureDispatch("PicoControl2.PicoVNA_2")
df = pd.DataFrame(columns=['Time', 'Frequency', 'Magnitude (dB)'])

print("Connecting VNA")
findVNA = picoVNACOMObj.FND()
print('VNA ' + str(findVNA) + ' Loaded')

print("Load Calibration")
ans = picoVNACOMObj.LoadCal("C:\\Users\\js637s\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\VNA\\500MHz_3GHz_MiniCirc_P1Short_P2Long_m3dBm_Lab103_Mar23_200MHz_6GHz_.cal")
print("Result " + str(ans))
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for index in range(N_RUNS):
    picoVNACOMObj.Measure(MEASURE)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    raw = picoVNACOMObj.GetData(MEASURE, "logmag", 0)
    split_data = raw.split(',')
    converted_data = np.array(split_data)
    converted_data = converted_data.astype(np.float)
    frequency = converted_data[:: 2]
    data = converted_data[1:: 2]

    df = df.append({'Time': current_time, 'Frequency': frequency, 'Magnitude (dB)': data}, ignore_index=True)

    if index % 10 == 0:
        print(f"Saving S21 LogMag Data Index is {index}")
        df.to_csv(f'{MEASURE}_{N_RUNS}_Runs_{start_time}.csv', index=False)

df.to_csv(f'{MEASURE}_{N_RUNS}_Runs_{start_time}.csv', index=False)

a = picoVNACOMObj.CloseVNA()
print("VNA Closed")


