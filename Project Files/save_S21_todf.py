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

picoVNACOMObj = win32com.client.gencache.EnsureDispatch("PicoControl2.PicoVNA_2")
df = pd.DataFrame(columns=['Time', 'S21 Data'])
index = 0
print("Connecting VNA")
findVNA = picoVNACOMObj.FND()
print('VNA ' + str(findVNA) + ' Loaded')

print("Load Calibration")
ans = picoVNACOMObj.LoadCal("C:\\Users\\js637s\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\VNA\\500MHz_3GHz_MiniCirc_P1Short_P2Long_m3dBm_Lab103_Mar23_200MHz_6GHz_.cal")
print("Result " + str(ans))

for i in range(50):
    picoVNACOMObj.Measure('S21')
    current_time = datetime.now()

    print("getting S21 LogMag Data")
    raw = picoVNACOMObj.GetData("S21", "logmag", 0)
    split_data = raw.split(',')
    converted_data = np.array(split_data)
    converted_data = converted_data.astype(np.float)

    df = df.append({'Time': current_time, 'S21 Data': converted_data}, ignore_index=True)
    index += 1

    if index % 10 == 0:
        df.to_csv(f'{}.csv', index=False)
