#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Pico Technology Ltd. See LICENSE file for terms.
#
import datetime

import win32com.client
import numpy as np
import matplotlib.pyplot as plt

picoVNACOMObj = win32com.client.Dispatch("PicoControl2.PicoVNA_2")

print("Connecting VNA")
findVNA = picoVNACOMObj.FND()
print('VNA ' + str(findVNA) + ' Loaded')

print("Load Calibration")
ans=picoVNACOMObj.LoadCal("C:\\Users\\js637s\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\VNA\\500MHz_3GHz_MiniCirc_P1Short_P2Long_m3dBm_Lab103_Mar23_200MHz_6GHz_.cal");
print("Result " + str(ans))

for i in range(10):

    print("Making Measurement")
    picoVNACOMObj.Measure('ALL')
    print(f"time is {datetime.datetime.now()}")
    raw = picoVNACOMObj.GetData("S11","logmag",0)
    splitdata = raw.split(',')
    converteddata = np.array(splitdata)
    converteddata = converteddata.astype(np.float)
    frequency = converteddata[: : 2]
    data = converteddata[1 : : 2]

    plt.plot(frequency, data)
    plt.ylabel("S11 LogMag")
    plt.xlabel("Frequency")
    plt.show()

a = picoVNACOMObj.CloseVNA()

print("VNA Closed")