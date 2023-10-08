#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Pico Technology Ltd. See LICENSE file for terms.
#
#
import os

import win32com.client
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from matplotlib import pyplot as plt

#csv_path = 'C:\\Users\\js637s\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\Code\\Pico VNA\\picosdk-picovna-python-examples\\Project Files\\S11_10_Runs_2023-09-05_13-53-57.csv'
save_directory = 'C:\\Users\\js637s\\OneDrive\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\Experiment Files\\VNA_Data'


def MHz_to_VNA_Fq(mhz_value:int):
    return mhz_value * 1000000

def string_to_datetime(string):
    return datetime.strptime(string, "%Y_%m_%d_%H_%M_%S.%f")

def calculate_dt(data_frame):
    return data_frame['Time'].diff().dt.total_seconds()

def extract_frequency_indexes(target_frequency:int, data_frame:pd.DataFrame)->list:
    frequency_index = []
    for index, row in data_frame.iterrows():
        if target_frequency in row['Frequency']:
            frequency_index.append(row['Frequency'].index(target_frequency))
        else:
            frequency_index.append(-1)
    return frequency_index

def find_min_fq(data_frame):
    min_index = []
    for df_index, row in data_frame.iterrows():
        row['Magnitude (dB)'].min()

def extract_magnitude(frequency_index:list, data_frame):
    magnitudes = []
    # each row has an index, this is used as the index in the
    # fq_index list, this index is the index of the
    # magnitude we want
    for df_index, row in data_frame.iterrows():
        magnitude_index = frequency_index[df_index]
        magnitudes.append(row['Magnitude (dB)'][magnitude_index])
    return magnitudes

def zero_ref_time(data_frame:pd.DataFrame):
    start_time = data_frame['Time'][0]
    data_frame['Time'] = data_frame['Time'].apply(lambda x: (x - start_time).total_seconds())


# for a given frequency plot a graph which has the magnitude over time
def plot_frequency(target_frequency:int, data_frame:pd.DataFrame, folder):
    # how can I get the time out?
    # data_frame['dt'] = calculate_dt(data_frame)
    # data_frame['dt'][0] = 0.0

    frequency_index = extract_frequency_indexes(target_frequency, data_frame)
    magnitudes = extract_magnitude(frequency_index, data_frame)
    target_frequency_GHz = target_frequency / 1000000000

    df_dict = {'time': data_frame['Time'], 'magnitude (dB)': magnitudes}
    df_to_save = pd.DataFrame(data=df_dict)
    try:
        df_to_save.to_csv(f'C:\\Users\\js637s\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\Experiment Files\\VNA_Data\\Frequency_Data\\{folder}\\{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}-{target_frequency_GHz}_GHz.csv')
    except FileNotFoundError as e:
        os.mkdir(f'C:\\Users\\js637s\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\Experiment Files\\VNA_Data\\Frequency_Data\\{folder}')
        df_to_save.to_csv(f'C:\\Users\\js637s\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\Experiment Files\\VNA_Data\\Frequency_Data\\{folder}\\{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}-{target_frequency_GHz}_GHz.csv')


    fig, ax = plt.subplots()
    ax.plot(data_frame['Time'], magnitudes)
    ax.set_ylabel("S21 Mag")
    ax.set_xlabel("Time")
    plt.title(f'S21 Over Time at {target_frequency_GHz} GHz')

    try:
        plt.savefig(os.getcwd() + '\\' + folder + f'\\{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}-{target_frequency_GHz}_GHz.png')
    except FileNotFoundError as e:
        os.mkdir(os.getcwd() + '\\' + folder)
        plt.savefig(os.getcwd() + '\\' + folder + f'\\{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}-{target_frequency_GHz}_GHz.png')

    plt.show()
    return frequency_index


def string_to_list(string):
    if string.startswith('[') and string.endswith(']'):
        return [float(i) for i in eval(string)]
    else:
        return string

def graph(df_row):
    time = df_row['Time']
    frequency_list = df_row['Frequency']
    magnitude_list = df_row['Magnitude (dB)']

    fig, ax = plt.subplots()
    ax.plot(frequency_list, magnitude_list)
    ax.set_ylabel("S11 LogMag")
    ax.set_xlabel("Frequency")
    plt.show()

def measure_from_vna(measureForMins):

    MEASURE = 'All'
    picoVNACOMObj = win32com.client.gencache.EnsureDispatch("PicoControl2.PicoVNA_2")
    CALIBRATION_PATH = "C:\\Users\\js637s\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\VNA\\800MHz_1GHz_201Points_MiniCirc_P1Short_P2Long_m3dBm_Lab103_Mar23_200MHz_6GHz_.cal"

    df = pd.DataFrame(columns=['Time', 'Measurement', 'Frequency', 'Magnitude', 'Phase'])
    # Define a custom function to convert strings back to lists
    measurement = ["S11", "S21", "S12", "S22"]

    print("Connecting VNA")
    findVNA = picoVNACOMObj.FND()
    print('VNA ' + str(findVNA) + ' Loaded')

    print("Load Calibration")
    ans = picoVNACOMObj.LoadCal(CALIBRATION_PATH)
    print("Result " + str(ans))
    start_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = datetime.now()
    run_time = timedelta(minutes=measureForMins)
    finish_time = start_time + run_time
    index = 0
    fileName = f'{MEASURE}_{measureForMins}_mins_{start_time_string}.csv'

    print(f'Starting, will record for {run_time} (until {finish_time})')
    while datetime.now() < finish_time:
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
        picoVNACOMObj.Measure(MEASURE)
        for measure in measurement:
            raw = picoVNACOMObj.GetData(measure, "logmag", 0)
            split_data = raw.split(',')
            # converted_data = np.array(split_data)
            # converted_data = converted_data.astype(np.float)
            frequency = split_data[::2]
            logMagData = split_data[1::2]

            raw = picoVNACOMObj.GetData(measure, "phase", 0)
            split_data = raw.split(',')
            # converted_data = np.array(split_data)
            # converted_data = converted_data.astype(np.float)
            phase = split_data[::2]

            data = {
                'Time' : [current_time for _ in range(len(frequency))],
                'Measurement' : [measure for _ in range(len(frequency))],
                'Frequency' : frequency,
                'Magnitude' : logMagData,
                'Phase' : phase
                }

            tempDataFrame = pd.DataFrame(data)

            df = df.append(tempDataFrame)

        if index % 10 == 0:
            print(f"Saving {MEASURE} Data Index is {index} running for another {(finish_time - datetime.now())}")
            df.to_csv(fileName, index=False)
        index += 1

    df.to_csv(fileName, index=False)

    a = picoVNACOMObj.CloseVNA()
    print("VNA Closed")
    return fileName

csv_path = measure_from_vna(5)
#csv_path = 'C:\\Users\\js637s\\OneDrive - University of Glasgow\\Glasgow\\Summer Project\\Code\\Pico VNA\\picosdk-picovna-python-examples\\Project Files\\S21_200_Runs_2023-09-05_17-22-26.csv'
# read out df and plot (but what?)
csv_df = pd.read_csv(csv_path)
# Apply the custom function to the 'frequency' and 'data' columns

csv_df['Time'] = csv_df['Time'].apply(string_to_datetime)
csv_df['Frequency'] = csv_df['Frequency'].apply(string_to_list)
csv_df['Magnitude (dB)'] = csv_df['Magnitude (dB)'].apply(string_to_list)

# for index, row in csv_df.iterrows():
#     if index == 100:
#         graph(row)
#fq's are in the format

min_index = csv_df['Magnitude (dB)'][5].index(min(csv_df['Magnitude (dB)'][5]))


target_frequency = 800
zero_ref_time(csv_df)
date_processed = datetime.now().strftime('%Y_%m_%d')
while target_frequency < 1000:
    plot_frequency(MHz_to_VNA_Fq(target_frequency), csv_df, folder=date_processed)
    target_frequency = target_frequency + 10

#get index of min
#csv_df['Magnitude (dB)'][0].index(min(csv_df['Magnitude (dB)'][0]))