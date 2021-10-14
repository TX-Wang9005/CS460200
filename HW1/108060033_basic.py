#!/usr/bin/env python
# coding: utf-8

# import packages
# Note: You cannot import any other packages!
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import random


# Global attributes
# Do not change anything here except TODO 1
StudentID = '108060033'  # TODO 1 : Fill your student ID here
# Please name your input csv file as 'input.csv'
input_dataroot = 'sample_input.csv'
# Output file will be named as '[StudentID]_basic_prediction.csv'
output_dataroot = StudentID + '_basic_prediction.csv'
input_datalist = []  # Initial datalist, saved as numpy array
output_datalist = []  # Your prediction, should be 20 * 2 matrix and saved as numpy array
training_data = []
validation_data = []
w = 0


# Read input csv to datalist
with open(input_dataroot, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))

# From TODO 2 to TODO 6, you can declare your own input parameters, local attributes and return parameters


def SplitData():
    # TODO 2: Split data, 2021/10/15 ~ 2021/11/11 for testing data, and the other for training data and validation data
    for i, sublist in enumerate(input_datalist):
        if "0" in sublist:
            break
    # print(i)
    train, valid = np.split(input_datalist, [i])
    return train, valid


def PreprocessData():
    # TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    x_datalist = []
    y_datalist = []
    for i in range(0, len(training_data)):
        x_datalist.append(training_data[i][1])
        y_datalist.append(training_data[i][2])
    return x_datalist, y_datalist


def OLS(x_datalist, y_datalist):
    w = np.linalg.inv(x_datalist.T @ x_datalist) @ x_datalist.T @ y_datalist
    return w


def MakePrediction(w, x):
    # TODO 6: Make prediction of testing data
    y = int(w * x)
    return str(y)


training_data, validation_data = SplitData()
x_datalist, y_datalist = PreprocessData()
x_datalist = list(map(int, x_datalist))
y_datalist = list(map(int, y_datalist))
x_array = np.array(x_datalist)
y_array = np.array(y_datalist)
x_array = np.reshape(x_array, (-1, 1))
y_array = np.reshape(y_array, (-1, 1))
w = OLS(x_array, y_array)
for idx_p, d in enumerate(validation_data):
    x_p = d[1]
    output_datalist.append(
        [validation_data[idx_p][0], MakePrediction(w, int(x_p))])

    # Write prediction to output csv
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for row in output_datalist:
        writer.writerow(row)
