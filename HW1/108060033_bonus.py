#!/usr/bin/env python
# coding: utf-8

from statsmodels.tsa.ar_model import AutoReg
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import random


# Global attributes
# Do not change anything here except TODO 1
StudentID = '108060033'  # TODO 1 : Fill your student ID here
# Please name your input csv file as 'input.csv'
input_dataroot = 'bonus_input.csv'
# Output file will be named as '[StudentID]_basic_prediction.csv'
output_dataroot = StudentID + '_bonus_prediction.csv'
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
    train, valid = np.split(input_datalist, [i])
    return train, valid


def PreprocessData():
    # TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    x_datalist = []
    for i in range(0, len(training_data)):
        x_datalist.append(training_data[i][1])
    return x_datalist


training_data, validation_data = SplitData()
x_datalist = PreprocessData()
ar_model = AutoReg(x_datalist, lags=20).fit()           # Use ar_model
pred = ar_model.predict(start=len(training_data), end=(   # Predict
    len(input_datalist)-1), dynamic=False)
for idx_p, d in enumerate(validation_data):
    output_datalist.append(
        [validation_data[idx_p][0], round(pred[idx_p])])
# Write prediction to output csv
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for row in output_datalist:
        writer.writerow(row)
