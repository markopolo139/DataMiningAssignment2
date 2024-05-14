import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(path):
    return pd.read_csv(path)


def drop_columns(data, column):
    data.drop(columns=column, inplace=True)
    

def drop_na(data):
    data.dropna(inplace = True)

def drop_duplicate(data):
    data.drop_duplicates(inplace = True)


def halo():
    print ("hi")

def merge_data(data1, data2, on, how):
    return pd.merge(data1, data2, on = on, how = how)


def calculate_average(data, groupby_column, average_column):
    average = data.groupby(groupby_column)[average_column].mean().reset_index()
    return average


def copy_data(data):
    return copy.deepcopy(data)

def transform_attribute_to_multiple(data, attribute,splitted_by):
    list = []
    for index, row in data.iterrows():
        sublist = row[attribute].split(splitted_by)
        for item in sublist:
            list.append(item)
    
    for item in list:
        data[item] = 0
    
    for index, row in data.iterrows():
        sublist = row[attribute].split(splitted_by)
        for item in sublist:
            data.at[index, item] = 1
    
    return data


def transform_strings_to_numbers(data, attribute):
    
    list = []

    for index,row in data.iterrows():
        if row[attribute] not in list:
            list.append(row[attribute])

    for i in range(len(list)):
        data.loc[data[attribute] == list[i], attribute] = i+1
    
    return data


def normalize_data(data, columns):
    normalize_data = copy_data(data)
    scaler = MinMaxScaler()

    normalize_data[columns] = scaler.fit_transform(normalize_data[columns])
    return normalize_data


def standardize_data(data, columns):
    standardize_data = copy_data(data)
    scaler = StandardScaler()

    standardize_data[columns] = scaler.fit_transform(standardize_data[columns])
    return standardize_data


def split_data(data):
     return train_test_split(data, test_size=0.2, random_state=42)