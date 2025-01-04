import numpy
import pandas
import matplotlib.pyplot as plt


data_path = "data/raw_iris.csv"

df = pandas.read_csv(data_path)

print(df.head())