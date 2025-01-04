import numpy
from sklearn.preprocessing import OneHotEncoder
import pandas
import matplotlib.pyplot as plt

TRAIN_TEST_RATIO = 0.8
data_path = "data/raw_iris.csv"

df = pandas.read_csv(data_path)

###############
# PREPROCESSING
###############


labels_df = pandas.DataFrame(df["variety"])
labels = ["Setosa", "Versicolor", "Virginica"]

encoder = OneHotEncoder(sparse_output=False)

encoded_data = encoder.fit_transform(labels_df)

encoded_df = pandas.DataFrame(encoded_data, columns=labels)

df = df.drop(["variety"], axis=1)

df = pandas.concat([df, encoded_df], axis=1)


###############
# TRAIN TEST SPLITTING
###############

df.to_csv("data/processed_iris.csv", index=False)