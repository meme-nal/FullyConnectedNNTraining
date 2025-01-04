import numpy
from sklearn.preprocessing import OneHotEncoder
import pandas
import matplotlib.pyplot as plt

TRAIN_TEST_RATIO = 0.85
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


df_train = df.iloc[:int(df.shape[0]*TRAIN_TEST_RATIO)]
df_test = df.iloc[int(df.shape[0]*TRAIN_TEST_RATIO):]

df_train.to_csv("data/train_data.csv", index=False)
df_test.to_csv("data/test_data.csv", index=False)