import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MaxAbsScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint


scaler = MaxAbsScaler()

# load linear regression model
with open('LinearRegressionModel.pkl', 'rb') as f:
    clf = pickle.load(f)


def process_data(pandas_df, label_col_name, other_drop_columns=[]):
    """Process data and scale"""

    drop_columns = [label_col_name]  #+ other_drop_columns
    pandas_df = pandas_df.dropna()

    # only include numeric data
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    pandas_df = pandas_df.select_dtypes(include=numerics)


    y = pandas_df[label_col_name].values
    X = pandas_df.drop(columns=drop_columns)
    feature_names = list(X.columns.values)
    X = X.values
    X = scaler.fit_transform(X)



    print(X.shape)
    print(y.shape)

    return X, y, feature_names


def get_features(X):
    """Analyze features in dataset"""
    features_dict = {}
    features_dict["number_of_samples"] = X.shape[0]
    features_dict["number_of_features"] = X.shape[1]
    X_flat = X.reshape(-1)
    features_dict["maximum"] = max(X_flat)
    features_dict["minimum"] = min(X_flat)
    features_dict["average"] = np.mean(X_flat)
    master_std = []
    for column in range(len(X[0])):
        temp_col = X[column]
        print(column)
        print(np.std(temp_col))
        master_std.append(np.std(temp_col))
    features_dict["std"] = np.mean(master_std)
    features_dict["regression"] = 0
    features_dict["multiclassification"] = 0
    features_dict["binclassification"] = 1
    return features_dict

def suggest_architecture(input_vector):
    """Suggest neural network topology"""
    print(input_vector)
    prediction = clf.predict(input_vector)
    rounded_prediction = [abs(int(x+0.5)) for x in prediction[0]]
    return prediction, rounded_prediction

def create_keras_model(number_of_neurons_per_layer, input_dim_num):
    """Create Keras model"""
    model = Sequential()
    counter = 0
    for number_of_neurons in number_of_neurons_per_layer:
        if counter == 0:
            model.add(Dense(number_of_neurons, input_dim=input_dim_num, activation="sigmoid"))
        else:
            model.add(Dense(number_of_neurons, activation="sigmoid"))
        counter += 1

    return model


def create_baseline_model():
    """Create baseline model"""
    model = Sequential()
    model.add(Dense(10, activation="sigmoid"))
    model.add(Dense(10, activation="sigmoid"))
    model.add(Dense(10, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    return model
