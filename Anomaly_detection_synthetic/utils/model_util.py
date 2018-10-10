# -*- coding: utf-8 -*-
"""
@author: akshitbudhraja
"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_json
import os
import numpy as np

def build_model():
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(LSTM(
            input_length=99,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
            output_dim=layers['output']))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop")
    return model

def load_model(model_path):
    json_file = open(os.path.join(model_path, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(model_path, "model.h5"))
    return loaded_model

def run_model(trained_model, start_index, stop_index, d):
    x, y, truncate_y = d.get_raw_data()
    sequence_length = x.shape[0] + 100
    if start_index >= stop_index or stop_index > sequence_length or start_index <= 0:
        print("Invalid start and end points")
        return list(), list()
    truncate_seq_length = max(0, min(100, stop_index + 1) - start_index)
    y_pred = list(truncate_y[start_index - 1:start_index + truncate_seq_length])
    y_ground = list(truncate_y[start_index - 1:start_index + truncate_seq_length])
    if stop_index >= 100:
        predicted = trained_model.predict(x[100 - min(100, stop_index):stop_index - 100,:,:])
        y_ground += list(y[100 - min(100, stop_index):stop_index - 100])
        y_pred += list(np.reshape(predicted, (predicted.size,)))
    return y_pred, y_ground
    
    
    
    
    
    