# -*- coding: utf-8 -*-
"""
@author: akshitbudhraja
"""
import numpy as np

class dataGenerator():
    
    def __init__(self, path):
        self.data_path = path
        self.sequence_length = 100
        self.random_data_dup = 10 # each sample randomly duplicated between 0 and 9 times, see dropin function
        self.raw_data = list()
        f = open(path, 'r')
        while True:
            line = f.readline()
            if not line:
                break
            line = line.rstrip('\n')
            self.raw_data.append(float(line))
        f.close()
    
    def z_norm(self, result):
        result_mean = result.mean()
        result_std = result.std()
        result -= result_mean
        result /= result_std
        return result, result_mean
    
    def dropin(self, X, y):
        """ The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
        http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
        :param X: Each row is a training sequence
        :param y: Tne target we train and will later predict
        :return: new augmented X, y
        """
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        X_hat = []
        y_hat = []
        for i in range(0, len(X)):
            for j in range(0, np.random.random_integers(0, self.random_data_dup)):
                X_hat.append(X[i, :])
                y_hat.append(y[i])
        return np.asarray(X_hat), np.asarray(y_hat)
    
    def get_split_prep_data(self, train_start, train_end, test_start, test_end):
        result = []
        for index in range(train_start, train_end - self.sequence_length):
            result.append(self.raw_data[index: index + self.sequence_length])
        result = np.array(result)  # shape (samples, sequence_length)
        result, result_mean = self.z_norm(result)
        
        train = result[train_start:train_end, :]
        np.random.shuffle(train)
        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_train, y_train = self.dropin(X_train, y_train)
        
        result = []
        for index in range(test_start, test_end - self.sequence_length):
            result.append(self.raw_data[index: index + self.sequence_length])
        result = np.array(result)  # shape (samples, sequence_length)
        result, result_mean = self.z_norm(result)
        
        X_test = result[:, :-1]
        y_test = result[:, -1]
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
        return X_train, y_train, X_test, y_test
    
    
            