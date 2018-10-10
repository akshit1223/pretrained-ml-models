# -*- coding: utf-8 -*-
"""
@author: akshitbudhraja
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from utils.data_util import dataGenerator
from utils.model_util import *

data_path = os.path.join(os.getcwd(), 'data')
model_path = os.path.join(os.getcwd(), 'models')

################## getting raw data #######################
d = dataGenerator(data_path)

################# running model ##########################
model = load_model(model_path)
results_dir = os.path.join(os.getcwd(), 'test_runs')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

##### 0 < test_start < =999  1 < test_end <= 1000 #########
test_start = 1
test_end = 1000

y_pred, y = run_model(model, test_start, test_end, d)

plt.figure(1)
plt.subplot(311)
plt.title("Actual Test Signal w/Anomalies")
plt.plot(y[:len(y)], 'b')
plt.subplot(312)
plt.title("Predicted Signal")
plt.plot(y_pred[:len(y_pred)], 'g')
plt.subplot(313)
plt.title("Squared Error")
mse = (((np.array(y_pred) - np.array(y)) ** 2))
plt.ylim(0, 3)
plt.plot(mse, 'r')
plt.savefig(os.path.join(results_dir, 'result_' + str(test_start) + '_' + str(test_end) + '.png'))