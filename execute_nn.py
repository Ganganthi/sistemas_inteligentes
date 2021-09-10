import tensorflow as tf
from tensorflow import keras
from keras.callbacks import History


import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split

X_NORM = 60
Y_NORM = 945
TOPOLOGY_ID = 'nn_1'

SAVE_DIR = 'models/' + TOPOLOGY_ID

expecteds = pandas.read_csv("dados.csv", usecols=[1])
expecteds = np.array(expecteds)

new_model = tf.keras.models.load_model(SAVE_DIR)
new_model.summary()


entry = int(input("\nPlease insert a value between 0 and 60: "))

while entry >=0 and entry <= 60:
    entry_norm = np.array([entry/X_NORM])
    result = new_model.predict(entry_norm)
    print("Predicted value: ", float(result*Y_NORM), "\nExpected value: ", int(expecteds[entry]))
    entry = int(input("Please insert a value between 0 and 60: "))