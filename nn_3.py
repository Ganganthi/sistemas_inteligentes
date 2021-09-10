import tensorflow as tf
from tensorflow import keras
from keras.callbacks import History


import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split


EPOCHS = 1000
X_NORM = 60
Y_NORM = 945
TOPOLOGY_ID = 'nn_3'


SAVE_DIR = 'models/' + TOPOLOGY_ID

csv_file = pandas.read_csv("dados.csv")


inputs_temp = pandas.read_csv("dados.csv", usecols=[0])
expecteds_temp = pandas.read_csv("dados.csv", usecols=[1])

x_train, x_test, y_train, y_test = train_test_split(inputs_temp/X_NORM, expecteds_temp/Y_NORM, test_size=0.2, random_state=10)

x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = np.array(x_train)
y_train = np.array(y_train)



model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(1,), activation='tanh'),
    keras.layers.Dense(16, activation='tanh'),
    keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    keras.layers.Dense(1, activation='linear')
])
	
model.summary()

history = History()


model.compile(optimizer='adam',
              loss='mse',
              metrics=['mean_absolute_error'])

model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[history])

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

model.save(SAVE_DIR)


x_axis = np.array(range(X_NORM+1))

results = model.predict(x_axis/X_NORM)

fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)
ax1.plot(x_train*X_NORM,y_train*Y_NORM,'o',x_test*X_NORM,y_test*Y_NORM,'o')
ax1.set_title('Valores retirados do CSV')
ax1.grid()

ax2.plot(range(EPOCHS), history.history.get('loss'))
ax2.set_title('Erro Medio Absoluto')
ax2.grid()

ax3.plot(x_axis, results*Y_NORM,color='green')
ax3.set_title('Valores previstos pela RNA')
ax3.grid()

ax4.plot(inputs_temp, expecteds_temp,color='red')
ax4.plot(x_axis, results*Y_NORM,color='green')
ax4.set_title('Original X RNA')
ax4.grid()

fig.suptitle(TOPOLOGY_ID)
fig.canvas.set_window_title(TOPOLOGY_ID)
plt.show()
