import keras

from keras.datasets import mnist
from keras import Sequential
from keras.layers import AveragePooling2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
print(np.shape(mnist.load_data()))

(x_train, y_train), (x_test, y_test)=mnist.load_data()

classNumber=len(set(y_test))
Y_train = keras.utils.to_categorical(y_train,classNumber)
Y_test = keras.utils.to_categorical(y_test ,classNumber)
X_train = x_train.reshape(x_train.shape + (1,))

X_test = x_test.reshape(x_test.shape + (1,))
input_shape =(28,28,1)
print(input_shape)

model = Sequential()

model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='tanh', input_shape=input_shape))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='tanh'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(classNumber, activation='softmax'))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

epochs = 5
batch_size = 1000
evaluate_every = 1
history=model.fit(x=X_train,y=Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=1)

#(print(history.history))
#(print(history.params))
(print(np.shape(history.validation_data[0])))


epoch_array = np.array(range(5))
plt.plot(epoch_array, history.history['loss'], history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()
print(model.summary())
print(Y_train)
