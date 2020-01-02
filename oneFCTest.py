
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist

import keras



(x_train, y_train), (x_test, y_test) = mnist.load_data()

classNumber = len(set(y_test))
Y_train = keras.utils.to_categorical(y_train, classNumber)
Y_test = keras.utils.to_categorical(y_test, classNumber)
X_train = x_train.reshape(x_train.shape + (1,))

X_test = x_test.reshape(x_test.shape + (1,))
input_shape = (28, 28, 1)



model = Sequential()
#model.add(Conv2D(20, kernel_size=(3,3), strides=(1, 1), padding='same', activation='tanh',name='conv',input_shape=(28, 28, 1)))
model.add(Flatten(name='flatten',input_shape=(28, 28, 1)))
model.add(Dense(10,activation='softmax',name='fc1'))
model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'])
print(model.summary())
batch_size = 100
n_epochs = 100
history=model.fit(x=X_train,y=Y_train, batch_size=batch_size,nb_epoch=n_epochs)