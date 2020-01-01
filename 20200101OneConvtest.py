import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.layers import Conv2D
from time import *
from keras.models import Model
import keras
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()

classNumber = len(set(y_test))
Y_train = keras.utils.to_categorical(y_train, classNumber)
Y_test = keras.utils.to_categorical(y_test, classNumber)
X_train = x_train.reshape(x_train.shape + (1,))

X_test = x_test.reshape(x_test.shape + (1,))
input_shape = (28, 28, 1)

def runParamater(N_filter,kernel_size):
    model = Sequential()
    model.add(Conv2D(N_filter, kernel_size=(kernel_size,kernel_size), strides=(1, 1), padding='same', activation='tanh',name='conv',input_shape=(28, 28, 1)))
    model.add(Flatten(name='flatten'))
    model.add(Dense(10,activation='softmax',name='d1'))
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])
    print(model.summary())
    batch_size = 100
    n_epochs = 1
    history=model.fit(x=X_train,y=Y_train, batch_size=batch_size,nb_epoch=n_epochs)
    return history.history['acc']




acclist = []
timer = []
N_filterRC=[]
data=[]
accmax=-1
maxfilter=0
for N_filter in range(1,50):
    N_filterRC.append(N_filter)

    begin_time = time()

    acc=runParamater(N_filter,5)
    end_time = time()
    run_time = end_time - begin_time

    acclist.append(acc[0])
    timer.append(run_time)
    data.append([acc[0],run_time])
    if acc[0]>accmax:
        maxfilter=N_filter
        accmax=acc[0]

    print('time：', run_time)









#print(np.shape(X_test[1]))
#testdata=np.array(X_test[1]).reshape(-1,28, 28, 1)
#intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv').output)

#intermediate_output = intermediate_layer_model.predict(testdata)

#print(np.shape(intermediate_output))
