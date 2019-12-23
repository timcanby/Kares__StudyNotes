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

evaluate_every = 1 # interval for evaluating on tasks
loss_every =1
best = -1
epoch=5
def loadData(X,Y,batch_size):
    return [X[x:x + batch_size] for x in range(0, len(X), batch_size)],[Y[y:y + batch_size] for y in range(0, len(Y), batch_size)]


(inputs, targets) = loadData(X_train, Y_train, 1000)


fig, ax = plt.subplots()
for i in range(1, epoch):
    (inputs,targets)=loadData(X_train,Y_train,1000)
    x1_plt=[]
    y1_plt=[]
    for N_iter in range(0,np.shape(inputs)[0]):

        ax.cla()
        loss=model.train_on_batch(inputs[N_iter],targets[N_iter])
        x1_plt.append(N_iter)
        y1_plt.append(loss[1])
        ax.plot(x1_plt, y1_plt,label='train')
        plt.pause(0.1)



    if i % evaluate_every == 0:
        print("evaluating")
        (inputs_test, targets_test) = loadData(X_test, Y_test, 1000)
        for N_iter in range(0, np.shape(inputs_test)[0]):
            val_acc = model.evaluate(x=inputs_test[N_iter], y=targets_test[N_iter])
            print(val_acc)
            if val_acc[1] >= best:
                print("saving")
                model.save('weight.h5')
                best=val_acc

    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i,loss))