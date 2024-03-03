from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.datasets import mnist
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32') / 255.0
x_test = x_test.reshape(10000,784).astype('float32') / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_val = x_train[:42000]
y_val = y_train[:42000]
x_train = x_train[42000:]
y_train = y_train[42000:]

model = Sequential()
model.add(Dense(units=64,input_dim=28*28,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer = 'sgd',metrics=['accuracy'])

early_stopping = EarlyStopping(patience=20)
hist = model.fit(x_train,y_train,epochs=1000, batch_size=10, validation_data=(x_val,y_val),callbacks=[early_stopping])

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y',label='train loss')
loss_ax.plot(hist.history['val_loss'],'r',label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='validation accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

loss_and_metrics = model.evaluate(x_test,y_test,batch_size=32)

print('##### Test Result #####')
print('loss : ',str(loss_and_metrics[0]))
print('Accuracy : ',str(loss_and_metrics[1]))

from tensorflow.python.keras.models import load_model
model.save('mnist_mlp_model.h5')

