import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
#from IPython.display import clear_output

batch_size = 500
num_classes = 2
epochs = 81

"""load data"""
data = np.load('02272018_1000_XY.npz')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

"""change array shape to (None, 100, 100, 3)"""
x_test = np.transpose(x_test, (3, 0, 1, 2))
x_train = np.transpose(x_train, (3, 0, 1, 2))
"""only consider one weather parameter"""
x_test = np.concatenate((x_test[:, :, :, 0:1], x_test[:, :, :, 0:1], x_test[:, :, :, 0:1]), axis=3)/100.0
x_train = np.concatenate((x_train[:, :, :, 0:1], x_train[:, :, :, 0:1], x_train[:, :, :, 0:1]), axis=3)/100.0

"""change y_test and y_train to auxiliary point"""
y_test = np.transpose(y_test[:, 4, :])/100.0
y_train = np.transpose(y_train[:, 4, :])/100.0

#y_test = np.transpose(y_test[:, :, :])/100.0
#y_train = np.transpose(y_train[:, :, :])/100.0

input_shape = (100, 100, 3)

"""build cnn model"""
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))


model.compile(
              loss='binary_crossentropy',
              optimizer='adam',
              metrics=['mae'])

'''
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()


plot_losses = PlotLosses()
'''

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          #callbacks=[plot_losses],
          validation_data=(x_test, y_test))
print(history.history.keys())

#score = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# plot
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'])

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mean_absolute_error')
plt.xlabel('epoch')
plt.ylabel('mean_absolute_error')
plt.legend(['train', 'test'])

plt.show()

# save model
model.save('Model_03162018_Epoch40.h5')
