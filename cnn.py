from __future__ import print_function

import keras

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 15

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()


model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu', input_shape = input_shape ))
model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu', input_shape = input_shape ))
model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()

#using tensorboard callback to visualize the CNN algorithm
tensor_b =keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

#DataAugmentation
gen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
generator = gen.flow(x_train, y_train, batch_size) 
model.fit_generator(
        generator,
        steps_per_epoch=60000/batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        validation_steps=10000/batch_size,
        use_multiprocessing=False,
        shuffle=True
        )


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, git validation_data=(x_test, y_test),callbacks=[tensor_b])

score = model.evaluate(x_test, y_test, verbose = 0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
