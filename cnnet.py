from keras.src.callbacks import CSVLogger
from keras.src.layers import BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input
from keras import backend as K
import matplotlib.pyplot as plt
from keras.layers import Activation, Dropout, Flatten, Dense

def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["accuracy"])
    ax2.plot(history.history["val_accuracy"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()


img_width, img_height = 500, 500  #324x324

callback = [CSVLogger('leaf_v1-1.csv')]

train_data_dir = 'data/training'
validation_data_dir = 'data/testing'
nb_train_samples = 1400
nb_validation_samples = 200
epochs = 100
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# layer 1
model = Sequential()
model.add(Input(shape=input_shape))

model.add(
    Conv2D(32, (5, 5), strides=(2, 2), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# layer 2
model.add(
    Conv2D(64, (5, 5), strides=(2, 2), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))

# layer 3
model.add(
    Conv2D(128, (5, 5), strides=(1, 1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

# layer 4
model.add(
    Conv2D(128, (5, 5), strides=(1, 1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# flatten
model.add(Flatten(input_shape=(11, 11)))

# dense layer Classifier
model.add(Dense(800))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(800))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones'))
model.add(Activation('softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# data augmentation for training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# data augmentation for testing:
valid_datagen = ImageDataGenerator(rescale=1. / 255,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# take path to dir and generate batches of augmented data
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    class_mode='categorical',
                                                    batch_size=batch_size)

# Repeat the training generator indefinitely

validation_generator = valid_datagen.flow_from_directory(validation_data_dir,
                                                         target_size=(img_width, img_height),
                                                         batch_size=batch_size,
                                                         class_mode='categorical')

# model training
history = model.fit(
    x=train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_steps=nb_validation_samples // batch_size,
    validation_data=validation_generator,
    callbacks=callback
)


display_learning_curves(history)

model.save('model.keras')

model.evaluate(validation_generator)