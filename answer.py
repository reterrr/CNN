#!/N/u/
'''
Our own model developped from scratch the do leaf classification
using batch Normalization as regularization method and relu as Activation function
'''

from keras.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator


img_width, img_height = 500, 500  #324x324
validation_data_dir = 'data/testing'
batch_size = 32


valid_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = valid_datagen.flow_from_directory(validation_data_dir,
                                                         target_size=(img_width, img_height),
                                                         batch_size=batch_size,
                                                         class_mode='categorical')

yep = load_model('model.h5')

yep.evaluate(validation_generator)