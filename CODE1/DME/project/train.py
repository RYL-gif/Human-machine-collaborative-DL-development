import os
import pandas as pd
import numpy as np

import random
import keras
import tensorflow as tf
# from keras import backend as K
import efficientnet.keras as efn
from keras_classification_models.keras import Classifiers
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# network parameters
model_name = 'effientnetB4'  # resnet50  inceptionv3  effientnetB4  inceptionresnetv2
model_save_name = '{}_best_model.h5'.format(model_name)
savepath = './logs/{}/'.format(model_name)
train_data_dir = '../training dataset'
validate_data_dir = '../val'
number_of_trainsample = 2160
number_of_validationsample = 720

n_classes = 2
epochs = 80
resize = 355

batch_size = 20
multi_cls = False

if not os.path.exists(savepath):
    os.makedirs(savepath)


def train():
    if model_name == 'effientnetB4':
        base_model = efn.EfficientNetB4(input_shape=(resize, resize, 3), weights='imagenet', include_top=False)
    else:
        base_net, preprocess_input = Classifiers.get(model_name.lower())
        base_model = base_net(input_shape=(resize, resize, 3), weights='imagenet', include_top=False)
    # build model top
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)

    # classification output
    if not multi_cls:
        output = keras.layers.Dense(1, activation="sigmoid")(x)
    else:
        output = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.models.Model(inputs=[base_model.input], outputs=[output])

    if multi_cls:
        reduce_lr = ReduceLROnPlateau(monitor='val_categorical_crossentropy', factor=0.1, patience=4, verbose=1,
                                      min_lr=0.000001)  # 1.mean_absolute_error  2.binary_crossentropy
        early_stopping = EarlyStopping(monitor='val_categorical_crossentropy', min_delta=0, patience=15)
        modelcheckpoint = ModelCheckpoint(savepath + model_save_name, monitor='val_categorical_crossentropy', verbose=0,
                                          save_best_only=True, save_weights_only=True, period=1)

        callbacks = [reduce_lr, early_stopping, modelcheckpoint]

        opt = optimizers.Adam(lr=1e-3, epsilon=1e-07)
        # opt = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_crossentropy'])


    else:
        reduce_lr = ReduceLROnPlateau(monitor='val_binary_crossentropy', factor=0.1, patience=3, verbose=1,
                                      # mode='max',
                                      min_lr=0.000001)  # 1.mean_absolute_error  2.binary_crossentropy
        early_stopping = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, patience=10)  # mode='max'
        modelcheckpoint = ModelCheckpoint(savepath + model_save_name, monitor='val_binary_crossentropy', verbose=0,
                                          # mode='max',
                                          save_best_only=True, save_weights_only=True, period=1)

        callbacks = [reduce_lr, early_stopping, modelcheckpoint]

        opt = optimizers.Adam(lr=1e-2, epsilon=1e-07)
        # opt = optimizers.SGD(lr=1e-3, decay=1e-7, momentum=0.99, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_crossentropy'])

    # generate training data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest")

    # flow_from_directory: PIL type RGB image
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(resize, resize),
        batch_size=batch_size,
        class_mode='binary')

    # generate testing data
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=30,
                                      horizontal_flip=True,
                                      fill_mode="nearest")

    validation_generator = test_datagen.flow_from_directory(
        validate_data_dir,
        target_size=(resize, resize),
        batch_size=batch_size,
        class_mode='binary')

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(train_generator,
                        steps_per_epoch=number_of_trainsample / batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        # class_weight=cw,
                        validation_data=validation_generator,
                        validation_steps=number_of_validationsample / batch_size)


if __name__ == '__main__':
    train()
