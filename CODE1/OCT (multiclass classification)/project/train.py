import os
import pandas as pd
import numpy as np

import random
import keras
import tensorflow as tf
#from keras import backend as K
import efficientnet.keras as efn
from keras_classification_models.keras import Classifiers
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import data_generator


from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))


#network parameters
model_name = 'resnet50' # resnet50  inceptionv3  effientnetB4  inceptionresnetv2
model_save_name = '{}_best_model.h5'.format(model_name)
savepath = './logs/{}/'.format(model_name)
train_data_dir='../training dataset2'
validate_data_dir='../val'
number_of_trainsample=4608
number_of_validationsample=1152

n_classes = 4
epochs = 80
resize = 255

batch_size = 16
multi_cls = True

# label file
id_tag_path='word_id.txt'
word_id={}
with open(id_tag_path,'r') as f:
    words=f.readlines()
    for item in words:
        arr=item.strip().split(' ',1)[1]
        lable = item.strip().split(' ',1)[0]
        word_id[arr]=lable



if not os.path.exists(savepath):
    os.makedirs(savepath)


def train():
    # train_generator = data_generator(train_data_dir, 100, BATCHSIZE = batch_size, image_size = resize, classes= n_classes, word_id = word_id)

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
        # opt = optimizers.SGD(lr=1e-2, decay=1e-5, momentum=0.98, nesterov=True)
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

        opt = optimizers.Adam(lr=1e-3, epsilon=1e-07)
        # opt = optimizers.SGD(lr=1e-3, decay=1e-5, momentum=0.95, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_crossentropy'])


    # generate training data
    train_generator = data_generator(train_data_dir, 100, BATCHSIZE = batch_size, image_size = resize, classes= n_classes, word_id = word_id)
    validation_generator = data_generator(validate_data_dir, 100, BATCHSIZE = batch_size, image_size = resize, classes= n_classes, word_id = word_id)



    # fits the model on batches with real-time data augmentation:
    model.fit_generator(train_generator.get_mini_batch(),
                        steps_per_epoch=number_of_trainsample / batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        # class_weight=cw,
                        validation_data=validation_generator.get_mini_batch(),
                        validation_steps=number_of_validationsample / batch_size)

if __name__ == '__main__':
    train()

