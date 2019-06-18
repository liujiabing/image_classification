#!/usr/bin/env python
#_*_coding:utf-8_*_

import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

K.set_learning_phase(1)

target_w, target_h = 128, 128
datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

test_datagen = ImageDataGenerator(
        rescale=1./255)

datadir="dataset"
train_generator = datagen.flow_from_directory(
        datadir,
        target_size=(target_w, target_h),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        datadir,
        target_size=(target_w, target_h),
        #batch_size=380,
        shuffle=True,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        datadir,
        target_size=(target_w, target_h),
        #batch_size=380,
        shuffle=True,
        class_mode='categorical')

base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(target_w, target_h, 3))
#base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(target_w, target_h, 3))
#print base_model.layers[3].name
#names = [weight.name for layer in model.layers for weight in layer.weights]
#print base_model.layers[3].weights
#weights = model.get_weights()
#for i in base_model.layers[3].get_weights():
#    print i
#sys.exit(0)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dropout(rate=0.3)(x)
predictions = Dense(train_generator.num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

cp_callback = ModelCheckpoint("./checkpoint", save_weights_only=False, verbose=1, save_best_only=True)

model.fit_generator(
        train_generator,
        epochs=30,
        callbacks=[cp_callback],
        validation_data=validation_generator,)
        #validation_steps=1)
        #validation_steps=validation_generator.samples/validation_generator.batch_size)

K.set_learning_phase(0)
#model.trainable=False
#for l in model.layers:
#    l.trainable = False
#    l.training = False
#print model.get_config()

#intermediate_model = Model(inputs=model.layers[0].input,
#                              outputs=[l.output for l in model.layers[1:]])
#print test_generator[0][0][0].tolist()
#int_res = intermediate_model.predict(np.expand_dims(test_generator[0][0][0], axis=0))
#for i in range(len(int_res)):
#    print model.layers[i+1].name
#    print int_res[i].tolist()
#print model.summary()
#print >>open("model.json", "w"), model.to_json()
#model.save_weights("model_weights.h5")

#from keras.models import model_from_json
#model = model_from_json(json_string)

#bn = Model(inputs=model.layers[0].input, outputs=[model.layers[2].output, model.layers[3].output])
#int_res = bn.predict(np.expand_dims(test_generator[0][0][0], axis=0))
#print model.layers[2].name
#print int_res[0].tolist()
#print model.layers[3].name
#print int_res[1].tolist()
#print bn.summary()
#print bn.get_weights()
#print "layer weights"
##print model.get_layer('conv1').get_weights()
#print "test_generator"
#print test_generator[0]
#print model.predict(test_generator[0])
#print model.evaluate(test_generator[0][0], test_generator[0][1], verbose=0)
model.save('model.h5')


## Save entire model to a HDF5 file
#model.save('my_model.h5')
## Recreate the exact same model, including weights and optimizer.
#new_model = keras.models.load_model('my_model.h5')
#tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

#https://www.tensorflow.org/tutorials/keras/save_and_restore_models
