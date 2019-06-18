#!/usr/bin/env python
#_*_coding:utf-8_*_

import argparse
import tensorflow as tf
import sys
import numpy as np
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image

import tensorflow.keras.backend as K
#K.clear_session()

# The export path contains the name and the version of the model
#tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
K.set_learning_phase(0)
model = tf.keras.models.load_model('model.h5')
#model = tf.keras.models.load_model('checkpoint')
#model.trainable=False
#for layer in model.layers:
#    layer.trainable = False
#    layer.training = False
#    if isinstance(layer, tf.keras.layers.BatchNormalization):
#        layer._per_input_updates = {}
#    elif isinstance(layer, tf.keras.layers.Dropout):
#        layer._per_input_updates = {}
#print model.get_config()
#sys.exit(0)
print model.layers[3].weights
for i in model.layers[3].get_weights():
    print i

#sys.exit(0)

target_w, target_h = 128, 128
test_datagen = ImageDataGenerator(
        rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'dataset',
        #'ds_0610',
        target_size=(target_w, target_h),
        #batch_size=380,
        shuffle=True,
        class_mode='categorical')
print test_generator.__dict__
#for i in test_generator[0][0]:
#    print i.tolist()
#sys.exit(0)
#print model.outputs
#inp = model.input
#outputs = [layer.output for layer in model.layers]
#print len(outputs)
#get_all_layer_outputs = K.function([basic_model.layers[0].input],
#                                  [l.output for l in basic_model.layers[1:]])
#print model.layers[0], model.layers[1]
#layer=2
#intermediate_model = Model(inputs=model.layers[0].input,
#                              outputs=[l.output for l in model.layers[1:]])
#                              #outputs=[model.layers[2].output])
#print test_generator[0][0].shape
#print test_generator[0][0][0].tolist()
#int_res = intermediate_model.predict(np.expand_dims(test_generator[0][0][0], axis=0))
##print len(int_res), len(model.layers[1:2])
#for i in range(len(int_res)):
#    print model.layers[i+1].name
#    print int_res[i].tolist()
#print model.summary()
#print model.get_weights()
#print "layer weights"
##print model.get_layer("conv1").get_weights()
#print "test_generator"
#print test_generator[0]
#print model.predict(test_generator)#, batch_size=len(test_generator[0]))
print model.evaluate_generator(test_generator)
#print model.evaluate(test_generator[0][0], test_generator[0][1], verbose=0)#, batch_size=len(test_generator[0]))
#print outputs
#functor = K.function([inp+[K.learning_phase()]], outputs)
#layer_outs = functor([test_generator[0], 0])
#print layer_outs

#print model.predict(test_generator[0])
#print model.evaluate(test_generator[0][0], test_generator[0][1])

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']

# Preprocessing our input image
img = image.img_to_array(image.load_img(image_path, target_size=(128, 128))) / 255.
img = np.expand_dims(img, axis=0)
print img.tolist()
print model.predict(img)

#tensorflow_model_server --model_base_path=/home/ubuntu/export/ --rest_api_port=9000 --model_name=test
#export FLASK_ENV=development && export FLASK_APP=app.py && flask run --host=0.0.0.0 
