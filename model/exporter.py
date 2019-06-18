#!/usr/bin/env python
#_*_coding:utf-8_*_

import tensorflow as tf
import sys
import numpy as np
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint


# The export path contains the name and the version of the model
#from tensorflow.keras.models import model_from_json
#model = model_from_json(open('model.json').read().strip())
#model.load_weights("model_weights.h5", by_name=True)

model = tf.keras.models.load_model('model.h5')

tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
export_path = './tmpp'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})

#tensorflow_model_server --model_base_path=/home/ubuntu/export/ --rest_api_port=9000 --model_name=test
#export FLASK_ENV=development && export FLASK_APP=app.py && flask run --host=0.0.0.0 
