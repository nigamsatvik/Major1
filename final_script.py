#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:22:55 2020

@author: satvik
"""
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
from tensorflow.keras.preprocessing import image
xclass=["COVID-19","NORMAL","Viral Pneumonia"]
img1 = image.load_img('test_img.jpg', color_mode='grayscale',target_size=(128, 128,))
input_arr = tensorflow.keras.preprocessing.image.img_to_array(img1)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
print(xclass[np.argmax(predictions)])
