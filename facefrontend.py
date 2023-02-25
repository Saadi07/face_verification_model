# -*- coding: utf-8 -*-
"""
@author: saadi09
"""

# In this file I have loaded the trained model, but it does not predict accurately.
# I want my model to predict the name of person when the face detected in webcam.
# Below method does not work fine for me
# Because instead of adding labels statically as I did below...
# My model should be able to detect and display that person name dynamically
# if (pred[0][0] > 0.5):
 #   name = "Saad"
# if (pred[0][1] > 0.5):
 #   name = "Nawaf"



from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image

model = load_model('face_verfication.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face


video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        print(pred[0][1])

        # name = "None Matching"

        if (pred[0][0] > 0.5):
            name = "Saad"
        if (pred[0][1] > 0.5):
            name = "Nawaf"

        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()