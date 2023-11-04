# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import cv2 as cv

#loading the model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
z='placeholder'
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Suprise', 'Neutral']

#taking video
cap = cv.VideoCapture(0)

while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face.detectMultiScale(gray, 1.3  , 10)
        #detecting faces
        for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0)
                cv.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv.NORM_L2, dtype=cv.CV_32F)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                #predicting the emotion
                yhat= loaded_model.predict(cropped_img)
                cv.putText(frame, labels[int(np.argmax(yhat))], (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv.LINE_AA)
                emotion=labels[int(np.argmax(yhat))]
                if emotion!=z:
                        print("Emotion: "+labels[int(np.argmax(yhat))])
                        z=emotion

                cv.imshow('camera',frame)

        if cv.waitKey(1) == ord('q'):
                break

cap.release()
cv.destroyAllWindows()