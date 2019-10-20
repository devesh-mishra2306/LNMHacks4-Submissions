from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


from predictor import EnsembleBuilder
#from train_driver import TrainDriver
#from flask import Flask, render_template, url_for, request
import os
import speech_recognition as sr



def largest(arr,n): 
  
    # Initialize maximum element 
    max = arr[0] 
  
    # Traverse array elements from second 
    # and compare every element with  
    # current max 
    for i in range(1, n): 
        if arr[i] > max: 
            max = arr[i] 
    return max

x='a'

def speech():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak the review")
        audio=r.listen(source)
        print("Thank you")
        text=r.recognize_google(audio)
#print("Text : "+text)
    eb = EnsembleBuilder()
    result = eb.make_prediction(text)
    x='q'
    return result, text


# parameters for loading data and images
detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []
count=0
x=''
ca=0
cs=0
ch=0
ss=0
cn=0

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
            ca=ca+1
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
            cs=cs+1
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
            ch=ch+1
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
            ss=ss+1
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
            cn=cn+1
        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    count=count+1
    #if(count==100):
     #   x='q'
    if(count==20):
        s_result,s_text=speech()
        break
    #if x=='q':
     #   break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


array=[]
array.append(ca)
array.append(cs)
array.append(ch)
array.append(ss)
array.append(cn)
n = len(array) 
Ans = largest(array,n)
#print(Ans)
#print(array)
a=""
for i in range (0,5):
    if(array[i]==Ans):
        if(i==0):
            a= "Reviewer was looking angry by the experience offered."
        if(i==1):
            a= "Reviewer was looking sad by the experience offered"
        if(i==2):
            a= "Reviewer was looking happy by the experience offered"
        if(i==3):
            a= "Reviewer was looking surprised by the experience offered"
        if(i==4):
            a= "Reviewer was looking neutral by the experience offered"
#print(a)
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("________________________________________________________________________________________________________________________________________________")
print("************************************************************************************************************************************************")
print("")
print("")
print("")
print("")
print("     Review : "+s_text+"        ")
print("")
print("")
print("")
print("     Prediction : "+s_result+"                   ")
print("")
print("")
print("")
print("     Video result: "+a+"                       ")
print("")
print("")
print("")
print("")
print("________________________________________________________________________________________________________________________________________________")
print("************************************************************************************************************************************************")
#print(a)
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")