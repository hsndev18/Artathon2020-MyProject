### General imports ###
from __future__ import division

import numpy as np
import pandas as pd
import cv2

from time import time
from time import sleep
import re
import os

import argparse
from collections import OrderedDict

### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage

import dlib

from tensorflow.keras.models import load_model
from imutils import face_utils

import requests

# from multiprocessing import Process

global shape_x
global shape_y
global input_shape
global nClasses

happy = cv2.imread('emotes/happy.jpg')
sad = cv2.imread('emotes/sad.jpg')
angry = cv2.imread('emotes/angry.jpg')
neutral = cv2.imread('emotes/neutral.jpg')

emote = {0 : angry, 3 : happy, 4 : sad, 6 : neutral}
current = 6
prev = 6
imgs = {'current' : 6, 'prev' : 6}
timers = {
    'timer' : 0.0,
    'startTime' : 0.0,
    'timerRunning' : False,
    'lock' : False
}
fadein = 0

# cv2.imshow('transition test', angry)

def checkTime ():
    # emote = {0 : angry, 3 : happy, 4 : sad, 6 : neutral}
    # for IN in range(0,100):
    #     fadein = IN/100.0
    #     dst = cv2.addWeighted(currentImg, 1 - IN/100.0, img, IN/100.0, 0)
    #     dim = (int(dst.shape[1] * 0.5), int(dst.shape[0] * 0.5))
    #     resized = cv2.resize(cv2.addWeighted(currentImg, 1 - IN/100.0, img, IN/100.0, 0),
    #     (int(neutral.shape[1] * 0.5), int(neutral.shape[0] * 0.5)),
    #     interpolation = cv2.INTER_AREA)
    #     print(fadein)
    #     sleep(0.015)
    #     if fadein == 1.0:
    #         fadein = 1.0
    timers['timer'] = time() - timers['startTime']
    if timers['timerRunning']:
        if timers['timer'] > 1:
            imgs['prev'] = imgs['current']
            timers['timer'] = 0
            timers['startTime'] = 0
            timers['timerRunning'] = False
            timers['lock'] = False
        return
    else:
        timers['startTime'] = time()
        timers['timerRunning'] = True
        timers['lock'] = True
        return
    

def show_webcam() :

    shape_x = 48
    shape_y = 48
    input_shape = (shape_x, shape_y, 1)
    nClasses = 7

    thresh = 0.25
    frame_check = 20

    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_face(frame):
        
        #Cascade classifier pre-trained model
        cascPath = 'Models/face_landmarks.dat'
        faceCascade = cv2.CascadeClassifier(cascPath)
        
        #BGR -> Gray conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Cascade MultiScale classifier
        detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,
                                                      minSize=(shape_x, shape_y),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
        coord = []
                                                      
        for x, y, w, h in detected_faces :
            if w > 100 :
                sub_img=frame[y:y+h,x:x+w]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
                coord.append([x,y,w,h])

        return gray, detected_faces, coord

    def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
        
        gray = faces[0]
        detected_face = faces[1]
        
        new_face = []
        
        for det in detected_face :
            #Region dans laquelle la face est détectée
            x, y, w, h = det
            #X et y correspondent à la conversion en gris par gray, et w, h correspondent à la hauteur/largeur
            
            #Offset coefficient, np.floor takes the lowest integer (delete border of the image)
            horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
            vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
            
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #gray transforme l'image
            extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
            
            #Zoom sur la face extraite
            new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
            #cast type float
            new_extracted_face = new_extracted_face.astype(np.float32)
            #scale
            new_extracted_face /= float(new_extracted_face.max())
            #print(new_extracted_face)
            
            new_face.append(new_extracted_face)
        
        return new_face


    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    model = load_model('Models/video.h5')
    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks  = dlib.shape_predictor("Models/face_landmarks.dat")

    #Lancer la capture video
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        
        face_index = 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)
        #gray, detected_faces, coord = detect_face(frame)

        for (i, rect) in enumerate(rects):
            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Identify face coordinates
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y+h,x:x+w]
            
            #Zoom on extracted face
            if (face.shape[0] != 0 and face.shape[1] != 0):
                face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
            
            #Cast type float
            face = face.astype(np.float32)
            
            #Scale
            face /= float(face.max())
            face = np.reshape(face.flatten(), (1, 48, 48, 1))
            
            #Make Prediction
            prediction = model.predict(face)
            # print(prediction)
            prediction[0][5] = 0
            prediction[0][2] = 0
            prediction[0][1] = 0
            prediction[0][4] += 0.05
            prediction_result = np.argmax(prediction)
            
            # Rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
            # cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
            for (j, k) in shape:
                cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)
            
            # # # 1. Add prediction probabilities
            cv2.putText(frame, "----------------",(40,100 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Emotional report : Face #" + str(i+1),(40,120 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Angry : " + str(round(prediction[0][0],3)),(40,140 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Disgust : " + str(round(prediction[0][1],3)),(40,160 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Fear : " + str(round(prediction[0][2],3)),(40,180 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Happy : " + str(round(prediction[0][3],3)),(40,200 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Sad : " + str(round(prediction[0][4],3)),(40,220 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Surprise : " + str(round(prediction[0][5],3)),(40,240 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Neutral : " + str(round(prediction[0][6],3)),(40,260 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            
            # 2. Annotate main image with a label
            if prediction_result == 0 :
                cv2.putText(frame, "Angry",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if not timers['lock'] and imgs['current'] != 0:
                    timers['timerRunning'] = True
                    imgs['current'] = 0

            # elif prediction_result == 1 :
            #     cv2.putText(frame, "Disgust",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # elif prediction_result == 2 :
            #     cv2.putText(frame, "Fear",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 3 :
                cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if not timers['lock'] and imgs['current'] != 3:
                    timers['timerRunning'] = True
                    imgs['current'] = 3
            elif prediction_result == 4 :
                cv2.putText(frame, "Sad",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if not timers['lock'] and imgs['current'] != 4:
                    timers['timerRunning'] = True
                    imgs['current'] = 4
            # elif prediction_result == 5 :
                # cv2.putText(frame, "Surprise",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Neutral",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if not timers['lock'] and imgs['current'] != 6:
                    timers['timerRunning'] = True
                    imgs['current'] = 6

            # 3. Eye Detection and Blink Count
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            # Compute Eye Aspect Ratio
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # And plot its contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # 4. Detect Nose
            nose = shape[nStart:nEnd]
            noseHull = cv2.convexHull(nose)
            cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

            # 5. Detect Mouth
            mouth = shape[mStart:mEnd]
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            
            # 6. Detect Jaw
            jaw = shape[jStart:jEnd]
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
            
            # 7. Detect Eyebrows
            ebr = shape[ebrStart:ebrEnd]
            ebrHull = cv2.convexHull(ebr)
            cv2.drawContours(frame, [ebrHull], -1, (0, 255, 0), 1)
            ebl = shape[eblStart:eblEnd]
            eblHull = cv2.convexHull(ebl)
            cv2.drawContours(frame, [eblHull], -1, (0, 255, 0), 1)

        if timers['timerRunning']:
            # if timers['timer'] > 
            timers['timer'] = time() - timers['startTime']
            if timers['timer'] > 1:
                imgs['prev'] = imgs['current']
                timers['timer'] = 0
                timers['startTime'] = 0
                timers['timerRunning'] = False
                timers['lock'] = False
        else:
            timers['startTime'] = time()

        dim = (int(neutral.shape[1] * 0.5), int(neutral.shape[0] * 0.5))
        
        # cv2.putText(frame,'Number of Faces : ' + str(len(rects)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
        # cv2.putText(frame, str(timers['timer']),(40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
        cv2.imshow('Video', frame)
        if timers['timer'] > 1:
            timers['timer'] = 1
        if timers['timer'] < 0.05:
            timers['timer'] = 0
        if imgs['prev'] != imgs['current']:
            cv2.imshow('Art', cv2.resize(cv2.addWeighted(emote[imgs['prev']], 1 - timers['timer'], emote[imgs['current']], timers['timer'], 0), dim, interpolation = cv2.INTER_AREA))
        # print(current, '\n', shared['lock'])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    show_webcam()

if __name__ == "__main__":
    main()
