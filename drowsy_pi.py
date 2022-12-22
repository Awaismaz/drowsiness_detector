import cv2
import os
# from keras.models import load_model
import numpy as np

# import tensorflow as tf
import tflite_runtime.interpreter as tflite

import time
import os
import OPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)

GPIO.output(11, False)
GPIO.output(13, False)
GPIO.output(15, False)


basedir = os.path.dirname(__file__)

def tflite_predict(input_data):

    input_data = np.float32(input_data)

    basedir = os.path.dirname(__file__)
    # interpreter = tf.lite.Interpreter(os.path.join(basedir,"models","drowsy.tflite"))
    interpreter = tflite.Interpreter(os.path.join(basedir,"models","drowsy.tflite"))
    
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return(output_data)


face = cv2.CascadeClassifier(os.path.join(basedir,"haar cascade files","haarcascade_frontalface_alt.xml"))
leye = cv2.CascadeClassifier(os.path.join(basedir,"haar cascade files","haarcascade_lefteye_2splits.xml"))
reye = cv2.CascadeClassifier(os.path.join(basedir,"haar cascade files","haarcascade_righteye_2splits.xml"))

lbl=['Close','Open']




# model = load_model(os.path.join(basedir,"models","stable1.h5"))

# Convert the model using TFLiteConverter
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open ("drowsy.tflite" , "wb") .write(tflite_model)



path = os.getcwd()
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

dec_counter=0
alarm_state=0


while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)



    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        

        # rpredict_x=model.predict(r_eye) 
        rpredict_x=tflite_predict(r_eye) 

        rpred=np.argmax(rpredict_x,axis=1)
        #rpred = model.predict(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        
        # lpredict_x=model.predict(r_eye) 
        lpredict_x=tflite_predict(l_eye) 
        
        lpred=np.argmax(lpredict_x,axis=1)
        #lpred = model.predict(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1


    elif(rpred[0]==1 or lpred[0]==1):
        if (alarm_state==1):
            dec_counter+=1
            if dec_counter==2:
                score=0
                dec_counter=0
                alarm_state=0

        score=score-1
    
        
    if(score<0):
        score=0   
    
    if(score>2):
        
        alarm_state=1

        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2

    if alarm_state==1:
        GPIO.output(11, True)
        GPIO.output(15, True)
    else:
        GPIO.output(11, False)
        GPIO.output(15, False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()