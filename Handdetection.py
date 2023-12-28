import numpy as np
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
##################
wCam, hCam=640,480
##################


cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

mpHands=mp.solutions.hands
hands = mpHands.Hands(False) 
mpDraw=mp.solutions.drawing_utils
while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    #lmlist=
    #print(results)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                #print(id, lm)
                            h, w,c = img.shape
                            cx, cy= int(lm.x * w), int(lm.y * h)
                        #print(id, cx, cy)
                            cv.circle(img,(cx,cy),15, (225,0,225),cv.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cv.imshow("img", img)
    cv.waitKey(0)
