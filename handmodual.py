import numpy as np
import cv2 as cv
import mediapipe as mp
import numpy as np
##################
wCam, hCam=640,480
##################

cap = cv.VideoCapture(0)

cap.set(3,wCam)
cap.set(4,hCam)

class handDetection():
    def __init__(self, mode=False, maxHands=2, detectioncon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectioncon=detectioncon
        self.trackCon=trackCon
        self.mpHands=mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectioncon,self.trackCon) 
        self.mpDraw=mp.solutions.drawing_utils
    def findHands(self,img):
        imgRGB= cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results= self.hands.process(imgRGB)
    #print(results)
        if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    
                        for id,lm in enumerate(handLms.landmark):
                #print(id, lm)
                            h, w,c = img.shape
                            cx, cy= int(lm.x * w), int(lm.y * h)
                        print(id, cx, cy)
                        cv.circle(img,(cx,cy),15, (225,0,225),cv.FILLED)
                
                self.mpDraw.draw_landmarks(img,handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
def main():
    
    cap = cv.VideoCapture(0)

    Detection=handDetection()


while True:
        success, img= cap.read()
        handDetection.findHands()
        cv.imshow("img",img)
        cv.waitKey(1)




if __name__=="__main__":
     main()