import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade=  cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade= cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

glass_cascade= cv2.CascadeClassifier('MK5-18.xml')

cap= cv2.VideoCapture(1)
ret,img = cap.read()
while True:
        
        ret,img = cap.read() 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
                for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

                edges = cv2.Canny(roi_gray,100,200)
                glass = glass_cascade.detectMultiScale(roi_gray,1.04,5)
                      
                for(gx,gy,gw,gh) in glass:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(roi_color, (gx,gy),(gx+gw, gy+gh), (255,255,0), 2)
                        cv2.putText(roi_color,'glass',(gx,gy-3), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
                               
        
        cv2.imshow('img',img)
        k= cv2.waitKey(30) & 0xff
        if k ==27:
                break
                       
               
cap.release()

cv2.destroyAllWindows()

