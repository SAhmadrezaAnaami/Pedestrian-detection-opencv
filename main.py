import cv2
import numpy as np

img = cv2.imread("RES/1.jpg" , 1)



hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
face_cascade = cv2.CascadeClassifier('RES/haarcascade_frontalface_default.xml')



(fullbodys, _) = hog.detectMultiScale(img, winStride=(6, 6),padding=(400, 400),scale=1.04)

# fullbody = cv2.CascadeClassifier("haarcascade_fullbody.xml")
# fullbodys = fullbody.detectMultiScale(img, scaleFactor=1.19, minNeighbors=3)


 
for (x, y, w, h) in fullbodys:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    faces = face_cascade.detectMultiScale(img[y:y+h , x:x+w],scaleFactor=1.2, minNeighbors=3)
    for (x1,y1,w1,h1) in faces: 
         cv2.rectangle(img[y:y+h , x:x+w],(x1,y1),(x1+w1,y1+h1),(255,255,0),2)
    # cv2.imshow("Image", img[y:y+h , x:x+w])
    # cv2.waitKey(0)

cv2.imshow("Image", img)
cv2.waitKey(0)
   
cv2.destroyAllWindows()



