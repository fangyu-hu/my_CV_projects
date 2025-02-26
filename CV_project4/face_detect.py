import cv2 as cv
face_cascade=cv.CascadeClassifier('D:/PYTHON/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade=cv.CascadeClassifier('D:/PYTHON/Lib/site-packages/cv2/data/haarcascade_eye.xml')
img =cv.imread('liz.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.1,3)
for (x,y,h,w) in faces:
  img=cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv.namedWindow('faces Detected!')
cv.imshow('faces Detected!',img)
cv.imwrite('faces.jpg',img)
cv.waitKey(0)