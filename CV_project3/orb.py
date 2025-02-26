import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1=cv.imread('blmg1.jpg',0)#queryImage
img2=cv.imread('blmg3.jpg',0)#trainImage
#Initiate ORB detector
orb=cv.ORB_create()##create函数参数较多
#find the keypointsand descriptors with ORB
kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2 =orb.detectAndCompute(img2,None)
#create BFMatcher object
bf=cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)##两点之间的距离A-B和B-A各计身
#Match descriptors.
matches =bf.match(des1,des2)
#Sortthem in theorder of their distance.
matches =sorted(matches,key =lambda x:x.distance)
#draw first few matches
img3 =cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
plt.imshow(img3),plt.show()