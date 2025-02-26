import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

image0 = mpimg.imread("blmg1.jpg")
image2 = mpimg.imread("blmg3.jpg")

plt.figure()
plt.imshow(image0)
plt.savefig('image0.png', dpi = 300)

plt.figure()
plt.imshow(image2)
plt.savefig('image2.png', dpi = 300)

sift = cv2.xfeatures2d.SIFT_create()
keyp0, des0 = sift.detectAndCompute(image0, None)
keyp2, des2 = sift.detectAndCompute(image2, None)
keyp_image0 = cv2.drawKeypoints(image0, keyp0, None)
keyp_image2 = cv2.drawKeypoints(image2, keyp2, None)

plt.figure()
plt.imshow(keyp_image0)
plt.savefig('keyp_image0.png', dpi = 300)

plt.figure()
plt.imshow(keyp_image2)
plt.savefig('keyp_image2.png', dpi = 300)

ratio = 0.85
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(des0, des2, k = 2)
good_matches = []
for m1, m2 in raw_matches:
    if m1.distance < ratio * m2.distance:
        good_matches.append([m1])

matches02 = cv2.drawMatchesKnn(image0, keyp0, image2, keyp2, good_matches, None, flags = 2)

plt.figure()
plt.imshow(matches02)
plt.savefig('matches02.png', dpi = 300)


print("匹配对的数目:", len(good_matches))


if len(good_matches) > 4:
    ptsA = np.float32([keyp0[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsB = np.float32([keyp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
    outputimg02 = cv2.warpPerspective(image2, H, (image0.shape[1], image0.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    plt.figure()
    plt.imshow(image0)
    plt.figure()
    plt.imshow(image2)
    plt.figure()
    plt.imshow(outputimg02)
    plt.savefig('outputimg02.png', dpi=300)