import cv2
import numpy as np


def Gradient_Magnitude(fx, fy):
    mag = np.sqrt((np.float64(fx) ** 2) + (np.float64(fy) ** 2))
    return np.around(mag)

def Gradient_Direction(fx, fy):
    g_dir = np.rad2deg(np.arctan2(np.float64(fy), np.float64(fx))) + 180
    return g_dir


# %%
def Digitize_angle(Angle):
    quantized = np.zeros((Angle.shape[0], Angle.shape[1]))
    for i in range(Angle.shape[0]):
        for j in range(Angle.shape[1]):
            if 0 <= Angle[i, j] <= 22.5 or 157.5 <= Angle[i, j] <= 202.5 or 337.5 < Angle[i, j] < 360:
                quantized[i, j] = 0
            elif 22.5 <= Angle[i, j] <= 67.5 or 202.5 <= Angle[i, j] <= 247.5:
                quantized[i, j] = 1
            elif 67.5 <= Angle[i, j] <= 122.5 or 247.5 <= Angle[i, j] <= 292.5:
                quantized[i, j] = 2
            elif 112.5 <= Angle[i, j] <= 157.5 or 292.5 <= Angle[i, j] <= 337.5:
                quantized[i, j] = 3
    return quantized

def Non_Max_Supp(qn, magni):
    M = np.zeros(qn.shape)
    a, b = np.shape(qn)
    for i in range(a - 1):
        for j in range(b - 1):
            if qn[i, j] == 0:
                if magni[i - 1, j] <= magni[i, j] and magni[i, j] >= magni[i + 1, j]:
                    M[i, j] = magni[i, j]
                else:
                    M[i, j] = 0
            if qn[i, j] == 1:
                if magni[i - 1, j - 1] <= magni[i, j] and magni[i, j] >= magni[i + 1, j + 1]:
                    M[i, j] = magni[i, j]
                else:
                    M[i, j] = 0
            if qn[i, j] == 2:
                if magni[i, j - 1] < magni[i, j] and magni[i, j] > magni[i, j + 1]:
                    M[i, j] = magni[i, j]
                else:
                    M[i, j] = 0
            if qn[i, j] == 3:
                if magni[i - 1, j + 1] <= magni[i, j] and magni[i, j] >= magni[i + 1, j - 1]:
                    M[i, j] = magni[i, j]
                else:
                    M[i, j] = 0
    return M

def _double_thresholding(g_suppressed, low_threshold, high_threshold):
    g_thresholded = np.zeros(g_suppressed.shape)
    for i in range(0, g_suppressed.shape[0]):  # loop over pixels
        for j in range(0, g_suppressed.shape[1]):
            if g_suppressed[i, j] < low_threshold:  # lower than low threshold
                g_thresholded[i, j] = 0
            elif g_suppressed[i, j] >= low_threshold and g_suppressed[i, j] < high_threshold:  # between thresholds
                g_thresholded[i, j] = 128
            else:  # higher than high threshold
                g_thresholded[i, j] = 255
    return g_thresholded


def _hysteresis(g_thresholded):
    g_strong = np.zeros(g_thresholded.shape)
    for i in range(0, g_thresholded.shape[0]):  # loop over pixels
        for j in range(0, g_thresholded.shape[1]):
            val = g_thresholded[i, j]
            if val == 128:  # check if weak edge connected to strong
                if g_thresholded[i - 1, j] == 255 or g_thresholded[i + 1, j] == 255 or g_thresholded[
                    i - 1, j - 1] == 255 or g_thresholded[i + 1, j - 1] == 255 or g_thresholded[i - 1, j + 1] == 255 or \
                        g_thresholded[i + 1, j + 1] == 255 or g_thresholded[i, j - 1] == 255 or g_thresholded[
                    i, j + 1] == 255:
                    g_strong[i, j] = 255  # replace weak edge as strong
            elif val == 255:
                g_strong[i, j] = 255  # strong edge remains as strong edge
    return g_strong

image_3 = cv2.imread("lena_gray_512.tif")
image = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("image")
cv2.imshow("image",image)
cv2.waitKey(0)
gauss=np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])/159
smooth_img=cv2.filter2D(image,-1,gauss)


gx = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
gy = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

fx = cv2.filter2D(smooth_img,-1,gx)
fy = cv2.filter2D(smooth_img,-1,gy)

mag = Gradient_Magnitude(fx, fy)
Angle = Gradient_Direction(fx, fy)
quantized = Digitize_angle(Angle)
nms = Non_Max_Supp(quantized, mag)

cv2.namedWindow("nms")
cv2.imshow("nms",nms)
cv2.waitKey(0)

threshold = _double_thresholding(nms, 20, 60)


hys = _hysteresis(threshold)
cv2.namedWindow("canny")
cv2.imshow("canny",hys)
cv2.waitKey(0)
