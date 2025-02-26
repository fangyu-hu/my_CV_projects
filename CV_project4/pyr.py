import cv2

# 读取图像
img = cv2.imread("orange.jpg")

# 显示原始图像
cv2.imshow('img', img)
print(img.shape)

# 上采样
up = cv2.pyrUp(img)
cv2.imshow('up', up)
print(up.shape)

# 保存上采样图片
cv2.imwrite('up_sampled.png', up)

# 下采样
down = cv2.pyrDown(img)
cv2.imshow('down', down)
print(down.shape)

# 保存下采样图片
cv2.imwrite('down_sampled.png', down)

# 等待按键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
