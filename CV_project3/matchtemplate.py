import numpy as np
import matplotlib.pyplot as plt
import cv2

# 读取图像，注意文件名应该在引号内
img_src = cv2.imread('cv3_1.png')

# 确保图像已经正确读取
if img_src is None:
    print("Error: 图像未正确读取，请检查文件名和路径。")
    exit()

# 提取模板图像
img_templ = img_src[200:400, 100:200].copy()

# 打印原图像和模板图像的形状
print('img_src.shape:', img_src.shape)
print('img_templ.shape:', img_templ.shape)

# 进行模板匹配，选择一种方法，例如 cv2.TM_CCOEFF_NORMED
method = cv2.TM_CCORR_NORMED
result = cv2.matchTemplate(img_src, img_templ, method)

# 打印结果矩阵的形状和数据类型
print('result.shape:', result.shape)
print('result.dtype:', result.dtype)

# 计算匹配位置（最小值和最大值的位置）
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 通常我们关心最大值的位置，因为它表示模板与源图像中某部分的最大匹配度
print('Best match at position:', max_loc)

# 绘制匹配的区域
h, w = img_templ.shape[:2]
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_src, top_left, bottom_right, 255, 2)

# 使用matplotlib显示图像
plt.imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)), plt.show()