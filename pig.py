import cv2
import numpy as np
# import pandas as pd
import csv

sigma = 1.0

#
gt_heatmap=np.ones((10, 10))
# print(skeletons[index])
# print(point_a, point_b)

# print(boundary_y)

cv2.line(gt_heatmap, (-2,-2),
             (5,5), 0)
print(gt_heatmap)
# gt_heatmap = np.uint8(gt_heatmap)
# gt_heatmap = cv2.distanceTransform(gt_heatmap, cv2.DIST_L2, 5)
# # gt_heatmap = cv2.GaussianBlur(gt_heatmap, ksize=(9, 9), sigmaX=0, sigmaY=0)
# gt_heatmap= np.float32(np.array(gt_heatmap))
# print(gt_heatmap)
#
# gt_heatmap = gt_heatmap.reshape(50*50)
#
# (gt_heatmap)[(gt_heatmap) < 10. * sigma] = \
#     np.exp(-(gt_heatmap)[(gt_heatmap) < 10 * sigma] / 2. * sigma * sigma)
# (gt_heatmap)[(gt_heatmap) >= 10. * sigma] = 0.
# gt_heatmap = gt_heatmap.reshape([50, 50])

# print(gt_heatmap[2][40:60, 50:70])
#
#
# cv2.imwrite('image/lalala.jpg',
#             np.uint8(gt_heatmap[:, :, np.newaxis] * 255))
#
# with open('test1.csv', 'w', newline='')as f:
#     f_csv = csv.writer(f)
#     f_csv.writerows(gt_heatmap)


