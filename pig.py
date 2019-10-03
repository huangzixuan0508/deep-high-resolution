import cv2
import numpy as np
# import pandas as pd
import csv

sigma = 1.0

#
# gt_heatmap=np.ones((50, 50))
# # print(skeletons[index])
# # print(point_a, point_b)
#
# # print(boundary_y)
#
# cv2.line(gt_heatmap, (20,25),
#              (35,35), 0)
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
#
# # print(gt_heatmap[2][40:60, 50:70])
#
#
# cv2.imwrite('image/lalala.jpg',
#             np.uint8(gt_heatmap[:, :, np.newaxis] * 255))
#
# with open('test1.csv', 'w', newline='')as f:
#     f_csv = csv.writer(f)
#     f_csv.writerows(gt_heatmap)

# axis_x = np.array([1,-1])
# axis_y = np.array([1,0])
# lx = np.sqrt(axis_x.dot(axis_x))
# ly = 1
# cos_angle = axis_x.dot(axis_y) / (lx * ly)
# angle = np.arccos(cos_angle)
# angle2 = angle / np.pi
#
# if axis_x[1] < 0:
#     angle2 = - angle2
#
# print(lx, int(angle2))
import torch
a = torch.tensor([[1,2],[3,4]])
a = a.repeat(1,2).reshape(2,2,2)
a = torch.transpose(a,0,2,1)
a = a.reshape((2,4))
print(a)