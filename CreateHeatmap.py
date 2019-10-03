import cv2
import numpy as np
# import pandas as pd
import csv

sigma = 1.0
skeletons = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
             [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
keypoints = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142, 309, 1, 177, 320, 2, 191, 398, 2, 237, 317, 2, 233, 426,
             2, 306, 233, 2, 92, 452, 2, 123, 468, 2, 0, 0, 0, 251, 469, 2, 0, 0, 0, 162, 551, 2]
x, y, w, h = [73.35, 206.02, 300.58, 372.5]
skeletons = np.array(skeletons) - 1

gt_heatmap = []
keypoints = np.array(keypoints).reshape((-1, 3))
keypoints = (keypoints - [x, y, 0]) / [w, h, 1] * [96, 72, 1]
# print(keypoints)
boundary_x = np.zeros((19, 2))
boundary_y = np.zeros((19, 2))

for index in range(19):
    gt_heatmap.append(np.ones((96, 72)))
    gt_heatmap[index].tolist()
    # print(skeletons[index])
    point_a = keypoints[skeletons[index][0]]
    point_b = keypoints[skeletons[index][1]]
    # print(point_a, point_b)
    if point_a[2] == 0 or point_b[2] == 0:
        continue
    boundary_x[index] = [point_a[0], point_b[0]]
    boundary_y[index] = [point_a[1], point_b[1]]

# print(boundary_y)

for index in range(19):
    if boundary_x[index][0] != 0 or boundary_x[index][1] != 0 or boundary_y[index][0] != 0 or boundary_y[index][1] != 0:
        cv2.line(gt_heatmap[index], (int(boundary_x[index][0]), int(boundary_y[index][0])),
                 (int(boundary_x[index][1]), int(boundary_y[index][1])), 0)
    gt_heatmap[index] = np.uint8(gt_heatmap[index])
    gt_heatmap[index] = cv2.distanceTransform(gt_heatmap[index], cv2.DIST_L2, 5)
    gt_heatmap[index] = np.float32(np.array(gt_heatmap[index]))
    gt_heatmap[index] = gt_heatmap[index].reshape(72 * 96)
    print((gt_heatmap[index]) < 3. * sigma)
    (gt_heatmap[index])[(gt_heatmap[index]) < 10. * sigma] = \
        np.exp(-(gt_heatmap[index])[(gt_heatmap[index]) < 10 * sigma] / 2. * sigma * sigma)
    (gt_heatmap[index])[(gt_heatmap[index]) >= 10. * sigma] = 0.
    gt_heatmap[index] = gt_heatmap[index].reshape([96, 72])

# print(gt_heatmap[2][40:60, 50:70])

for i in range(19):
    cv2.imwrite('image/' + str(i) + '.jpg',
                np.uint8(gt_heatmap[i][:, :, np.newaxis] * 255))

with open('test2.csv', 'w', newline='')as f:
    f_csv = csv.writer(f)
    f_csv.writerows(gt_heatmap[2])
