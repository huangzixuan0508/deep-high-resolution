# import numpy as np
# import math
# #123
# # def angle(v1):
# #   dx1 = v1[0]
# #   dy1 = v1[1]
# #   dx2 = 1
# #   dy2 = 0
# #   angle1 = math.atan2(dy1, dx1)
# #   angle1 = int(angle1 * 180/math.pi)
# #   # print(angle1)
# #   angle2 = math.atan2(dy2, dx2)
# #   angle2 = int(angle2 * 180/math.pi)
# #   # print(angle2)
# #   if angle1*angle2 >= 0:
# #     included_angle = abs(angle1-angle2)
# #   else:
# #     included_angle = abs(angle1) + abs(angle2)
# #     # if included_angle > 180:
# #     #   included_angle = 360 - included_angle
# #   return included_angle
# x = np.array([89,-82])
# y = np.array([1,0])
# lx = np.sqrt(x.dot(x))
# ly = np.sqrt(y.dot(y))
# print(lx)
# print(ly)
# cos_angle=x.dot(y)/(lx*ly)
# angle=np.arccos(cos_angle)
# print(cos_angle)
# print(angle)
# angle2=angle*180/np.pi
# angle2 = - angle2
# print(angle2)
#
# # print(angle(x))
#
#
# # a = np.array([[1,2,3],[3,4,5]])
# a=np.array([2,2,2])
# print(a[:-1])

import torch
import torch.nn as nn
m = nn.AdaptiveAvgPool2d((2,2))
input = torch.randn(1, 64, 8, 9)
output = m(input)
print(output.size())

import numpy as np
import torch
import torch.nn

a = torch.rand([4,5,96,72])
a = a.reshape((4,5,-1)).split(1,1)
for i in a:
    print(i.size())

