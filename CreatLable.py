import numpy as np
import json

skeletons = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
             [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 121.01652779682617, -132.65586752593015, 1.0, 128.00390619039717, -90.44761417086055, 1.0, 34.88552708502482, 62.7004277886672, 1.0, 151.48927354766738, 19.272197067269854, 1.0, 157.54364474646383, 20.045249425420995, 1.0, 36.68787265568828, 72.55281157671782, 1.0, 101.59724405711013, 28.835500847743077, 1.0, 60.07495318350236, -92.86240522611175, 1.0, 50.47771785649585, 56.309932474020215, 1.0, 108.70602559196064, -140.59933933652056, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

skeletons = np.array(skeletons) - 1
# print(skeletons.shape)
axis_y = np.array([1, 0])
picture_size = [288, 384, 1]
new_annotations = []

with open('person_keypoints_val2017.json', 'r') as f:
    label = json.load(f)
    annotations = label['annotations']
    for item in annotations:
        if item['category_id'] == 1:
            people = dict()
            people['num_keypoints'] = item['num_keypoints']
            people['iscrowd'] = item['iscrowd']
            keypoints = np.array(item['keypoints'], dtype=float).reshape((-1, 3))

            # 裁剪稍后再写
            x, y, w, h = item['bbox']
            # print(x,y,w,h)

            keypoints -= [x, y, 0]
            keypoints = keypoints / [w, h, 1] * picture_size
            # print(keypoints)

            new_skeleton = np.zeros((19, 3), dtype=int)
            for id, skeleton in enumerate(skeletons):
                point_a = keypoints[skeleton[0]]
                # print(point_a)
                point_b = keypoints[skeleton[1]]
                if point_a[2] == 0 or point_b[2] == 0:
                    continue
                axis_x = (point_b - point_a)[:-1]
                # print(x)
                lx = np.sqrt(axis_x.dot(axis_x))
                if lx == 0:
                    continue
                ly = 1
                cos_angle = axis_x.dot(axis_y) / (lx * ly)
                angle = np.arccos(cos_angle)
                angle2 = angle * 180 / np.pi

                if axis_x[1] < 0:
                    angle2 = - angle2
                # print(angle2)
                # print(lx,angle2)
                new_skeleton[id] = [int(lx), int(angle2), 1]
            new_skeleton = new_skeleton.flatten().tolist()
            people['skeleton'] = new_skeleton
            people['image_id'] = item['image_id']
            people['bbox'] = item['bbox']
            people['category_id'] = item['category_id']
            people['id'] = item['id']
            print(people['id'])
            new_annotations.append(people)
            # break
my_dict = dict()
my_dict['annotations'] = new_annotations
with open("val_box_skeleton.json", "w") as dump_f:
    json.dump(my_dict, dump_f)
