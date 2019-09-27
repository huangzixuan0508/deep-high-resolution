# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.num_body = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root  # ''
        self.image_set = image_set  # train

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY  # 8
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE  # 高斯
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA  # 3
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.use_different_body_weight = True
        self.axis_y = None
        self.joints_weight = 1
        self.body_weight = 1

        self.skeletons = None

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]  # 所有x的平均值和y的平均值

        left_top = np.amin(selected_joints, axis=0)  # 这些点涉及的左上角
        right_bottom = np.amax(selected_joints, axis=0)  # 这些点涉及的右下角

        w = right_bottom[0] - left_top[0]  # 所选区域的宽高
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:  # 保持长宽比
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5  # 返回的

        return center, scale

    def __len__(self, ):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        # print(joints)
        joints_copy = db_rec['joints_3d_copy']
        joints_vis = db_rec['joints_3d_vis']
        # body = db_rec['body_3d']
        # body_vis = db_rec['body_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)  # 随机缩放因子
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0  # 随机旋转因子

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                # 加我们的对称
                c[0] = data_numpy.shape[1] - c[0] - 1  # 重新确定镜像翻转后的中心点

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        body = np.zeros((self.num_body, 3), dtype=np.float)
        body_vis = np.zeros((self.num_body, 3), dtype=np.float)
        for idbody, skeleton in enumerate(self.skeletons):
            point_a = joints[skeleton[0]]
            # print(point_a)
            point_b = joints[skeleton[1]]
            # if point_a[2] == 0 or point_b[2] == 0:
            if joints_copy[skeleton[0]][2] == 0 or joints_copy[skeleton[1]][2] == 0:
                continue
            axis_x = (point_b - point_a)[:-1]
            # print(x)
            lx = np.sqrt(axis_x.dot(axis_x))
            if lx == 0:
                continue
            ly = 1
            cos_angle = axis_x.dot(self.axis_y) / (lx * ly)
            angle = np.arccos(cos_angle)
            angle = angle / np.pi
            # angle2 = angle * 180 / np.pi

            if axis_x[1] < 0:
                angle = - angle
            # print(angle2)
            # print(lx,angle2)
            body[idbody] = [lx/332.55, angle, 1]
            body_vis[idbody] = [1, 1, 0]

        joint_target, joint_target_weight = self.generate_target(joints, joints_vis)
        body_target, body_target_weight = self.generate_body_target(joints, joints_copy, body_vis)
        # for i in range(19):
        #     # print(image_file)
        #     cv2.imwrite('image/'+image_file.split('/')[-1][:-4]+'_'+str(i)+'.jpg', np.uint8(body_target[i][:,:,np.newaxis]*255))
        # for i in range(17):
        #     # print(image_file)
        #     cv2.imwrite('image/'+image_file.split('/')[-1][:-4]+'_'+str(i)+'_point.jpg', np.uint8(joint_target[i][:,:,np.newaxis]*255))
        joint_target = torch.from_numpy(joint_target)
        joint_target_weight = torch.from_numpy(joint_target_weight)
        body_target = torch.from_numpy(body_target)
        body_target_weight = torch.from_numpy(body_target_weight)
        body = torch.from_numpy(body)
        body_vis = torch.from_numpy(body_vis)



        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'body': body,
            'body_vis': body_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, joint_target, joint_target_weight, body_target, body_target_weight, body, body_vis, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std ** 2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2 ** 2) / ((0.2) ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size  # 缩小的尺寸
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)  # 点缩小后对应的尺寸
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]  # upleft
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]  # bottomright
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)  # [0~18}
                y = x[:, np.newaxis]  # 19 * 19
                x0 = y0 = size // 2  # 中心点
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_body_target(self, joints, joints_copy, body_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_body, 1), dtype=np.float32)
        target_weight[:, 0] = body_vis[:, 0]
        # print(joints)

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            gt_heatmap = []
            sigma = 1
            tmp_size = sigma * 3
            boundary_x = np.zeros((self.num_body, 2))
            boundary_y = np.zeros((self.num_body, 2))
            for index in range(self.num_body):
                feat_stride = self.image_size / self.heatmap_size

                gt_heatmap.append(np.ones((self.heatmap_size[1],
                                          self.heatmap_size[0]),
                                          np.float32))
                gt_heatmap[index].tolist()
                # print(skeletons[index])

                point_a = joints[self.skeletons[index][0]]
                point_b = joints[self.skeletons[index][1]]
                point_a_x = point_a[0] / feat_stride[0] + 0.5
                point_b_x = point_b[0] / feat_stride[0] + 0.5
                point_a_y = point_a[1] / feat_stride[1] + 0.5
                point_b_y = point_b[1] / feat_stride[1] + 0.5
                # print('nnnn:',joints[self.skeletons[index][0]], point_a_x,point_a_y)
                # if point_a[2] == 0 or point_b[2] == 0:
                if joints_copy[self.skeletons[index][0]][2] == 0 or joints_copy[self.skeletons[index][1]][2] == 0:
                    target_weight[index] = 0
                    continue
                ul_a = [int(point_a_x - tmp_size), int(point_a_y - tmp_size)]  # upleft
                br_a = [int(point_a_x + tmp_size + 1), int(point_a_y + tmp_size + 1)]  # bottomright

                if ul_a[0] >= self.heatmap_size[0] or ul_a[1] >= self.heatmap_size[1] \
                        or br_a[0] < 0 or br_a[1] < 0:
                    # If not, just return the image as is
                    target_weight[index] = 0
                    continue

                ul_b = [int(point_b_x - tmp_size), int(point_b_y - tmp_size)]  # upleft
                br_b = [int(point_b_x + tmp_size + 1), int(point_b_y + tmp_size + 1)]  # bottomright
                if ul_b[0] >= self.heatmap_size[0] or ul_b[1] >= self.heatmap_size[1] \
                        or br_b[0] < 0 or br_b[1] < 0:
                    # If not, just return the image as is
                    target_weight[index] = 0
                    continue



                boundary_x[index] = [point_a_x, point_b_x]
                boundary_y[index] = [point_a_y, point_b_y]

            # print(boundary_y)

            for index in range(self.num_body):
                # if index == 8 or index == 9:
                #     print('boundary:',boundary_x[index],boundary_y[index])
                #     print('point:', joints[self.skeletons[index][0]], joints[self.skeletons[index][1]])

                if (boundary_x[index][0] > 0.1 and boundary_x[index][1] > 0.1 and boundary_y[index][0] > 0.1 and \
                    boundary_y[index][1] > 0.1) and (
                        boundary_x[index][0] != boundary_x[index][1] or boundary_y[index][0] != boundary_y[index][1]):
                    # print('lalalala')
                    cv2.line(gt_heatmap[index], (int(boundary_x[index][0]), int(boundary_y[index][0])),
                             (int(boundary_x[index][1]), int(boundary_y[index][1])), 0)
                else:
                    target_weight[index] = 0
                gt_heatmap[index] = np.uint8(gt_heatmap[index])
                gt_heatmap[index] = cv2.distanceTransform(gt_heatmap[index], cv2.DIST_L2, 5)
                gt_heatmap[index] = np.float32(np.array(gt_heatmap[index]))
                gt_heatmap[index] = gt_heatmap[index].reshape(self.heatmap_size[0] * self.heatmap_size[1])
                (gt_heatmap[index])[(gt_heatmap[index]) < 3. * sigma] = \
                    np.exp(-(gt_heatmap[index])[(gt_heatmap[index]) < 3 * sigma] *
                           (gt_heatmap[index])[(gt_heatmap[index]) < 3 * sigma] / 2. * sigma * sigma)
                (gt_heatmap[index])[(gt_heatmap[index]) >= 3. * sigma] = 0.
                gt_heatmap[index] = gt_heatmap[index].reshape([self.heatmap_size[1], self.heatmap_size[0]])

        if self.use_different_body_weight:
            target_weight = np.multiply(target_weight, self.body_weight)
        return np.array(gt_heatmap), target_weight
