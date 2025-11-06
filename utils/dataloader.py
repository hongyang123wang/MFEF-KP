import math
import random

import cv2
import cv2 as cv
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input, coord_to_homog, coord_to_normal
import imgaug.augmenters as iaa


def create_light_augmenters():
    return {
        "brightness": iaa.OneOf([
            iaa.Add((-60, 60)),
            iaa.Multiply((0.3, 1.8)),
            iaa.LinearContrast((0.5, 2.0))
        ]),
        "highlight": iaa.OneOf([
            iaa.BlendAlphaHorizontalLinearGradient(
                foreground=iaa.Add(120),
                min_value=0.3,
                max_value=0.7
            )
        ]),
        # "color_temp": iaa.OneOf([
        #     iaa.AddToHueAndSaturation((-30, 30)),
        #     iaa.WithChannels(0, iaa.Add((-20, 20))),
        #     iaa.WithChannels(2, iaa.Add((-20, 20)))
        # ]),
        "light_dir": iaa.OneOf([
            iaa.BlendAlphaVerticalLinearGradient(
                foreground=iaa.Multiply((0.5, 1.5)),
                min_value=0.2,
                max_value=0.8
            ),
            iaa.BlendAlphaHorizontalLinearGradient(
                foreground=iaa.Multiply((0.5, 1.5)),
                min_value=0.2,
                max_value=0.8
            )
        ])
    }


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class CenternetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, poly_setting=None):
        super(CenternetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)

        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] / 2), int(input_shape[1] / 2))
        self.num_classes = num_classes
        self.train = train
        if poly_setting is None:
            poly_setting = {"poly": False, "two_offset_branch": False, "direct_theta": False}
        self.poly = poly_setting["poly"]
        self.two_offset_branch = poly_setting["two_offset_branch"]
        self.direct_theta = poly_setting["direct_theta"]
        # 创建一个全局增强器字典
        self.AUGMENTERS_DICT = create_light_augmenters()

    def __len__(self):
        return self.length

    # 在线增强函数
    def apply_online_light_augmentation(self, image, apply_prob=0.5):
        """
        在线光照增强函数，替代原有的HSV变换

        Args:
            image: 输入图像，numpy数组，RGB格式，值范围0-255
            apply_prob: 应用增强的概率

        Returns:
            增强后的图像
        """
        # 以一定概率不执行增强
        if random.random() > apply_prob:
            return image

        # 随机决定应用1-2种增强（在线增强不宜过多，以免训练不稳定）
        num_augments = random.randint(1, 2)

        # 随机选择增强类型
        augment_types = random.sample(list(self.AUGMENTERS_DICT.keys()), num_augments)

        # 创建临时序列
        augmenters_to_apply = [self.AUGMENTERS_DICT[aug_type] for aug_type in augment_types]
        seq = iaa.Sequential(augmenters_to_apply, random_order=True)

        # 确保图像类型正确并应用增强
        if isinstance(image, np.ndarray):
            return seq(image=image)
        else:
            # 如果是PIL图像，先转numpy
            img_array = np.array(image)
            return seq(image=img_array)

    def __getitem__(self, index):
        index = index % self.length

        # -------------------------------------------------#
        #   进行数据增强
        #   此时获得的vertexes均相对于128*128特征图
        # -------------------------------------------------#
        image, vertexes, raw_img_w, raw_img_h = self.get_random_data(self.annotation_lines[index], self.input_shape,
                                                                     random=self.train)

        # 生成一系列特征图
        # 中心点热力图
        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        # 关键点热力图
        batch_vertexes_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        # 中心点偏移量
        batch_reg_ct_offset = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        if not self.poly:
            # 关键点相对于中心点偏移量
            batch_reg_vt_ct_offset = np.zeros((self.output_shape[0], self.output_shape[1], 8), dtype=np.float32)
        else:
            # 关键点相对于中心点偏移量,中心归一化距离、与中心点的角度cos、sin
            if self.two_offset_branch:
                batch_reg_vt_ct_offset = np.zeros((self.output_shape[0], self.output_shape[1], 4), dtype=np.float32)
                if self.direct_theta:
                    batch_reg_vt_ct_offset_angle = np.zeros((self.output_shape[0], self.output_shape[1], 4),
                                                            dtype=np.float32)
                else:
                    batch_reg_vt_ct_offset_angle = np.zeros((self.output_shape[0], self.output_shape[1], 8),
                                                            dtype=np.float32)
            else:
                if self.direct_theta:
                    batch_reg_vt_ct_offset = np.zeros((self.output_shape[0], self.output_shape[1], 8), dtype=np.float32)
                else:
                    batch_reg_vt_ct_offset = np.zeros((self.output_shape[0], self.output_shape[1], 12),
                                                      dtype=np.float32)
        # 关键点真实坐标(相对于128*128特征图)
        batch_vertexes = np.zeros((self.output_shape[0], self.output_shape[1], 8), dtype=np.float32)
        # 特征图每个位置处是否包含目标
        batch_reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        # 目标多边形变换至标准DM码的透视变换矩阵
        batch_reg_trans_matrixes = np.zeros((self.output_shape[0], self.output_shape[1], 3, 3), dtype=np.float32)

        if len(vertexes) != 0:
            raw_vertex = np.array(vertexes[:, 0:10], dtype=np.float32)  # 相对于512*512特征图
            raw_vertex_mask = np.array(vertexes[:, 0:10], dtype=np.float32)  # 相对于512*512特征图

            # vertex = np.array(vertexes[:, 0:10], dtype=np.float32)  # 相对于128*128特征图
            # 将关键点坐标在128*128特征图上裁切，防止超出范围
            raw_vertex[:, 0:10:2] = np.clip(raw_vertex[:, 0:10:2] / self.input_shape[1] * self.output_shape[1], 0,
                                            self.output_shape[1] - 1)
            raw_vertex[:, 1:10:2] = np.clip(raw_vertex[:, 1:10:2] / self.input_shape[0] * self.output_shape[0], 0,
                                            self.output_shape[0] - 1)
            vertex = raw_vertex.copy()

            # 初始化一个掩膜数组，用于保存每个目标的二值分割图
            segmentation_mask = np.zeros((self.input_shape[0], self.input_shape[1]), dtype=np.uint8)

            # 将关键点坐标变换至相对于原始图像
            if raw_img_h >= raw_img_w:
                raw_vertex[:, 0:10:2] = raw_vertex[:, 0:10:2] * max(raw_img_w, raw_img_h) / self.output_shape[1] - abs(
                    raw_img_w - raw_img_h) / 2.
                raw_vertex[:, 1:10:2] = raw_vertex[:, 1:10:2] * max(raw_img_w, raw_img_h) / self.output_shape[0]
            else:
                raw_vertex[:, 0:10:2] = raw_vertex[:, 0:10:2] * max(raw_img_w, raw_img_h) / self.output_shape[1]
                raw_vertex[:, 1:10:2] = raw_vertex[:, 1:10:2] * max(raw_img_w, raw_img_h) / self.output_shape[0] - abs(
                    raw_img_w - raw_img_h) / 2.

        # 循环每一个目标
        for i in range(len(vertexes)):
            bbox = vertex[i].copy()  # [ct, pt1, pt2, pt3, pt4]  相对于特征图
            raw_bbox = raw_vertex[i].copy()  # [ct, pt1, pt2, pt3, pt4]  相对于原图
            cls_id = int(vertexes[i, -1])

            bbox_raw_mask = raw_vertex_mask[i].copy()
            points_mask = bbox_raw_mask[2:10].reshape(4, 2).astype(np.int32)
            # 绘制多边形区域（白色）
            cv2.fillPoly(segmentation_mask, [points_mask], color=1)

            h, w = np.max(bbox[3:10:2]) - np.min(bbox[3:10:2]), np.max(bbox[2:10:2]) - np.min(bbox[2:10:2])
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                # -------------------------------------------------#
                #   计算真实框所属的特征点
                # -------------------------------------------------#
                ct = bbox[0:2]
                ct_int = ct.astype(np.int32)
                # ----------------------------#
                #   绘制中心点的高斯热力图
                # ----------------------------#
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                # 关键点热力图
                batch_vertexes_hm[:, :, cls_id] = draw_gaussian(batch_vertexes_hm[:, :, cls_id], bbox[2:4], radius)
                batch_vertexes_hm[:, :, cls_id] = draw_gaussian(batch_vertexes_hm[:, :, cls_id], bbox[4:6], radius)
                batch_vertexes_hm[:, :, cls_id] = draw_gaussian(batch_vertexes_hm[:, :, cls_id], bbox[6:8], radius)
                batch_vertexes_hm[:, :, cls_id] = draw_gaussian(batch_vertexes_hm[:, :, cls_id], bbox[8:10], radius)

                # ---------------------------------------------------#
                #   计算中心、关键点相对于中心的偏移量
                # ---------------------------------------------------#
                new_bbox = np.reshape(bbox[0:10], [-1, 2])
                new_ct_int = np.reshape(ct_int, [-1, 2])
                new_offset = new_bbox - new_ct_int
                new_offset = np.reshape(new_offset, [1, -1])
                batch_reg_ct_offset[ct_int[1], ct_int[0]] = new_offset[:, 0:2]
                if not self.poly:
                    batch_reg_vt_ct_offset[ct_int[1], ct_int[0]] = new_offset[:, 2:10]
                else:
                    dis_center = np.sqrt(np.sum((new_bbox - new_ct_int) ** 2, 1))
                    dis_center = np.reshape(dis_center, [-1, 1])  # [n, 1]
                    dis_angle = np.reshape(new_bbox - new_ct_int, [-1, 2])  # [n, 2]
                    thetas = []
                    if self.direct_theta:
                        for y, x in zip(dis_angle[1:, 1], dis_angle[1:, 0]):
                            theta = math.atan2(y,
                                               x)  # 必须要这样计算才能得到正确的[-pi, pi]角度值，np.arctan、math.atan只能获得[-pi/2, pi/2]的角度值，不正确
                            thetas.append(theta)
                        thetas = np.expand_dims(np.array(thetas), -1)
                    else:
                        dis_sin, dis_cos = dis_angle[1:, 1] / dis_center[1:, 0], dis_angle[1:, 0] / dis_center[1:, 0]
                        dis_sin, dis_cos = np.expand_dims(dis_sin, -1), np.expand_dims(dis_cos, -1)  # 添加维度
                    if self.two_offset_branch:
                        if self.direct_theta:
                            new_offset_angle = np.reshape(thetas, [1, -1])
                        else:
                            new_offset_angle = np.concatenate([dis_sin, dis_cos], 1)
                            new_offset_angle = np.reshape(new_offset_angle, [1, -1])
                        batch_reg_vt_ct_offset[ct_int[1], ct_int[0]] = np.reshape(dis_center[1:, :], [1, -1])
                        batch_reg_vt_ct_offset_angle[ct_int[1], ct_int[0]] = new_offset_angle
                    else:
                        if self.direct_theta:
                            new_offset = np.concatenate([dis_center[1:, :], thetas], 1)
                        else:
                            new_offset = np.concatenate([dis_center[1:, :], dis_sin, dis_cos], 1)
                        new_offset = np.reshape(new_offset, [1, -1])
                        batch_reg_vt_ct_offset[ct_int[1], ct_int[0]] = new_offset

                # 计算透视变换矩阵
                raw_points = np.array(raw_bbox[2:10]).reshape(-1, 2)
                dst_points = np.array([[0, 100], [0, 0], [100, 0], [100, 100]], dtype=np.float32)
                trans_matrix = cv.getPerspectiveTransform(raw_points, dst_points)
                batch_reg_trans_matrixes[ct_int[1], ct_int[0]] = trans_matrix

                # ---------------------------------------------------#
                #   将对应的mask设置为1
                # ---------------------------------------------------#
                batch_reg_mask[ct_int[1], ct_int[0]] = 1
                batch_vertexes[ct_int[1], ct_int[0]] = bbox[2:10]

        image = np.transpose(preprocess_input(image), (2, 0, 1))
        if not self.two_offset_branch or not self.poly:
            return (
                image, batch_hm, batch_vertexes_hm, batch_reg_ct_offset, batch_reg_vt_ct_offset, batch_reg_mask,
                batch_vertexes, batch_reg_trans_matrixes, raw_img_w, raw_img_h,
                segmentation_mask)
        else:
            return (
                image, batch_hm, batch_vertexes_hm, batch_reg_ct_offset, batch_reg_vt_ct_offset,
                batch_reg_vt_ct_offset_angle, batch_reg_mask, batch_vertexes, batch_reg_trans_matrixes, raw_img_w,
                raw_img_h,
                segmentation_mask)

    def letterbox_image(self, image, image_h, image_w, input_shape):
        featmap_w, featmap_h = input_shape
        scale = min(featmap_w / image_w, featmap_h / image_h)
        new_image_w = int(image_w * scale)
        new_image_h = int(image_h * scale)
        image = image.resize((new_image_w, new_image_h), Image.BICUBIC)
        new_image = Image.new('RGB', input_shape, (128, 128, 128))
        new_image.paste(image, ((featmap_w - new_image_w) // 2, (featmap_h - new_image_h) // 2))
        return new_image

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        image_w, image_h = image.size  # 原始图像尺寸
        featmap_h, featmap_w = input_shape  # feature_map尺寸
        scale = min(featmap_w / image_w, featmap_h / image_h)
        new_image_w = int(image_w * scale)
        new_image_h = int(image_h * scale)
        dx = (featmap_w - new_image_w) // 2
        dy = (featmap_h - new_image_h) // 2
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        # vertexs: [ct, pt1, pt2, pt3, pt4, cls]
        vertexes = np.array([np.array(list(map(int, vertex.split(',')))) for vertex in line[1:]])

        # 无数据增强
        # 有数据增强时的先验操作
        # resize image (加灰条)
        image = self.letterbox_image(image, image_h, image_w, input_shape)  # RGB, HWC

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(vertexes) > 0:
            np.random.shuffle(vertexes)  # 相对于h, w
            vertexes[:, 0:10:2] = vertexes[:, 0:10:2] * new_image_w / image_w + dx
            vertexes[:, 1:10:2] = vertexes[:, 1:10:2] * new_image_h / image_h + dy
            vertexes[:, 0:10][vertexes[:, 0:10] < 0] = 0
            vertexes[:, 0:10:2][vertexes[:, 0:10:2] > featmap_w] = featmap_w
            vertexes[:, 1:10:2][vertexes[:, 1:10:2] > featmap_h] = featmap_h

        # 如果进行数据增强(色域变换/随机水平翻转)
        if random:
            # 随机水平翻转
            flip = self.rand() < .5
            # 随机中心旋转
            rotation = self.rand() < .1
            # 随机色域变换
            # color_change = self.rand() < .5
            color_change = False
            # 随机透视变换
            # persp_transform = self.rand() < .1
            persp_transform = False

            if persp_transform:
                cv_img = np.array(image, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR, HWC
                shift = np.random.randint(-15, 15, [4, 2])  # 随机值大一点效果更明显
                pers_vertexes = []
                pers_control_points = np.array([0, 0, featmap_w, 0, 0, featmap_h, featmap_w, featmap_h]).astype(
                    np.float32).reshape(-1, 2)
                if len(vertexes) > 0:
                    pers_control_points_aft = pers_control_points + np.array(shift).astype(np.float32)
                    # 计算透视变换矩阵, 及变换后的图像
                    pers_trans_m = cv.getPerspectiveTransform(pers_control_points, pers_control_points_aft)
                    pers_trans_cv_img = cv.warpPerspective(cv_img, pers_trans_m, (cv_img.shape[1], cv_img.shape[0]))
                    image = Image.fromarray(np.uint8(pers_trans_cv_img[:, :, ::-1]))  # BGR -> RGB
                    # 对vertex标注做相应变换
                    for i in range(vertexes.shape[0]):
                        # 5个点的格式(ct, kp)
                        box2d_five_points = np.array([vertexes[i, 0:10]])
                        raw_box2d_five_points = box2d_five_points.reshape(5, 2)
                        # 变换为齐次坐标
                        homo_box2d_five_points = coord_to_homog(raw_box2d_five_points)
                        pers_box2d_five_points = np.matmul(pers_trans_m, homo_box2d_five_points.T)
                        pers_box2d_five_points = pers_box2d_five_points.T
                        # 归一化，并且变换为非齐次坐标
                        pers_box2d_five_points = coord_to_normal(pers_box2d_five_points)
                        pers_box2d_five_points = pers_box2d_five_points.reshape(1, -1)
                        pers_vertexes.append(pers_box2d_five_points)
                    pers_vertex = np.array(pers_vertexes).astype(np.float32).squeeze(axis=1)
                    vertexes[:, 0:10] = pers_vertex

            if rotation:
                rot_vertexes = []
                if len(vertexes) > 0:
                    rot_angle = np.random.uniform(-5, 5)
                    rot_theta = np.deg2rad(rot_angle)
                    rot_center = np.array([featmap_h / 2, featmap_w / 2]).reshape(2, 1)
                    # 绕中心点顺时针旋转
                    rot_matrix = np.array(
                        [np.cos(rot_theta), np.sin(rot_theta), -np.sin(rot_theta), np.cos(rot_theta)]).reshape(2, 2)
                    for i in range(vertexes.shape[0]):
                        # 5个点的格式(ct, kp)
                        box2d_five_points = np.array([vertexes[i, 0:10]])
                        rot_box2d_five_points = np.matmul(rot_matrix,
                                                          box2d_five_points.reshape(5, 2).T - rot_center) + rot_center
                        rot_box2d_five_points = rot_box2d_five_points.T.reshape(1, -1)
                        rot_vertexes.append(rot_box2d_five_points)
                    rot_vertex = np.array(rot_vertexes).astype(np.float32).squeeze(axis=1)
                    vertexes[:, 0:10] = rot_vertex
                    image = image.rotate(rot_angle)
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if len(vertexes) > 0:
                    vertexes[:, 0:10:2] = featmap_w - vertexes[:, 0:10:2]

            if color_change:
                image = self.apply_online_light_augmentation(image, apply_prob=0.5)
                # hue = self.rand(-hue, hue)
                # sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
                # val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
                # x = cv.cvtColor(np.array(image, np.float32) / 255, cv.COLOR_RGB2HSV)
                # x[..., 0] += hue * 360
                # x[..., 0][x[..., 0] > 1] -= 1
                # x[..., 0][x[..., 0] < 0] += 1
                # x[..., 1] *= sat
                # x[..., 2] *= val
                # x[x[:, :, 0] > 360, 0] = 360
                # x[:, :, 1:][x[:, :, 1:] > 1] = 1
                # x[x < 0] = 0
                # image = cv.cvtColor(x, cv.COLOR_HSV2RGB) * 255

        return image, vertexes, image_w, image_h


# DataLoader中collate_fn使用
def centernet_dataset_collate(batch):
    imgs, batch_hms, batch_vertexes_hms, batch_regs_ct_offset, batch_regs_vt_ct_offset, batch_reg_masks, batch_vertexes, batch_reg_trans_ms, raw_img_ws, raw_img_hs, seg_masks = [], [], [], [], [], [], [], [], [], [], []

    for img, batch_hm, batch_vertexes_hm, batch_reg_ct_offset, batch_reg_vt_ct_offset, batch_reg_mask, batch_vertex, batch_reg_trans_m, raw_img_w, raw_img_h, seg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_vertexes_hms.append(batch_vertexes_hm)
        batch_regs_ct_offset.append(batch_reg_ct_offset)
        batch_regs_vt_ct_offset.append(batch_reg_vt_ct_offset)
        batch_reg_masks.append(batch_reg_mask)
        batch_vertexes.append(batch_vertex)
        batch_reg_trans_ms.append(batch_reg_trans_m)
        raw_img_hs.append(raw_img_h)
        raw_img_ws.append(raw_img_w)
        seg_masks.append(seg_mask)

    imgs = np.array(imgs)
    batch_hms = np.array(batch_hms)
    batch_vertexes_hms = np.array(batch_vertexes_hms)
    batch_regs_ct_offset = np.array(batch_regs_ct_offset)
    batch_regs_vt_ct_offset = np.array(batch_regs_vt_ct_offset)
    batch_reg_masks = np.array(batch_reg_masks)
    batch_vertexes = np.array(batch_vertexes)
    batch_reg_trans_ms = np.array(batch_reg_trans_ms)
    raw_img_hs = np.array(raw_img_hs)
    raw_img_ws = np.array(raw_img_ws)
    seg_masks = np.array(seg_masks)
    return imgs, batch_hms, batch_vertexes_hms, batch_regs_ct_offset, batch_regs_vt_ct_offset, batch_reg_masks, batch_vertexes, batch_reg_trans_ms, raw_img_ws, raw_img_hs, seg_masks


def centernet_dataset_collate_poly_two_branch(batch):
    imgs, batch_hms, batch_vertexes_hms, batch_regs_ct_offset, batch_regs_vt_ct_offset, batch_regs_vt_ct_offset_angle, batch_reg_masks, batch_vertexes, batch_reg_trans_ms, raw_img_ws, raw_img_hs, seg_masks = [], [], [], [], [], [], [], [], [], [], [], []

    for img, batch_hm, batch_vertexes_hm, batch_reg_ct_offset, batch_reg_vt_ct_offset, batch_reg_vt_ct_offset_angle, batch_reg_mask, batch_vertex, batch_reg_trans_m, raw_img_w, raw_img_h, seg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_vertexes_hms.append(batch_vertexes_hm)
        batch_regs_ct_offset.append(batch_reg_ct_offset)
        batch_regs_vt_ct_offset.append(batch_reg_vt_ct_offset)
        batch_regs_vt_ct_offset_angle.append(batch_reg_vt_ct_offset_angle)
        batch_reg_masks.append(batch_reg_mask)
        batch_vertexes.append(batch_vertex)
        batch_reg_trans_ms.append(batch_reg_trans_m)
        raw_img_hs.append(raw_img_h)
        raw_img_ws.append(raw_img_w)
        seg_masks.append(seg_mask)

    imgs = np.array(imgs)
    batch_hms = np.array(batch_hms)
    batch_vertexes_hms = np.array(batch_vertexes_hms)
    batch_regs_ct_offset = np.array(batch_regs_ct_offset)
    batch_regs_vt_ct_offset = np.array(batch_regs_vt_ct_offset)
    batch_regs_vt_ct_offset_angle = np.array(batch_regs_vt_ct_offset_angle)
    batch_reg_masks = np.array(batch_reg_masks)
    batch_vertexes = np.array(batch_vertexes)
    batch_reg_trans_ms = np.array(batch_reg_trans_ms)
    raw_img_hs = np.array(raw_img_hs)
    raw_img_ws = np.array(raw_img_ws)
    seg_masks = np.array(seg_masks)
    return imgs, batch_hms, batch_vertexes_hms, batch_regs_ct_offset, batch_regs_vt_ct_offset, batch_regs_vt_ct_offset_angle, batch_reg_masks, batch_vertexes, batch_reg_trans_ms, raw_img_ws, raw_img_hs, seg_masks
