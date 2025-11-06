import os
import shutil
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
# from utils.utils import poly_iou

import torch.nn.functional as F

from utils.utils import poly_iou


def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_bbox(outputs, confidence, cuda, poly_setting=None):
    # -------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128,
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    # -------------------------------------------------------------------------#
    if poly_setting is None:
        poly_setting = {"poly": False, "two_offset_branch": False, "direct_theta": False}
    poly = poly_setting["poly"]
    two_offset_branch = poly_setting["two_offset_branch"]
    direct_theta = poly_setting["direct_theta"]

    (pred_hms, pred_offsets_ct, pred_offsets_vt_ct, kphm), fg = outputs

    pred_hms = pool_nms(pred_hms)

    def save_feature_map_nomal(feature_map, base_name, save_dir="output_maps"):

        # 2. 生成毫秒级时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{base_name}_{timestamp}.png"
        full_path = os.path.join(save_dir, filename)

        # 3. 归一化并保存
        feat = feature_map.detach().cpu().numpy().squeeze()
        feat = ((feat - feat.min()) / (feat.max() - feat.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(full_path, feat)
        print(f"[Saved] {full_path}")

    def save_feature_map(feature_map, base_name, save_dir="output_maps"):
        # 2. 生成毫秒级时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{base_name}_{timestamp}.png"
        full_path = os.path.join(save_dir, filename)

        # 3. 归一化并应用色彩映射
        feat = feature_map.detach().cpu().numpy().squeeze()
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)  # 归一化到0-1

        # 应用jet色彩映射
        colormap = plt.get_cmap('coolwarm')
        colored_feat = colormap(feat)  # shape: (H, W, 4)

        # 转换为BGR格式并缩放到0-255
        colored_feat_bgr = colored_feat[..., ::-1][:, :, :3]  # RGB转BGR，去掉alpha通道
        colored_feat_bgr = (colored_feat_bgr * 255).astype(np.uint8)

        cv2.imwrite(full_path, colored_feat_bgr)
        print(f"[Saved] {full_path}")

    # 保存为PNG文件
    # save_feature_map(fg, "kp")
    # save_feature_map(kphm, "kphm")
    # save_feature_map_nomal(kphm, "kphm")
    # save_feature_map_nomal(pred_hms, "pred_hms")

    b, c, output_h, output_w = pred_hms.shape
    detects = []
    # -------------------------------------------------------------------------#
    #   只传入一张图片，循环只进行一次
    # -------------------------------------------------------------------------#
    for batch in range(b):
        # -------------------------------------------------------------------------#
        #   heat_map            128*128, num_classes    热力图
        #   vertex_heat_map     128*128, num_classes    关键点热力图
        #   pred_offsets_ct     128*128, 2              中心点的xy轴偏移坐标
        #   pred_offsets_vt_ct  128*128, 8              关键点相对于中心点的xy轴偏移坐标
        # -------------------------------------------------------------------------#
        heat_map = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        pred_offset_ct = pred_offsets_ct[batch].permute(1, 2, 0).view([-1, 2])
        if not poly:
            pred_offset_vt_ct = pred_offsets_vt_ct[batch].permute(1, 2, 0).view([-1, 8])
        else:
            if direct_theta:
                pred_offset_vt_ct = pred_offsets_vt_ct[batch].permute(1, 2, 0).view([-1, 8])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))

        # -------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        # -------------------------------------------------------------------------#
        xv, yv = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        # -------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        # -------------------------------------------------------------------------#
        class_conf, class_pred = torch.max(heat_map, dim=-1)
        mask = class_conf > confidence

        # -----------------------------------------#
        #   取出得分筛选后对应的结果
        # -----------------------------------------#
        pred_offset_ct_mask = pred_offset_ct[mask]
        pred_offset_vt_ct_mask = pred_offset_vt_ct[mask]

        if len(pred_offset_ct_mask) == 0:
            detects.append([])
            continue

        # ----------------------------------------#
        #   计算调整后的关键点 (1 + 4)
        # ----------------------------------------#
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_ct_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_ct_mask[..., 1], -1)

        xv_mask_1 = torch.unsqueeze(
            xv[mask] + pred_offset_vt_ct_mask[..., 0] * torch.cos(pred_offset_vt_ct_mask[..., 1]), -1)
        yv_mask_1 = torch.unsqueeze(
            yv[mask] + pred_offset_vt_ct_mask[..., 0] * torch.sin(pred_offset_vt_ct_mask[..., 1]), -1)
        xv_mask_2 = torch.unsqueeze(
            xv[mask] + pred_offset_vt_ct_mask[..., 2] * torch.cos(pred_offset_vt_ct_mask[..., 3]), -1)
        yv_mask_2 = torch.unsqueeze(
            yv[mask] + pred_offset_vt_ct_mask[..., 2] * torch.sin(pred_offset_vt_ct_mask[..., 3]), -1)
        xv_mask_3 = torch.unsqueeze(
            xv[mask] + pred_offset_vt_ct_mask[..., 4] * torch.cos(pred_offset_vt_ct_mask[..., 5]), -1)
        yv_mask_3 = torch.unsqueeze(
            yv[mask] + pred_offset_vt_ct_mask[..., 4] * torch.sin(pred_offset_vt_ct_mask[..., 5]), -1)
        xv_mask_4 = torch.unsqueeze(
            xv[mask] + pred_offset_vt_ct_mask[..., 6] * torch.cos(pred_offset_vt_ct_mask[..., 7]), -1)
        yv_mask_4 = torch.unsqueeze(
            yv[mask] + pred_offset_vt_ct_mask[..., 6] * torch.sin(pred_offset_vt_ct_mask[..., 7]), -1)
        # ----------------------------------------#
        #   获得预测结果
        # ----------------------------------------#
        bboxes = torch.cat(
            [xv_mask, yv_mask, xv_mask_1, yv_mask_1, xv_mask_2, yv_mask_2, xv_mask_3, yv_mask_3, xv_mask_4, yv_mask_4],
            dim=1)
        # 归一化至0-1之间
        bboxes[:, 0::2] /= output_w
        bboxes[:, 1::2] /= output_h
        detect = torch.cat(
            [bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask], -1).float()], dim=-1)
        detects.append(detect)

    return detects


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / torch.clamp(b1_area + b2_area - inter_area, min=1e-6)

    return iou


def centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        # -----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        # -----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def postprocess(prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]

    # ----------------------------------------------------------#
    #   预测只用一张图片，只会进行一次
    # ----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        detections = prediction[i]
        if len(detections) == 0:
            continue
        # ------------------------------------------#
        #   获得预测结果中包含的所有种类
        # ------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()

        if detections.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            # ------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            # ------------------------------------------#
            detections_class = detections[detections[:, -1] == c]
            if need_nms:
                # ------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                # ------------------------------------------#
                # keep = nms(
                #     detections_class[:, :4],
                #     detections_class[:, 4],
                #     nms_thres
                # )
                # max_detections = detections_class[keep]

                # ------------------------------------------#
                #   按照存在物体的置信度排序
                # ------------------------------------------#
                _, conf_sort_index = torch.sort(detections_class[:, -2], descending=True)
                detections_class = detections_class[conf_sort_index]
                # ------------------------------------------#
                #   进行非极大抑制
                # ------------------------------------------#
                max_detections = []

                while detections_class.size(0):
                    ious = []
                    # ---------------------------------------------------#
                    #   取出这一类置信度最高的，一步一步往下判断。
                    #   判断重合程度是否大于nms_thres，如果是则去除掉
                    # ---------------------------------------------------#
                    max_detections.append(detections_class[0].unsqueeze(0))
                    if len(detections_class) == 1:
                        break
                    # ious = bbox_iou(max_detections[-1], detections_class[1:])
                    for idx in range(1, len(detections_class)):
                        single_iou = poly_iou(max_detections[-1][0][2:10], detections_class[idx][2:10])
                        ious.append(single_iou)
                    ious = torch.tensor(ious).cuda()

                    detections_class = detections_class[1:][ious < nms_thres]
                    # detections_class = detections_class[1:]
                # ------------------------------------------#
                #   堆叠
                # ------------------------------------------#
                max_detections = torch.cat(max_detections).data
            else:
                max_detections = detections_class

            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
    return output
