import colorsys
import copy
import os
import time
import cv2 as cv

import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont, ImageOps

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_bbox, postprocess
from utils.dataloader import gaussian_radius, draw_gaussian


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class CenterNet(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'logs/c-8779.pth',
        "classes_path": 'model_data/classes_dpm.txt',
        # --------------------------------------------------------------------------#
        #   用于选择所使用的模型的主干
        #   resnet50, hourglass
        # --------------------------------------------------------------------------#
        "backbone": 'resnet50',
        # --------------------------------------------------------------------------#
        #   输入图片的大小，设置成32的倍数
        # --------------------------------------------------------------------------#
        "input_shape": [512, 512],
        # --------------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # --------------------------------------------------------------------------#
        "confidence": 0.3,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # --------------------------------------------------------------------------#
        #   是否进行非极大抑制，可以根据检测效果自行选择
        #   backbone为resnet50时建议设置为True、backbone为hourglass时建议设置为False
        # --------------------------------------------------------------------------#
        "nms": True,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
        "poly_setting": {"poly": True, "two_offset_branch": False, "direct_theta": True}
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化centernet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   计算总的类的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        assert self.backbone in ['resnet50', 'hourglass']
        if self.backbone == "resnet50":
            self.net = CenterNet_Resnet50(num_classes=self.num_classes, pretrained=False,
                                          poly_setting=self.poly_setting)
        else:
            self.net = CenterNet_HourglassNet({'hm': self.num_classes, 'wh': 2, 'reg': 2})

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        raw_img_w, raw_img_h = image.size
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        raw_img = cv.cvtColor(np.array(image_data), cv.COLOR_RGB2BGR)  # 缩放后的原图像

        # letter_img = cv.cvtColor(np.asarray(image_data), cv.COLOR_RGB2BGR)
        # cv.imshow("heat", letter_img)

        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            t1 = time.time()
            outputs = self.net(images)

            #
            # hotmaps = outputs[1][0].cpu().numpy().transpose(1, 2, 0)[..., 0]
            #
            # import matplotlib.pyplot as plt
            #
            # heatmap = np.maximum(hotmaps, 0)
            # heatmap /= np.max(heatmap)
            #
            # heatmap = cv.resize(heatmap, (self.input_shape[0], self.input_shape[1]))
            # heatmap = np.uint8(255 * heatmap)
            # heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
            #
            # superimposed_img = heatmap
            #
            # cv.imshow("heatmap", superimposed_img)

            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decode_bbox(outputs, self.confidence, self.cuda, self.poly_setting)

            # -------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            # -------------------------------------------------------#
            outputs = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            # --------------------------------------#
            #   如果没有检测到物体，则返回原图
            # --------------------------------------#
            if outputs[0] is None:
                t2 = time.time()
                return image, raw_img, 1 / (t2 - t1)

            top_label = outputs[0][:, 11].cpu().detach().numpy().astype("int")
            top_conf = outputs[0][:, 10].cpu().detach().numpy()
            top_boxes = outputs[0][:, :10].cpu().detach().numpy()

            top_boxes_f = copy.deepcopy(top_boxes)

            if raw_img_h >= raw_img_w:
                top_boxes[:, 0:10:2] = top_boxes[:, 0:10:2] * max(raw_img_w, raw_img_h) - abs(
                    raw_img_w - raw_img_h) / 2.
                top_boxes[:, 1:10:2] = top_boxes[:, 1:10:2] * max(raw_img_w, raw_img_h)
            else:
                top_boxes[:, 0:10:2] = top_boxes[:, 0:10:2] * max(raw_img_w, raw_img_h)
                top_boxes[:, 1:10:2] = top_boxes[:, 1:10:2] * max(raw_img_w, raw_img_h) - abs(
                    raw_img_w - raw_img_h) / 2.

            t2 = time.time()

        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/Times New Roman.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        vertexes_hm = np.zeros((self.input_shape[0] // 2, self.input_shape[1] // 2, self.num_classes), dtype=np.float32)

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            box = np.array(box, dtype=np.int64)
            box_f = self.input_shape[0] // 2 * np.array(top_boxes_f[i])  # 相对于128*128特征图的尺寸
            ct_x, ct_y, pt_x1, pt_y1, pt_x2, pt_y2, pt_x3, pt_y3, pt_x4, pt_y4 = box

            label = '{} {:.2f}'.format(predicted_class, score)
            label = label.encode("utf-8")
            draw = ImageDraw.Draw(image)

            draw.text((ct_x, ct_y), str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            draw.ellipse((ct_x - 3, ct_y - 3, ct_x + 3, ct_y + 3), outline=(0, 0, 255), width=thickness)

            draw.line([pt_x1, pt_y1, pt_x2, pt_y2], fill=self.colors[c], width=thickness + 4)
            draw.line([pt_x2, pt_y2, pt_x3, pt_y3], fill=self.colors[c], width=thickness + 4)
            draw.line([pt_x3, pt_y3, pt_x4, pt_y4], fill=self.colors[c], width=thickness + 4)
            draw.line([pt_x4, pt_y4, pt_x1, pt_y1], fill=self.colors[c], width=thickness + 4)

            # TODO 绘制关键点热力图
            h, w = np.max(box_f[3:10:2]) - np.min(box_f[3:10:2]), np.max(box_f[2:10:2]) - np.min(box_f[2:10:2])
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                # 关键点热力图
                vertexes_hm[:, :, int(c)] = draw_gaussian(vertexes_hm[:, :, int(c)], box_f[2:4], radius)
                vertexes_hm[:, :, int(c)] = draw_gaussian(vertexes_hm[:, :, int(c)], box_f[4:6], radius)
                vertexes_hm[:, :, int(c)] = draw_gaussian(vertexes_hm[:, :, int(c)], box_f[6:8], radius)
                vertexes_hm[:, :, int(c)] = draw_gaussian(vertexes_hm[:, :, int(c)], box_f[8:10], radius)

            del draw

        vertex_hotmaps = vertexes_hm[..., 0]  # cls_index
        vertex_heatmap = np.maximum(vertex_hotmaps, 0)
        vertex_heatmap /= np.max(vertex_heatmap)
        # pseudo color map vertex
        vertex_hotmaps = cv.resize(vertex_hotmaps, (512, 512))
        vertex_hotmaps = np.uint8(255 * vertex_hotmaps)
        vertex_hotmaps = cv.applyColorMap(vertex_hotmaps, cv.COLORMAP_JET)

        superimposed_img = vertex_hotmaps * 0.5 + raw_img * 0.5  # BGR
        superimposed_img = np.array(superimposed_img, dtype=np.uint8)

        return image, superimposed_img, 1 / (t2 - t1)

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                outputs = self.net(images)

                # -----------------------------------------------------------#
                #   利用预测结果进行解码
                # -----------------------------------------------------------#
                outputs = decode_bbox(outputs, self.confidence, self.cuda, self.poly_setting)

                # -------------------------------------------------------#
                #   对于centernet网络来讲，确立中心非常重要。
                #   对于大目标而言，会存在许多的局部信息。
                #   此时对于同一个大目标，中心点比较难以确定。
                #   使用最大池化的非极大抑制方法无法去除局部框
                #   所以我还是写了另外一段对框进行非极大抑制的代码
                #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
                # -------------------------------------------------------#
                outputs = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image,
                                      self.nms_iou)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        raw_img_h, raw_img_w = image_shape
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            t1 = time.time()
            outputs = self.net(images)
            t2 = time.time()

            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decode_bbox(outputs, self.confidence, self.cuda, self.poly_setting)

            # -------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            # -------------------------------------------------------#
            outputs = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            # --------------------------------------#
            #   如果没有检测到物体，则返回原图
            # --------------------------------------#
            if outputs[0] is None:
                return 1 / (t2 - t1)

            top_label = outputs[0][:, 11].cpu().detach().numpy().astype("int")
            top_conf = outputs[0][:, 10].cpu().detach().numpy()
            top_boxes = outputs[0][:, :10].cpu().detach().numpy()

            if raw_img_h >= raw_img_w:
                top_boxes[:, 0:10:2] = top_boxes[:, 0:10:2] * max(raw_img_w, raw_img_h) - abs(
                    raw_img_w - raw_img_h) / 2.
                top_boxes[:, 1:10:2] = top_boxes[:, 1:10:2] * max(raw_img_w, raw_img_h)
            else:
                top_boxes[:, 0:10:2] = top_boxes[:, 0:10:2] * max(raw_img_w, raw_img_h)
                top_boxes[:, 1:10:2] = top_boxes[:, 1:10:2] * max(raw_img_w, raw_img_h) - abs(
                    raw_img_w - raw_img_h) / 2.

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            box = np.array(box, dtype=np.int64)
            ct_x, ct_y, pt_x1, pt_y1, pt_x2, pt_y2, pt_x3, pt_y3, pt_x4, pt_y4 = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s %s %s %s %s\n" % (predicted_class, score[:6],
                                                         str(int(pt_x1)), str(int(pt_y1)),
                                                         str(int(pt_x2)), str(int(pt_y2)),
                                                         str(int(pt_x3)), str(int(pt_y3)),
                                                         str(int(pt_x4)), str(int(pt_y4))))

        f.close()
        return 1 / (t2 - t1)
