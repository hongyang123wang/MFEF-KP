import numpy as np
from PIL import Image
import torch
import shapely
from shapely.geometry import Polygon, MultiPoint


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def preprocess_input(image):
    image   = np.array(image,dtype = np.float32)[:, :, ::-1]
    mean    = [0.40789655, 0.44719303, 0.47026116]
    std     = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255. - mean) / std


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_point_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def poly_iou(point_set_a, point_set_b):
    point_set_a = list(map(tuple, np.array(point_set_a.cpu().detach().numpy()).reshape([-1, 2])))
    point_set_b = list(map(tuple, np.array(point_set_b.cpu().detach().numpy()).reshape([-1, 2])))

    poly1 = Polygon(point_set_a).convex_hull
    poly2 = Polygon(point_set_b).convex_hull

    union_poly = np.concatenate((point_set_a, point_set_b))

    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = MultiPoint(union_poly).convex_hull.area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print("shapely.geos.TopologicalError occured, iou set to 0")
            iou = 0

    return iou


def poly_iou_map(point_set_a, point_set_b):
    point_set_a = np.array(list(map(tuple, np.array(point_set_a).reshape([-1, 2]))))
    point_set_b = np.array(list(map(tuple, np.array(point_set_b).reshape([-1, 2]))))

    poly1 = Polygon(point_set_a).convex_hull
    poly2 = Polygon(point_set_b).convex_hull

    union_poly = np.concatenate((point_set_a, point_set_b))

    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = MultiPoint(union_poly).convex_hull.area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print("shapely.geos.TopologicalError occured, iou set to 0")
            iou = 0

    return iou


def coord_to_homog(points):
    # 将普通坐标转换为齐次表示
    # points: [num_point, [pt_x, pt_y]]
    return np.hstack((points, np.ones((points.shape[0], 1))))


def coord_to_normal(points):
    # 将齐次坐标归一化，并转换为普通坐标
    # points: [num_point, [pt_x, pt_y]]
    points[:, 0] = points[:, 0] / points[:, 2]
    points[:, 1] = points[:, 1] / points[:, 2]
    return np.delete(points, 2, axis=1)


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
