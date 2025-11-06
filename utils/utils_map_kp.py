import json
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_image_id_map(gt_path, dr_path):
    """构建全局唯一的整数ID映射表"""
    all_files = set()
    for f in os.listdir(gt_path):
        if f.endswith(".txt"):
            all_files.add(os.path.splitext(f)[0])
    for f in os.listdir(dr_path):
        if f.endswith(".txt"):
            all_files.add(os.path.splitext(f)[0])
    return {name: idx + 1 for idx, name in enumerate(sorted(all_files))}


def preprocess_gt(gt_path, class_names, id_map):
    image_ids = os.listdir(gt_path)
    results = {
        "info": {"description": "COCO keypoint dataset", "version": "1.0"},  # added info field
        "images": [],
        "categories": [],
        "annotations": []
    }

    # 构建类别
    results["categories"] = [
        {"id": i + 1, "name": cls, "supercategory": "none"}
        for i, cls in enumerate(class_names)
    ]

    # 定义关键点信息
    num_keypoints = 4  # 四个顶点
    keypoint_names = ["point1", "point2", "point3", "point4"]

    # 更新类别信息，添加关键点定义
    for cat in results["categories"]:
        cat["keypoints"] = keypoint_names
        cat["skeleton"] = [[1, 2], [2, 3], [3, 4], [4, 1]]  # 连接顺序

    # 图像路径（根据实际情况修改）
    JPEG_PATH = "VOCdevkit/VOC2007/JPEGImages/"

    for filename in image_ids:
        if not filename.endswith(".txt"):
            continue

        base_id = os.path.splitext(filename)[0]

        # 获取全局唯一ID
        if base_id not in id_map:
            print(f"警告: {base_id} 未在ID映射表中，跳过处理")
            continue
        image_id = id_map[base_id]

        # 获取图像尺寸
        img_path = os.path.join(JPEG_PATH, base_id + ".jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
            else:
                width, height = 512, 512  # 默认值
        else:
            width, height = 640, 480  # 默认值

        # 添加图像信息
        results["images"].append({
            "id": image_id,
            "file_name": base_id + ".jpg",
            "width": width,
            "height": height
        })

        # 解析标注
        with open(os.path.join(gt_path, filename), "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:  # 1类别+8坐标
                continue

            class_name = parts[0]
            coords = list(map(float, parts[1:9]))

            # 转换为4个点
            keypoints = []
            for i in range(0, 8, 2):
                # 每个关键点格式: [x, y, v] 其中v是可见性标志(0:不可见, 1:可见但标注不准, 2:可见且标注准确)
                keypoints.extend([coords[i], coords[i + 1], 2])  # 假设所有点都是可见的

            # 计算外接矩形用于bbox字段
            points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            width_box = x_max - x_min
            height_box = y_max - y_min
            area = width_box * height_box

            # 获取类别ID
            if class_name not in class_names:
                print(f"警告: 类别 '{class_name}' 不在类别列表中，跳过")
                continue
            cls_id = class_names.index(class_name) + 1

            results["annotations"].append({
                "id": len(results["annotations"]) + 1,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [x_min, y_min, width_box, height_box],
                "area": area,
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": num_keypoints
            })

    return results


def preprocess_dr(dr_path, class_names, id_map):
    results = []

    for filename in os.listdir(dr_path):
        if not filename.endswith(".txt"):
            continue

        base_id = os.path.splitext(filename)[0]

        # 获取全局唯一ID
        if base_id not in id_map:
            print(f"警告: {base_id} 未在ID映射表中，跳过处理")
            continue
        image_id = id_map[base_id]

        with open(os.path.join(dr_path, filename), "r") as f:
            lines = f.readlines()

        # 处理空检测结果
        if not lines:
            print(f"警告: {filename} 无检测结果，添加空记录")
            # 空检测结果时添加一个占位的关键点记录
            dummy_keypoints = [0, 0, 0] * 4  # 4个不可见的关键点
            results.append({
                "image_id": image_id,
                "category_id": 1,  # 虚拟类别
                "keypoints": dummy_keypoints,
                "bbox": dummy_keypoints,
                "score": 0.0
            })
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 10:  # 1类别+1置信度+8坐标
                continue

            class_name = parts[0]
            confidence = float(parts[1])
            coords = list(map(float, parts[2:10]))

            # 转换为关键点格式
            keypoints = []
            for i in range(0, 8, 2):
                keypoints.extend([coords[i], coords[i + 1], 2])  # 假设所有点都可见

            # 计算外接矩形用于bbox字段
            points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, y_min = min(xs), min(ys)
            width_box = max(xs) - x_min
            height_box = max(ys) - y_min

            # 获取类别ID
            if class_name not in class_names:
                print(f"警告: 类别 '{class_name}' 不在类别列表中，跳过")
                continue
            cls_id = class_names.index(class_name) + 1

            results.append({
                "image_id": image_id,
                "category_id": cls_id,
                "keypoints": keypoints,
                "bbox": [x_min, y_min, width_box, height_box],  # 可选，但有助于评估
                "score": confidence
            })

    return results


def get_coco_keypoint_map(class_names, path, return_metrics=False):
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    COCO_PATH = os.path.join(path, 'coco_eval')

    os.makedirs(COCO_PATH, exist_ok=True)
    GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_keypoints_gt.json')
    DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_keypoints_dr.json')

    # 1. 生成 JSON
    id_map = build_image_id_map(GT_PATH, DR_PATH)
    with open(GT_JSON_PATH, 'w') as f:
        results_gt = preprocess_gt(GT_PATH, class_names, id_map)
        json.dump(results_gt, f)
    with open(DR_JSON_PATH, 'w') as f:
        results_dr = preprocess_dr(DR_PATH, class_names, id_map)
        json.dump(results_dr, f)
    # 2. 计算 OKS
    cocoGt = COCO(GT_JSON_PATH)
    cocoDt = cocoGt.loadRes(DR_JSON_PATH)
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    # 3. 关键：把 sigmas 换成 4 个，长度必须 = num_keypoints
    num_keypoints = 4
    # 可以按自己数据的重要性给不同权重，这里直接用等权重
    cocoEval.params.kpt_oks_sigmas = np.ones(num_keypoints) * 0.1

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if return_metrics:
        stats = cocoEval.stats
        return {
            "AP_50_95": stats[0],  # AP @ OKS=0.50:0.95
            "AP_50": stats[1],  # AP @ OKS=0.50
            "AP_75": stats[2],  # AP @ OKS=0.75
            "AP_medium": stats[3],
            "AP_large": stats[4],
            "AR_50_95": stats[5],  # AR @ OKS=0.50:0.95
            "AR_50": stats[6],  # AR @ OKS=0.50
            "AR_75": stats[7],  # AR @ OKS=0.75
            "AR_medium": stats[8],
            "AR_large": stats[9],
        }
    return None
