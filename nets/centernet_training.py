import torch
import torch.nn.functional as F

import cv2 as cv
import math
import numpy as np

from utils.utils import poly_iou
from utils.dataloader import gaussian_radius, draw_gaussian


def focal_loss(pred, target,epoch=None, max_epoch=240):
    if pred.shape[1] != target.shape[1]:
        pred = pred.permute(0, 2, 3, 1)

    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


# def focal_loss(pred, target, epoch=None, max_epoch=240,
#                   alpha_base=0.8, gamma_base=2.0, neg_ratio_base=3.0, bg_threshold_base=0.1):
#     """
#     函数式 Focal Loss + Hard Negative Mining
#     支持动态课程学习（Curriculum Learning），自动渐进式收紧约束

#     Args:
#         pred: [B, H, W]，模型输出经过 sigmoid 后的概率热图
#         target: [B, H, W]，GT 热图（0~1 连续值）
#         epoch: 当前 epoch，用于动态调整参数（可选）
#         max_epoch: 总 epoch 数，用于计算进度（可选）
#         alpha_base: 最终 alpha 值，默认 0.8
#         gamma_base: 最终 gamma 值，默认 2.0
#         neg_ratio_base: 最终负样本采样比例，默认 3.0
#         bg_threshold_base: 最终背景阈值，默认 0.1

#     Returns:
#         loss: 标量损失值
#     """
#     if pred.shape[1] != target.shape[1]:
#         pred = pred.permute(0, 2, 3, 1)
#     # print("pred shape:", pred.shape)
#     B, _, H, W = pred.shape
#     pred = torch.clamp(pred, 1e-6, 1 - 1e-6)  # 数值稳定

#     # ========== 动态参数计算（课程学习） ==========
#     if epoch is not None and max_epoch is not None:
#         progress = min(epoch / max_epoch, 1.0)
#         # 平滑渐进：初期宽松，后期收紧
#         alpha = 0.6 + (alpha_base - 0.6) * progress
#         gamma = 1.0 + (gamma_base - 1.0) * progress
#         neg_ratio = 2.0 + (neg_ratio_base - 2.0) * progress
#         bg_threshold = 0.2 - (0.2 - bg_threshold_base) * progress
#     else:
#         # 使用基础参数（适合验证/测试/固定参数训练）
#         alpha = alpha_base
#         gamma = gamma_base
#         neg_ratio = neg_ratio_base
#         bg_threshold = bg_threshold_base

#     # ========== 构建掩码 ==========
#     pos_mask = target.gt(0.5).float()          # 前景区域（高斯中心）
#     bg_mask = target.le(bg_threshold).float()   # 真背景区（低于阈值）
#     num_pos = pos_mask.sum()

#     # ========== Focal Loss 计算 ==========
#     # 正样本损失：-α * (1-p)^γ * log(p)
#     pos_loss = -alpha * torch.pow(1 - pred, gamma) * torch.log(pred) * pos_mask

#     # 负样本基础损失（用于 Hard Mining）：-(1-α) * p^γ * log(1-p)
#     neg_focal = -(1 - alpha) * torch.pow(pred, gamma) * torch.log(1 - pred)

#     # ========== Hard Negative Mining ==========
#     # 在真背景区中，找出“预测值最高”的难负样本
#     neg_score = pred * bg_mask  # 背景中预测值越高 → 越难
#     neg_score_flat = neg_score.view(B, -1)

#     # 动态采样数量，防止越界
#     num_neg = min(int(num_pos * neg_ratio + 1), H * W - 1)

#     if num_neg > 0:
#         # 选 top-k 难负样本
#         _, topk_indices = torch.topk(neg_score_flat, num_neg, dim=1)  # [B, K]
#         # gather 对应位置的 loss
#         neg_focal_flat = neg_focal.view(B, -1)  # [B, H*W]
#         neg_loss = torch.gather(neg_focal_flat, 1, topk_indices).sum()
#     else:
#         neg_loss = torch.tensor(0.0, device=pred.device)

#     # ========== 损失归一化 ==========
#     total_loss = pos_loss.sum() + neg_loss
#     loss = total_loss / (num_pos + 1e-6)

#     return loss


def reg_l1_loss(pred, target, mask, expand_dim):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    if pred.shape[1] != pred.shape[2]:
        pred = pred.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, expand_dim)

    loss = F.smooth_l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


import torch
import torch.nn.functional as F

def back_vertex_loss(offset_vt_ct, gt_vertex_hm, gt_vertex, mask, expand_dim, poly, two_offset_branch, direct_angle,epoch):
    """
    改进版 vertex loss:
      - 移除无效的 heatmap focal loss（因为 back_vertex_hms 不是网络输出，不可导）
      - 保留并优化 L1 offset loss（实际使用 Smooth L1）
      - 新增 OKS Loss（尺度归一化的关键点相似度损失）
      - ✅ 新增 Edge Length Loss（边长一致性约束）
      - ✅ 新增 Angle Loss（角度一致性约束）
      - 所有新增损失加权合并进返回值，保持接口 (l1_loss, oks_loss) 不变

    返回: (l1_loss, oks_loss) —— 兼容原有调用
    """
    if not two_offset_branch or not poly:
        if offset_vt_ct.shape[1] != offset_vt_ct.shape[2]:
            offset_vt_ct = offset_vt_ct.permute(0, 2, 3, 1)
        batch_size, featmap_h, featmap_w = offset_vt_ct.shape[0:3]
    else:
        offset_vt_ct, offset_angle_vt_ct = offset_vt_ct
        if offset_vt_ct.shape[1] != offset_vt_ct.shape[2]:
            offset_vt_ct = offset_vt_ct.permute(0, 2, 3, 1)
        if offset_angle_vt_ct.shape[1] != offset_angle_vt_ct.shape[2]:
            offset_angle_vt_ct = offset_angle_vt_ct.permute(0, 2, 3, 1)
        batch_size, featmap_h, featmap_w = offset_vt_ct.shape[0:3]

    if gt_vertex.shape[1] != gt_vertex.shape[2]:
        gt_vertex = gt_vertex.permute(0, 2, 3, 1)

    batch_calc_vertex = torch.zeros((batch_size, featmap_h, featmap_w, expand_dim), device=offset_vt_ct.device)
    batch_gt_vertex = torch.zeros((batch_size, featmap_h, featmap_w, expand_dim), device=offset_vt_ct.device)

    # 用于 OKS 计算
    all_d_sq = []   # 所有点的 d^2
    all_s_sq = []   # 对应点的 s^2（目标尺度平方）

    # 用于 Edge 和 Angle Loss（只在正样本上计算）
    pred_kpts_list = []
    gt_kpts_list = []

    for b in range(batch_size):
        obj_indices = torch.where(mask[b] == 1.0)  # foreground objects
        for y, x in zip(obj_indices[0], obj_indices[1]):
            # ========== 预测角点坐标 ==========
            if not poly:
                vertex_p = torch.tensor([x, y], device=offset_vt_ct.device, dtype=torch.float32) + offset_vt_ct[b, y, x]
            else:
                if two_offset_branch:
                    r = offset_vt_ct[b, y, x].reshape(-1, 1)
                    if direct_angle:
                        theta = offset_angle_vt_ct[b, y, x].reshape(-1, 1)
                        dx = r * torch.cos(theta)
                        dy = r * torch.sin(theta)
                    else:
                        sin_cos = offset_angle_vt_ct[b, y, x].reshape(-1, 2)
                        dx = r * sin_cos[:, 1:2]  # cos
                        dy = r * sin_cos[:, 0:1]  # sin
                else:
                    if direct_angle:
                        r_theta = offset_vt_ct[b, y, x].reshape(-1, 2)
                        dx = r_theta[:, 0:1] * torch.cos(r_theta[:, 1:2])
                        dy = r_theta[:, 0:1] * torch.sin(r_theta[:, 1:2])
                    else:
                        r_sin_cos = offset_vt_ct[b, y, x].reshape(-1, 3)
                        dx = r_sin_cos[:, 0:1] * r_sin_cos[:, 2:3]  # r * cos
                        dy = r_sin_cos[:, 0:1] * r_sin_cos[:, 1:2]  # r * sin
                offset = torch.cat([dx, dy], dim=1).reshape(-1)
                center = torch.tensor([x, y], device=offset_vt_ct.device, dtype=torch.float32).repeat(4, 1)  # [4, 2]
                offset_reshaped = offset.reshape(4, 2)  # [8] -> [4, 2]
                vertex_p = (center + offset_reshaped).reshape(-1)  # [4, 2] -> [8]

            vertex_gt = gt_vertex[b, y, x]

            batch_calc_vertex[b, y, x] = vertex_p
            batch_gt_vertex[b, y, x] = vertex_gt

            # ========== 计算 OKS 所需的 d^2 和 s^2 ==========
            pred_pts = vertex_p.reshape(-1, 2)      # [4, 2]
            gt_pts = vertex_gt.reshape(-1, 2)       # [4, 2]

            d_sq = torch.sum((pred_pts - gt_pts) ** 2, dim=1)  # [4]

            # 使用 GT 四边形的包围盒对角线长度作为尺度 s
            x_coords = gt_pts[:, 0]
            y_coords = gt_pts[:, 1]
            width = torch.max(x_coords) - torch.min(x_coords)
            height = torch.max(y_coords) - torch.min(y_coords)
            s_sq = width ** 2 + height ** 2 + 1e-8  # 防止为0

            all_d_sq.append(d_sq)
            all_s_sq.append(s_sq.repeat(4))  # 4个点共享同一个s

            # ========== 收集用于 Edge 和 Angle Loss 的点 ==========
            pred_kpts_list.append(pred_pts)
            gt_kpts_list.append(gt_pts)


    # print(f"📦 Processing batch with {len(pred_kpts_list)} samples")
    # if len(pred_kpts_list) > 0:
    #     pred_kpts = torch.stack(pred_kpts_list)
    #     gt_kpts = torch.stack(gt_kpts_list)
    #     print(f"🔍 Pred kpts range: [{pred_kpts.min().item():.3f}, {pred_kpts.max().item():.3f}]")
    #     print(f"🔍 GT kpts range: [{gt_kpts.min().item():.3f}, {gt_kpts.max().item():.3f}]")
    # ========== Smooth L1 Loss (主损失) ==========
    expand_mask = mask.unsqueeze(-1).expand_as(batch_calc_vertex)
    smooth_l1_loss = F.smooth_l1_loss(batch_calc_vertex * expand_mask, batch_gt_vertex * expand_mask, reduction='sum')
    smooth_l1_loss = smooth_l1_loss / (mask.sum() + 1e-4)

    # ========== OKS Loss ==========
    if epoch < 50:
        oks_loss = torch.tensor(0.0, device=offset_vt_ct.device)
    else:
        if len(all_d_sq) > 0:
            all_d_sq = torch.cat(all_d_sq)
            all_s_sq = torch.cat(all_s_sq)

            sigma = 0.15
            denominator = 2 * (sigma ** 2) * all_s_sq
            denominator = torch.clamp(denominator, min=1e-8)

            exponent = -all_d_sq / denominator
            exponent = torch.clamp(exponent, max=50)

            oks = torch.exp(exponent)
            oks = torch.clamp(oks, 0.0, 1.0)

            oks_loss = 1.0 - oks.mean()

    # ✅✅✅ 带 warmup 的 OKS 权重调度
    if epoch < 50:
        oks_weight = 0.0
    elif epoch < 70:  # 50~70 轮线性增加
        oks_weight = (epoch - 50) / 20.0
        oks_loss = oks_weight * (1.0 - oks.mean())
    else:
        oks_weight = 1.0
        oks_loss = oks_weight * (1.0 - oks.mean())

    # ========== Edge Loss ==========
    edge_loss = torch.tensor(0.0, device=offset_vt_ct.device)
    # if len(pred_kpts_list) > 0:
    #     pred_kpts = torch.stack(pred_kpts_list)  # [N, 4, 2]
    #     gt_kpts = torch.stack(gt_kpts_list)      # [N, 4, 2]

    #     edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    #     edge_losses = []

    #     for i, j in edges:
    #         diff_pred = pred_kpts[:, i] - pred_kpts[:, j]
    #         diff_gt = gt_kpts[:, i] - gt_kpts[:, j]

    #         len_pred = torch.norm(diff_pred, dim=1)
    #         len_gt = torch.norm(diff_gt, dim=1)

    #         loss_ij = F.l1_loss(len_pred, len_gt, reduction='none')
    #         edge_losses.append(loss_ij)

    #     edge_loss = torch.stack(edge_losses).mean()

    # # ========== Angle Loss (新增) ==========
    # angle_loss_val = torch.tensor(0.0, device=offset_vt_ct.device)
    # if len(pred_kpts_list) > 0:
    #     angle_losses = []
    #     for i in range(4):
    #         prev_i = (i - 1) % 4
    #         next_i = (i + 1) % 4

    #         vec_a_pred = pred_kpts[:, i] - pred_kpts[:, prev_i]  # [N, 2]
    #         vec_b_pred = pred_kpts[:, next_i] - pred_kpts[:, i]  # [N, 2]

    #         vec_a_gt = gt_kpts[:, i] - gt_kpts[:, prev_i]        # [N, 2]
    #         vec_b_gt = gt_kpts[:, next_i] - gt_kpts[:, i]        # [N, 2]

    #         def compute_angle(vec_a, vec_b):
    #             dot = torch.sum(vec_a * vec_b, dim=1)
    #             norm_a = torch.norm(vec_a, dim=1)
    #             norm_b = torch.norm(vec_b, dim=1)
    #             cos_angle = dot / (norm_a * norm_b + 1e-8)
    #             cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    #             angle = torch.acos(cos_angle)
    #             return angle

    #         angle_pred = compute_angle(vec_a_pred, vec_b_pred)
    #         angle_gt = compute_angle(vec_a_gt, vec_b_gt)

    #         loss_angle = F.l1_loss(angle_pred, angle_gt, reduction='none')
    #         angle_losses.append(loss_angle)

    #     angle_loss_val = torch.stack(angle_losses).mean()

    # ========== 合并损失（保持接口兼容）==========
    # 将 edge 和 angle loss 加权合并进 smooth_l1_loss（因为它们都是几何正则项）
    # 权重可调，初始建议：edge_weight=0.3, angle_weight=0.2
    edge_weight = 0.05
    # angle_weight = 0.2

    # ✅ 最终 l1_loss = smooth_l1 + edge_weight * edge_loss + angle_weight * angle_loss
    # l1_loss = smooth_l1_loss + edge_weight * edge_loss + angle_weight * angle_loss_val

    # ✅ 仍然返回 (l1_loss, oks_loss)，接口完全不变！
    # ========== ✅ NaN 保护 + 惩罚机制（分别处理）==========
    # PENALTY_EDGE = 2.0
    PENALTY_OKS = 1.5

    # if torch.isnan(edge_loss):
    #     print(f"⚠️  Edge Loss NaN! Applying penalty = {PENALTY_EDGE}")
    #     edge_loss = torch.tensor(PENALTY_EDGE, device=offset_vt_ct.device)

    if torch.isnan(oks_loss):
        print(f"⚠️  OKS Loss NaN! Applying penalty = {PENALTY_OKS}")
        oks_loss = torch.tensor(PENALTY_OKS, device=offset_vt_ct.device)

    # ========== ✅ 返回三个独立 loss 分量！==========
    return smooth_l1_loss, oks_loss, edge_weight * edge_loss , torch.tensor(0.0, device=offset_vt_ct.device) # angle_weight * angle_loss_val



# def back_vertex_loss(offset_vt_ct, gt_vertex_hm, gt_vertex, mask, expand_dim, poly, two_offset_branch, direct_angle):
#     """
#     改进版 vertex loss:
#       - 移除无效的 heatmap focal loss（因为 back_vertex_hms 不是网络输出，不可导）
#       - 保留并优化 L1 offset loss
#       - 新增 OKS Loss（尺度归一化的关键点相似度损失）
#       - 保持原函数参数不变

#     返回: (l1_loss, oks_loss)
#     """
#     if not two_offset_branch or not poly:
#         if offset_vt_ct.shape[1] != offset_vt_ct.shape[2]:
#             offset_vt_ct = offset_vt_ct.permute(0, 2, 3, 1)
#         batch_size, featmap_h, featmap_w = offset_vt_ct.shape[0:3]
#     else:
#         offset_vt_ct, offset_angle_vt_ct = offset_vt_ct
#         if offset_vt_ct.shape[1] != offset_vt_ct.shape[2]:
#             offset_vt_ct = offset_vt_ct.permute(0, 2, 3, 1)
#         if offset_angle_vt_ct.shape[1] != offset_angle_vt_ct.shape[2]:
#             offset_angle_vt_ct = offset_angle_vt_ct.permute(0, 2, 3, 1)
#         batch_size, featmap_h, featmap_w = offset_vt_ct.shape[0:3]

#     if gt_vertex.shape[1] != gt_vertex.shape[2]:
#         gt_vertex = gt_vertex.permute(0, 2, 3, 1)

#     batch_calc_vertex = torch.zeros((batch_size, featmap_h, featmap_w, expand_dim), device=offset_vt_ct.device)
#     batch_gt_vertex = torch.zeros((batch_size, featmap_h, featmap_w, expand_dim), device=offset_vt_ct.device)

#     # 用于 OKS 计算
#     all_d_sq = []   # 所有点的 d^2
#     all_s_sq = []   # 对应点的 s^2（目标尺度平方）

#     for b in range(batch_size):
#         obj_indices = torch.where(mask[b] == 1.0)  # foreground objects
#         for y, x in zip(obj_indices[0], obj_indices[1]):
#             # ========== 预测角点坐标 ==========
#             if not poly:
#                 vertex_p = torch.tensor([x, y], device=offset_vt_ct.device, dtype=torch.float32) + offset_vt_ct[b, y, x]
#             else:
#                 if two_offset_branch:
#                     r = offset_vt_ct[b, y, x].reshape(-1, 1)
#                     if direct_angle:
#                         theta = offset_angle_vt_ct[b, y, x].reshape(-1, 1)
#                         dx = r * torch.cos(theta)
#                         dy = r * torch.sin(theta)
#                     else:
#                         sin_cos = offset_angle_vt_ct[b, y, x].reshape(-1, 2)
#                         dx = r * sin_cos[:, 1:2]  # cos
#                         dy = r * sin_cos[:, 0:1]  # sin
#                 else:
#                     if direct_angle:
#                         r_theta = offset_vt_ct[b, y, x].reshape(-1, 2)
#                         dx = r_theta[:, 0:1] * torch.cos(r_theta[:, 1:2])
#                         dy = r_theta[:, 0:1] * torch.sin(r_theta[:, 1:2])
#                     else:
#                         r_sin_cos = offset_vt_ct[b, y, x].reshape(-1, 3)
#                         dx = r_sin_cos[:, 0:1] * r_sin_cos[:, 2:3]  # r * cos
#                         dy = r_sin_cos[:, 0:1] * r_sin_cos[:, 1:2]  # r * sin
#                 offset = torch.cat([dx, dy], dim=1).reshape(-1)
#                 center = torch.tensor([x, y], device=offset_vt_ct.device, dtype=torch.float32).repeat(4, 1)  # [4, 2]
#                 offset_reshaped = offset.reshape(4, 2)  # [8] -> [4, 2]
#                 vertex_p = (center + offset_reshaped).reshape(-1)  # [4, 2] -> [8]

#             vertex_gt = gt_vertex[b, y, x]

#             batch_calc_vertex[b, y, x] = vertex_p
#             batch_gt_vertex[b, y, x] = vertex_gt

#             # ========== 计算 OKS 所需的 d^2 和 s^2 ==========
#             pred_pts = vertex_p.reshape(-1, 2)      # [4, 2]
#             gt_pts = vertex_gt.reshape(-1, 2)       # [4, 2]

#             d_sq = torch.sum((pred_pts - gt_pts) ** 2, dim=1)  # [4]

#             # 使用 GT 四边形的包围盒对角线长度作为尺度 s
#             x_coords = gt_pts[:, 0]
#             y_coords = gt_pts[:, 1]
#             width = torch.max(x_coords) - torch.min(x_coords)
#             height = torch.max(y_coords) - torch.min(y_coords)
#             s_sq = width ** 2 + height ** 2 + 1e-8  # 防止为0

#             all_d_sq.append(d_sq)
#             all_s_sq.append(s_sq.repeat(4))  # 4个点共享同一个s

#     # ========== L1 Loss ==========
#     expand_mask = mask.unsqueeze(-1).expand_as(batch_calc_vertex)
#     l1_loss = F.smooth_l1_loss(batch_calc_vertex * expand_mask, batch_gt_vertex * expand_mask, reduction='sum')
#     l1_loss = l1_loss / (mask.sum() + 1e-4)

#     # ========== OKS Loss ==========
#     if len(all_d_sq) > 0:
#         all_d_sq = torch.cat(all_d_sq)  # [总点数]
#         all_s_sq = torch.cat(all_s_sq)  # [总点数]

#         sigma = 0.15  # 可调参数，控制容忍度（类似COCO OKS）
#         # OKS = exp(- d^2 / (2 * sigma^2 * s^2))
#         oks = torch.exp(-all_d_sq / (2 * (sigma ** 2) * all_s_sq))
#         oks_loss = 1.0 - oks.mean()  # 越相似，损失越小
#     else:
#         oks_loss = torch.tensor(0.0, device=offset_vt_ct.device)

#     # ✅ 返回 (L1 Loss, OKS Loss)，保持接口兼容性
#     # 原调用处如果是 loss, hm_loss = back_vertex_loss(...)，现在 hm_loss 就是 oks_loss
#     return l1_loss, oks_loss


def center_trans_loss(offset_ct, offset_vt_ct, trans_m, mask, raw_img_hs, raw_img_ws, poly, two_offset_branch, direct_angle, ct_only):
    return torch.tensor(0.0, device='cuda')
    # expand_dim = 4 if ct_only else 5
    # # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    # if offset_ct.shape[1] != offset_ct.shape[2]:
    #     offset_ct = offset_ct.permute(0, 2, 3, 1)
    # if not two_offset_branch or not poly:
    #     if offset_vt_ct.shape[1] != offset_vt_ct.shape[2]:
    #         offset_vt_ct = offset_vt_ct.permute(0, 2, 3, 1)
    #     batch_size, featmap_h, featmap_w = offset_ct.shape[0:3]
    # else:
    #     offset_vt_ct, offset_angle_vt_ct = offset_vt_ct
    #     if offset_vt_ct.shape[1] != offset_vt_ct.shape[2]:
    #         offset_vt_ct = offset_vt_ct.permute(0, 2, 3, 1)
    #     if offset_angle_vt_ct.shape[1] != offset_angle_vt_ct.shape[2]:
    #         offset_angle_vt_ct = offset_angle_vt_ct.permute(0, 2, 3, 1)
    #     batch_size, featmap_h, featmap_w = offset_vt_ct.shape[0:3]
    #
    # batch_calc_dis = torch.zeros((batch_size, featmap_h, featmap_w, expand_dim)).cuda()
    # batch_gt_dis = torch.zeros((batch_size, featmap_h, featmap_w, expand_dim)).cuda()
    #
    # for b in range(batch_size):
    #     image_h, image_w = raw_img_hs[b], raw_img_ws[b]
    #     obj_indices = torch.where(mask[b] == 1.0)  # all object
    #     for y, x in zip(obj_indices[0], obj_indices[1]):
    #         trans_matrix = trans_m[b, y, x].cpu().detach().numpy()
    #         vertex_ct = torch.reshape(torch.tensor([x, y]).cuda(), [-1, 2]) + torch.reshape(offset_ct[b, y, x], [-1, 2])
    #         vertex_ct = torch.reshape(vertex_ct, [1, -1]).squeeze(0)
    #         if not poly:
    #             vertex_p = torch.reshape(torch.tensor([x, y]).cuda(), [-1, 2]) + torch.reshape(offset_vt_ct[b, y, x], [-1, 2])
    #         else:
    #             if two_offset_branch:
    #                 new_offset_vt_ct = torch.reshape(offset_vt_ct[b, y, x], [-1, 1])
    #                 if direct_angle:
    #                     new_offset_angle_vt_ct = torch.reshape(offset_angle_vt_ct[b, y, x], [-1, 1])
    #                     new_offset_x = torch.unsqueeze(new_offset_vt_ct[:, 0] * torch.cos(new_offset_angle_vt_ct[:, 0]), -1)
    #                     new_offset_y = torch.unsqueeze(new_offset_vt_ct[:, 0] * torch.sin(new_offset_angle_vt_ct[:, 0]), -1)
    #                 else:
    #                     new_offset_angle_vt_ct = torch.reshape(offset_angle_vt_ct[b, y, x], [-1, 2])
    #                     new_offset_x = torch.unsqueeze(new_offset_vt_ct[:, 0] * new_offset_angle_vt_ct[:, 1], -1)
    #                     new_offset_y = torch.unsqueeze(new_offset_vt_ct[:, 0] * new_offset_angle_vt_ct[:, 0], -1)
    #             else:
    #                 if direct_angle:
    #                     new_offset_ct = torch.reshape(offset_vt_ct[b, y, x], [-1, 2])
    #                     new_offset_x = torch.unsqueeze(new_offset_ct[:, 0] * torch.cos(new_offset_ct[:, 1]), -1)
    #                     new_offset_y = torch.unsqueeze(new_offset_ct[:, 0] * torch.sin(new_offset_ct[:, 1]), -1)
    #                 else:
    #                     new_offset_ct = torch.reshape(offset_vt_ct[b, y, x], [-1, 3])
    #                     new_offset_x = torch.unsqueeze(new_offset_ct[:, 0] * new_offset_ct[:, 2], -1)
    #                     new_offset_y = torch.unsqueeze(new_offset_ct[:, 0] * new_offset_ct[:, 1], -1)
    #             vertex_p = torch.reshape(torch.tensor([x, y]).cuda(), [-1, 2]) + torch.reshape(torch.cat([new_offset_x, new_offset_y], -1), [-1, 2])
    #         vertex_p = torch.reshape(vertex_p, [1, -1]).squeeze(0)
    #         # 相对于原图
    #         if image_h >= image_w:
    #             vertex_ct[0::2] = vertex_ct[0::2] * max(image_w, image_h) / featmap_w - abs(image_w - image_h) / 2.
    #             vertex_ct[1::2] = vertex_ct[1::2] * max(image_w, image_h) / featmap_h
    #             vertex_p[0::2] = vertex_p[0::2] * max(image_w, image_h) / featmap_w - abs(image_w - image_h) / 2.
    #             vertex_p[1::2] = vertex_p[1::2] * max(image_w, image_h) / featmap_h
    #         else:
    #             vertex_ct[0::2] = vertex_ct[0::2] * max(image_w, image_h) / featmap_w
    #             vertex_ct[1::2] = vertex_ct[1::2] * max(image_w, image_h) / featmap_h - abs(image_w - image_h) / 2.
    #             vertex_p[0::2] = vertex_p[0::2] * max(image_w, image_h) / featmap_w
    #             vertex_p[1::2] = vertex_p[1::2] * max(image_w, image_h) / featmap_h - abs(image_w - image_h) / 2.
    #         ct_p = vertex_ct[0:2]
    #         dst_ct = cv.perspectiveTransform(ct_p.cpu().detach().numpy().reshape(1, -1, 2), trans_matrix).squeeze(0)
    #         if ct_only:  # 只使用中心点计算标准DM码投影损失
    #             standard_vertex_points = np.array([0, 100, 0, 0, 100, 0, 100, 100]).reshape(-1, 2)
    #             ct_vertex_dis = np.sqrt(np.sum((dst_ct - standard_vertex_points) ** 2, 1))
    #             abs_ct_vertex_diff = torch.tensor(ct_vertex_dis).reshape(expand_dim).cuda()
    #             batch_calc_dis[b, y, x] = abs_ct_vertex_diff
    #             batch_gt_dis[b, y, x] = torch.tensor([np.sqrt(5000), np.sqrt(5000), np.sqrt(5000), np.sqrt(5000)]).cuda()
    #         else:
    #             dst_p = cv.perspectiveTransform(vertex_p.cpu().detach().numpy().reshape(1, -1, 2), trans_matrix).squeeze(0)
    #             dst_v = np.concatenate([dst_ct, dst_p], 0)  # [5, 2]
    #             standard_vertex_points = np.array([50, 50, 0, 100, 0, 0, 100, 0, 100, 100], dtype=np.float32).reshape(-1, 2)
    #             ct_vertex_dis = np.sqrt(np.sum((dst_v - standard_vertex_points) ** 2, 1))
    #             abs_ct_vertex_diff = torch.tensor(ct_vertex_dis).reshape(expand_dim).cuda()
    #             batch_calc_dis[b, y, x] = abs_ct_vertex_diff
    #             # batch_gt_dis[b, y, x] = abs_ct_vertex_diff  # 真值为0
    #
    # expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, expand_dim)
    #
    # loss = F.l1_loss(batch_calc_dis * expand_mask, batch_gt_dis * expand_mask, reduction='sum')
    # loss = loss / (mask.sum() + 1e-4)
    # return loss


def polygon_iou_loss(offset_vt_ct, gt_vertex, mask, expand_dim, raw_img_hs, raw_img_ws, poly, two_offset_branch, direct_angle):
    return torch.tensor(0.0, device='cuda')
    # if not two_offset_branch or not poly:
    #     # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    #     if offset_vt_ct.shape[1] != offset_vt_ct.shape[2]:
    #         offset_vt_ct = offset_vt_ct.permute(0, 2, 3, 1)
    #     batch_size, featmap_h, featmap_w = offset_vt_ct.shape[0:3]
    # else:
    #     offset_vt_ct, offset_angle_vt_ct = offset_vt_ct
    #     if offset_vt_ct.shape[1] != offset_vt_ct.shape[2]:
    #         offset_vt_ct = offset_vt_ct.permute(0, 2, 3, 1)
    #     if offset_angle_vt_ct.shape[1] != offset_angle_vt_ct.shape[2]:
    #         offset_angle_vt_ct = offset_angle_vt_ct.permute(0, 2, 3, 1)
    #     batch_size, featmap_h, featmap_w = offset_vt_ct.shape[0:3]

    # batch_calc_ious_loss = torch.zeros((batch_size, featmap_h, featmap_w)).cuda()
    # batch_gt_ious_loss = torch.zeros((batch_size, featmap_h, featmap_w)).cuda()
    # target_new = torch.zeros_like(gt_vertex)

    # for b in range(batch_size):
    #     image_h, image_w = raw_img_hs[b], raw_img_ws[b]
    #     # 相对于原图
    #     if image_h >= image_w:
    #         target_new[b, :, :, 0:8:2] = gt_vertex[b, :, :, 0:8:2] * max(image_w, image_h) / featmap_w - abs(image_w-image_h)/2.
    #         target_new[b, :, :, 1:8:2] = gt_vertex[b, :, :, 1:8:2] * max(image_w, image_h) / featmap_h
    #     else:
    #         target_new[b, :, :, 0:8:2] = gt_vertex[b, :, :, 0:8:2] * max(image_w, image_h) / featmap_w
    #         target_new[b, :, :, 1:8:2] = gt_vertex[b, :, :, 1:8:2] * max(image_w, image_h) / featmap_h - abs(image_w-image_h)/2.
    #     obj_indices = torch.where(mask[b] == 1.0)  # all object
    #     for y, x in zip(obj_indices[0], obj_indices[1]):
    #         if not poly:
    #             vertex_p = torch.reshape(torch.tensor([x, y]).cuda(), [-1, 2]) + torch.reshape(offset_vt_ct[b, y, x], [-1, 2])
    #         else:
    #             if two_offset_branch:
    #                 new_offset_vt_ct = torch.reshape(offset_vt_ct[b, y, x], [-1, 1])
    #                 if direct_angle:
    #                     new_offset_angle_vt_ct = torch.reshape(offset_angle_vt_ct[b, y, x], [-1, 1])
    #                     new_offset_x = torch.unsqueeze(new_offset_vt_ct[:, 0] * torch.cos(new_offset_angle_vt_ct[:, 0]), -1)
    #                     new_offset_y = torch.unsqueeze(new_offset_vt_ct[:, 0] * torch.sin(new_offset_angle_vt_ct[:, 0]), -1)
    #                 else:
    #                     new_offset_angle_vt_ct = torch.reshape(offset_angle_vt_ct[b, y, x], [-1, 2])
    #                     new_offset_x = torch.unsqueeze(new_offset_vt_ct[:, 0] * new_offset_angle_vt_ct[:, 1], -1)
    #                     new_offset_y = torch.unsqueeze(new_offset_vt_ct[:, 0] * new_offset_angle_vt_ct[:, 0], -1)
    #             else:
    #                 if direct_angle:
    #                     new_offset_ct = torch.reshape(offset_vt_ct[b, y, x], [-1, 2])
    #                     new_offset_x = torch.unsqueeze(new_offset_ct[:, 0] * torch.cos(new_offset_ct[:, 1]), -1)
    #                     new_offset_y = torch.unsqueeze(new_offset_ct[:, 0] * torch.sin(new_offset_ct[:, 1]), -1)
    #                 else:
    #                     new_offset_ct = torch.reshape(offset_vt_ct[b, y, x], [-1, 3])
    #                     new_offset_x = torch.unsqueeze(new_offset_ct[:, 0] * new_offset_ct[:, 2], -1)
    #                     new_offset_y = torch.unsqueeze(new_offset_ct[:, 0] * new_offset_ct[:, 1], -1)
    #             vertex_p = torch.reshape(torch.tensor([x, y]).cuda(), [-1, 2]) + torch.reshape(torch.cat([new_offset_x, new_offset_y], -1), [-1, 2])
    #         vertex_p = torch.reshape(vertex_p, [1, -1]).squeeze(0)
    #         if image_h >= image_w:
    #             vertex_p[0::2] = vertex_p[0::2] * max(image_w, image_h) / featmap_w - abs(image_w - image_h) / 2.
    #             vertex_p[1::2] = vertex_p[1::2] * max(image_w, image_h) / featmap_h
    #         else:
    #             vertex_p[0::2] = vertex_p[0::2] * max(image_w, image_h) / featmap_w
    #             vertex_p[1::2] = vertex_p[1::2] * max(image_w, image_h) / featmap_h - abs(image_w - image_h) / 2.
    #         vertex_gt = target_new[b, y, x]
    #         poly_iou_result = poly_iou(vertex_p, vertex_gt)
    #         batch_calc_ious_loss[b, y, x] = 1.0 - poly_iou_result
    # batch_calc_ious_loss = torch.unsqueeze(batch_calc_ious_loss, -1).repeat(1, 1, 1, expand_dim)
    # batch_gt_ious_loss = torch.unsqueeze(batch_gt_ious_loss, -1).repeat(1, 1, 1, expand_dim)
    # expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, expand_dim)
    # loss = F.l1_loss(batch_calc_ious_loss * expand_mask, batch_gt_ious_loss * expand_mask, reduction='sum')
    # loss = loss / (mask.sum() + 1e-4)
    # return loss
