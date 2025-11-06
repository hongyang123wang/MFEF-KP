import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.utils import get_lr
from nets.centernet_training import focal_loss, reg_l1_loss, polygon_iou_loss, back_vertex_loss, center_trans_loss
import torch.nn.functional as F


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, backbone, poly_setting):
    poly = poly_setting["poly"]
    two_offset_branch = poly_setting["two_offset_branch"]
    ct_only = poly_setting["ct_only"]
    direct_theta = poly_setting["direct_theta"]

    total_hm_loss = 0
    total_offset_ct_loss = 0
    total_offset_vt_ct_loss = 0
    total_offset_angle_vt_ct_loss = 0
    total_kphm_loss = 0
    total_back_vertex_loss = 0
    total_oks_loss = 0
    total_angle_loss = 0
    lamda_back_vt_loss, lamda_oks_loss = 0.1, 1
    lamda_angle_loss = 0.1 if ct_only else 0.01
    t_all_lamda = 1
    total_other_loss = 0

    total_train_loss = 0
    val_loss = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            with torch.no_grad():
                if cuda:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch]
                else:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in batch]

            if not two_offset_branch or not poly:
                batch_images, batch_hms, batch_vertexes_hms, batch_regs_ct, batch_regs_vt_offset, batch_reg_masks, batch_vertexes, batch_reg_trans_matrix, raw_img_ws, raw_img_hs, seg_masks = batch
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()

            if backbone == "resnet50":
                if not two_offset_branch or not poly:
                    (hm, offset_ct, offset_vt_ct, kp_hm), kp = model_train(batch_images)

                # ------------  辅助损失 s ------------
                batch_hms_32 = batch_hms.permute(0, 3, 1, 2)  # [B,1,H,W]
                batch_vts_32 = batch_vertexes_hms.permute(0, 3, 1, 2)  # [B,4,H,W]
                gt_heatmap = torch.cat([batch_hms_32, batch_vts_32], dim=1).max(dim=1, keepdim=True)[0]  # [B,1,64,64]
                loss_s_ct_vt = focal_loss(torch.sigmoid(kp), gt_heatmap, epoch)
                t_all = loss_s_ct_vt * t_all_lamda
                # t_all = torch.tensor(0.0).cuda()
                # ------------  辅助损失 e ------------

                hm_loss = focal_loss(hm, batch_hms, epoch)
                kp_loss = focal_loss(kp_hm, gt_heatmap, epoch)

                off_ct_loss = reg_l1_loss(offset_ct, batch_regs_ct, batch_reg_masks, 2)

                if not two_offset_branch or not poly:  # 单分支/笛卡尔坐标
                    off_angle_vt_ct_loss = 0
                    if not poly:
                        off_vt_ct_loss = 0.1 * reg_l1_loss(offset_vt_ct, batch_regs_vt_offset, batch_reg_masks, 8)
                    else:
                        if direct_theta:
                            off_vt_ct_loss = 0.1 * reg_l1_loss(offset_vt_ct, batch_regs_vt_offset, batch_reg_masks, 8)
                        else:
                            off_vt_ct_loss = 0.1 * reg_l1_loss(offset_vt_ct, batch_regs_vt_offset, batch_reg_masks, 12)
                    back_vt_loss, oks_loss, kphm_loss, angle_loss = back_vertex_loss(offset_vt_ct,
                                                                                     batch_vertexes_hms,
                                                                                     batch_vertexes,
                                                                                     batch_reg_masks, 8,
                                                                                     poly,
                                                                                     two_offset_branch,
                                                                                     direct_theta, epoch)
                back_vt_loss *= lamda_back_vt_loss
                oks_loss *= lamda_oks_loss

                # 重新赋值
                kphm_loss = kp_loss
                loss = t_all + hm_loss + off_ct_loss + off_vt_ct_loss + off_angle_vt_ct_loss + back_vt_loss + oks_loss + angle_loss + kphm_loss
                # loss end

                total_train_loss += loss.item()
                total_hm_loss += hm_loss.item()
                total_offset_ct_loss += off_ct_loss.item()
                total_offset_vt_ct_loss += off_vt_ct_loss.item()
                if poly and two_offset_branch:
                    total_offset_angle_vt_ct_loss += off_angle_vt_ct_loss.item()
                total_kphm_loss += kphm_loss.item()
                total_back_vertex_loss += back_vt_loss.item()
                total_oks_loss += oks_loss.item()
                total_angle_loss += angle_loss.item()
                total_other_loss += t_all.item()

            loss.backward()
            optimizer.step()

            train_loss_avg = total_train_loss / (iteration + 1)
            hm_loss_avg = total_hm_loss / (iteration + 1)
            offset_ct_loss_avg = total_offset_ct_loss / (iteration + 1)
            offset_vt_ct_loss_avg = total_offset_vt_ct_loss / (iteration + 1)
            kphm_loss_avg = total_kphm_loss / (iteration + 1)
            angle_loss_avg = total_angle_loss / (iteration + 1)
            back_vt_loss_avg = total_back_vertex_loss / (iteration + 1)
            oks_loss_avg = total_oks_loss / (iteration + 1)

            other_loss = total_other_loss / (iteration + 1)
            # 在训练循环中添加计数器
            print_interval = 50  # 每10个batch打印一次

            if iteration % print_interval == 0:
                print(
                    f"\nTotal Loss: {train_loss_avg:.3f}, "
                    f"HM Loss: {hm_loss_avg:.3f}, "
                    f"Offset CT Loss: {offset_ct_loss_avg:.3f}, "
                    f"Offset VT Loss: {offset_vt_ct_loss_avg:.3f}, "
                    f"Pre SHM loss: {other_loss:.3f}, "
                    f"Kp Hm Loss: {kphm_loss_avg:.3f}"
                )
                # 第二行显示剩余的损失信息和学习率
                print(
                    # f"Angle Loss: {angle_loss_avg:.3f}, "
                    f"back_vt_loss_avg: {back_vt_loss_avg:.3f}, "
                    f"OKS_avg: {oks_loss_avg:.3f}, "
                    f"Learning Rate: {get_lr(optimizer):.2e}, "
                    f"cur_kphm_loss: {kphm_loss:.2e}, "
                    f"cur_shm_loss: {t_all:.2e}, "
                )
                # 第三行
                print(
                    f"current loss: {loss:.3f}, "
                    f"hm_loss: {hm_loss:.3f}, "
                    f"offset_ct_loss: {off_ct_loss:.3f}, "
                    f"off_vt_ct_loss: {off_vt_ct_loss:.3f}, "
                    f"back_vt_loss: {back_vt_loss:.3f}, "
                    f"oks_loss: {oks_loss:.3f},"
                    # f"cur_angle_loss:{angle_loss:.3f}"
                )
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{train_loss_avg:.3f}',
                'current Loss': f'{loss:.3f}',
            })
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            with torch.no_grad():
                if cuda:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch]
                else:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in batch]

                if not two_offset_branch or not poly:
                    batch_images, batch_hms, batch_vertexes_hms, batch_regs_ct, batch_regs_vt_offset, batch_reg_masks, batch_vertexes, batch_reg_trans_matrix, raw_img_ws, raw_img_hs, seg_masks = batch
                else:
                    batch_images, batch_hms, batch_vertexes_hms, batch_regs_ct, batch_regs_vt_offset, batch_regs_vt_offset_angle, batch_reg_masks, batch_vertexes, batch_reg_trans_matrix, raw_img_ws, raw_img_hs, seg_masks = batch

                if backbone == "resnet50":
                    if not two_offset_branch or not poly:
                        (hm, offset_ct, offset_vt_ct, kp_hm), kp = model_train(batch_images)

                    # ------------  辅助损失 s ------------
                    batch_hms_32 = batch_hms.permute(0, 3, 1, 2)  # [B,1,H,W]
                    batch_vts_32 = batch_vertexes_hms.permute(0, 3, 1, 2)  # [B,4,H,W]
                    gt_heatmap = torch.cat([batch_hms_32, batch_vts_32], dim=1).max(dim=1, keepdim=True)[
                        0]  # [B,1,64,64]
                    loss_s_ct_vt = focal_loss(torch.sigmoid(kp), gt_heatmap, epoch)
                    t_all = loss_s_ct_vt * t_all_lamda
                    # t_all = torch.tensor(0.0).cuda()
                    # ------------  辅助损失 e ------------

                    hm_loss = focal_loss(hm, batch_hms, epoch)
                    kp_loss = focal_loss(kp_hm, gt_heatmap, epoch)

                    off_ct_loss = reg_l1_loss(offset_ct, batch_regs_ct, batch_reg_masks, 2)

                    if not two_offset_branch or not poly:  # 单分支/笛卡尔坐标
                        off_angle_vt_ct_loss = 0
                        if not poly:
                            off_vt_ct_loss = 0.1 * reg_l1_loss(offset_vt_ct, batch_regs_vt_offset, batch_reg_masks,
                                                               8)
                        else:
                            if direct_theta:
                                off_vt_ct_loss = 0.1 * reg_l1_loss(offset_vt_ct, batch_regs_vt_offset,
                                                                   batch_reg_masks, 8)
                            else:
                                off_vt_ct_loss = 0.1 * reg_l1_loss(offset_vt_ct, batch_regs_vt_offset,
                                                                   batch_reg_masks, 12)
                        back_vt_loss, oks_loss, kphm_loss, angle_loss = back_vertex_loss(offset_vt_ct,
                                                                                         batch_vertexes_hms,
                                                                                         batch_vertexes,
                                                                                         batch_reg_masks, 8,
                                                                                         poly,
                                                                                         two_offset_branch,
                                                                                         direct_theta, epoch)
                    back_vt_loss *= lamda_back_vt_loss
                    oks_loss *= lamda_oks_loss

                    kphm_loss = kp_loss

                    loss = t_all + hm_loss + off_ct_loss + off_vt_ct_loss + off_angle_vt_ct_loss + back_vt_loss + oks_loss + angle_loss + kphm_loss
                    # loss end

                    val_loss += loss.item()

                pbar.set_postfix(**{'total_val_loss': val_loss / (iteration + 1)})
                pbar.update(1)
    print('Finish Validation')

    # loss_history.append_loss(total_train_loss / epoch_step, val_loss / epoch_step_val)
    # 记录损失
    epoch_train_loss = total_train_loss / epoch_step
    epoch_hm = total_hm_loss / epoch_step
    epoch_offset_ct = total_offset_ct_loss / epoch_step
    epoch_offset_vt = total_offset_vt_ct_loss / epoch_step
    epoch_offset_ang = total_offset_angle_vt_ct_loss / epoch_step
    epoch_poly_iou = total_kphm_loss / epoch_step
    epoch_back_vt = total_back_vertex_loss / epoch_step
    epoch_back_vt_hm = total_oks_loss / epoch_step
    epoch_ct_trans = total_angle_loss / epoch_step
    val_loss_avg = val_loss / max(1, epoch_step_val)
    epoch_other_avg = total_other_loss / epoch_step

    loss_history.append_loss(
        epoch_train_loss, val_loss_avg,
        epoch_hm, epoch_offset_ct, epoch_offset_vt,
        epoch_offset_ang, epoch_poly_iou,
        epoch_back_vt, epoch_back_vt_hm,
        epoch_ct_trans, epoch_other_avg
    )
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_train_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (
        epoch + 1, total_train_loss / epoch_step, val_loss / epoch_step_val))

    return val_loss / epoch_step_val
