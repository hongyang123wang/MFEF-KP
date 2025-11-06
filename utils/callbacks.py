import os

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory:
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, "loss_" + self.time_str)
        os.makedirs(self.save_path, exist_ok=True)

        # 主损失
        self.losses = []  # float
        self.val_loss = []  # float

        # 子损失（统一存 float）
        self.hm_losses = []
        self.offset_ct_losses = []
        self.offset_vt_ct_losses = []
        self.offset_angle_vt_ct_losses = []
        self.poly_iou_losses = []
        self.back_vertex_losses = []
        self.back_vertex_hm_losses = []
        self.ct_trans_losses = []
        self.fg_losses = []

    # ----------  追加一次 epoch  ----------
    def append_loss(
            self,
            loss, val_loss,
            hm_loss, offset_ct_loss, offset_vt_ct_loss,
            offset_angle_vt_ct_loss, poly_iou_loss,
            back_vertex_loss, back_vertex_hm_loss,
            ct_trans_loss, fg_loss):
        # 主损失
        self.losses.append(float(loss))
        self.val_loss.append(float(val_loss))

        # 子损失（全部转 float，防止 CUDA tensor）
        self.hm_losses.append(float(hm_loss))
        self.offset_ct_losses.append(float(offset_ct_loss))
        self.offset_vt_ct_losses.append(float(offset_vt_ct_loss))
        self.offset_angle_vt_ct_losses.append(float(offset_angle_vt_ct_loss))
        self.poly_iou_losses.append(float(poly_iou_loss))
        self.back_vertex_losses.append(float(back_vertex_loss))
        self.back_vertex_hm_losses.append(float(back_vertex_hm_loss))
        self.ct_trans_losses.append(float(ct_trans_loss))
        self.fg_losses.append(float(fg_loss))

        # 写 txt
        with open(os.path.join(self.save_path, "epoch_loss_" + self.time_str + ".txt"), 'a') as f:
            f.write(f"{loss:.4f},{hm_loss:.4f},{offset_ct_loss:.4f},"
                    f"{offset_vt_ct_loss:.4f},{offset_angle_vt_ct_loss:.4f},{poly_iou_loss:.4f},"
                    f"{back_vertex_loss:.4f},{back_vertex_hm_loss:.4f},{ct_trans_loss:.4f},{fg_loss:.4f}\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + self.time_str + ".txt"), 'a') as f:
            f.write(f"{val_loss:.4f}\n")

        # 画图
        self.loss_plot()

    # ----------  画图 ----------
    def loss_plot(self):
        # 通用 2×2 拼图
        def plot_quad(losses, titles, fname):
            n = len(losses)
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            axes = axes.ravel()
            for i in range(4):
                ax = axes[i]
                if i < n:
                    ax.plot(losses[i])
                    ax.set_title(titles[i], fontsize=10)
                else:
                    ax.axis('off')  # 不足 4 个直接留白
                ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, fname))
            plt.close()

        # 1. 中心检测
        plot_quad(
            [self.hm_losses, self.offset_ct_losses, self.ct_trans_losses, self.fg_losses],
            ['Heatmap', 'Offset-CT', 'CT-Trans', 'FG-DCE'],
            f"loss_center_{self.time_str}.png"
        )

        # 2. 顶点回归
        plot_quad(
            [self.offset_vt_ct_losses, self.offset_angle_vt_ct_losses],
            ['Offset-VT-CT', 'Offset-Angle-VT-CT', '', ''],
            f"loss_vertex_{self.time_str}.png"
        )

        # 3. 几何约束
        plot_quad(
            [self.poly_iou_losses, self.ct_trans_losses],
            ['Poly-IoU', 'CT-Trans', '', ''],
            f"loss_geo_{self.time_str}.png"
        )

        # 4. 辅助任务
        plot_quad(
            [self.back_vertex_losses, self.back_vertex_hm_losses, self.fg_losses],
            ['Back-Vertex', 'Back-Vertex-HM', 'FG-DCE', ''],
            f"loss_aux_{self.time_str}.png"
        )

        # 5. 主损失
        plt.figure(figsize=(6, 4))
        plt.plot(self.losses, label='train')
        plt.plot(self.val_loss, label='val')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, f"loss_main_{self.time_str}.png"))
        plt.close()
