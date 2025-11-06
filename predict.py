# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import shutil
import time

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
from centernet import CenterNet

if __name__ == "__main__":
    centernet = CenterNet()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # -------------------------------------------------------------------------#
    #   crop指定了是否在单张图片预测后对目标进行截取
    #   crop仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    # -------------------------------------------------------------------------#
    test_interval = 100
    # -------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    if mode == "predict":
        '''
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_dr_txt.py，在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
        2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        3、如果想要获得预测框的坐标，可以进入centernet.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        4、如果想要利用预测框截取下目标，可以进入centernet.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入centernet.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        # 2. 要处理的图片文件名列表（放在 img/ 目录下）
        img_names = [
            # "00326.jpg",
            # "00431.jpg",
            # "00572.jpg",
            # "001066.jpg",
            # "00309.jpg",
            # "005375.jpg",
            # "00001.jpg",
            # "raw.jpg",
            # "004329.jpg",  # 画图
            "test-bs1.jpg",  # 比赛用图
            # "plot.jpg"
        ]

        base_img_dir = "img"
        base_out_dir = "outputs_imgs"

        save_dir = "output_maps"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        for img_name in img_names:
            img_path = os.path.join(base_img_dir, img_name)
            if not os.path.exists(img_path):
                print(f"[Skip] {img_path} not found.")
                continue

            # 3. 推理
            image = Image.open(img_path).convert("RGB")
            start_time = time.time()
            r_image, ht_img, _ = centernet.detect_image(image)
            end_time = time.time()
            ht_img = Image.fromarray(ht_img)

            # 计算处理时间
            processing_time = end_time - start_time
            fps = 1 / processing_time

            print(f"{img_name} - Processing time: {processing_time:.4f} seconds, FPS: {fps:.2f}")

            # 4. 创建独立输出目录
            out_subdir = os.path.join(base_out_dir, os.path.splitext(img_name)[0])
            os.makedirs(out_subdir, exist_ok=True)

            # 5. 保存单图
            r_image.save(os.path.join(out_subdir, "detection_result.jpg"))
            ht_img.save(os.path.join(out_subdir, "heatmap.jpg"))

            # 6. 保存并排对比图
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(r_image)
            plt.title("Detection Result")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(ht_img)
            plt.title("Heatmap")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(out_subdir, "detection_comparison.jpg"),
                        bbox_inches='tight', dpi=300)
            plt.close()  # 防止内存泄漏

            print(f"[Done] {img_name} -> {out_subdir}")

        print("All images processed.")

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame, _ = centernet.detect_image(frame)
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/00001.jpg')
        tact_time = centernet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image, _ = centernet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
