import sys
import os
import traceback
import cv2

import torch
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QLabel, QVBoxLayout, QStyle, \
    QFileDialog, QSplitter, QFrame
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from matplotlib import pyplot as plt

from deep_sort.utils.parser import get_config
from models.experimental import attempt_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 绝对路径转换为相对路径

from utils.datasets import LoadImages
from utils.general import (check_img_size, cv2, xyxy2xywh, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from deep_sort.deep_sort import DeepSort


class Yolov5Thread(QThread):
    send_img = pyqtSignal(np.ndarray)  # pyqtSignal是pyqt里非常实用的一个接口，是PyQt5 提供的自定义信号类；

    def __init__(self):
        super(Yolov5Thread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = ''
        self.conf = 0.3
        self.iou = 0.5
        self.jump_out = False  # 暂停视频
        self.is_continue = True  # continue/pause

    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # try-catch是用来捕捉异常, 并进行安全输出。
        try:
            print("Yolo start")
            device = select_device(device)  # 获取设备, CPU还是GPU
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size, 能否整除32, 调整图片
            if half:
                model.half()  # to FP16

            # 加载图片和视频
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
            # COCO数据集：人, 车
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            dataset = iter(dataset)

            while True:  # 循环每一张图片
                if self.jump_out:
                    break

                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

                    if half:
                        model.half()  # to FP16

                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

                    self.current_weight = self.weights

                if self.is_continue:
                    path, img, im0s, vid_cap = next(dataset)  # 读取文件路径, 图片, 视频
                    img = torch.from_numpy(img).to(device)  # 转化Tensor格式
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0

                    # 在没有batch_size时, 添加一个轴
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    """
                        pred.shape=(1, num_boxes, 5+num_class)
                        h,w为传入网络图片的长和宽,注意dataset在检测时使用了矩形推理,所以这里h不一定等于w
                        num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
                        pred[..., 0:4]为预测框坐标=预测框坐标为xywh(中心点+宽长)格式
                        pred[..., 4]为objectness置信度
                        pred[..., 5:-1]为分类结果
                    """

                    pred = model(img, augment=augment)[0]

                    """
                      pred: 网络的输出结果
                      conf:置信度阈值
                      iou:iou阈值
                      classes: 是否只保留特定的类别
                      agnostic_nms: 进行nms是否也去除不同类别之间的框
                      max-det: 保留的最大检测框数量
                      ---NMS, 预测框格式: xywh(中心点+长宽)-->xyxy(左上角右下角)
                      pred是一个列表list[torch.tensor], 长度为batch_size
                      每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
                    """

                    # Apply NMS, 非极大抑直
                    pred = non_max_suppression(pred, self.conf, self.iou, classes, agnostic_nms, max_det=max_det)

                    # Process detections, 对每一张图片进行处理
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()  # 是否裁剪检测框
                        '''
                        这段代码是使用一个名为Annotator的对象来标注一张图片（im0），
                        并将线条宽度设置为line_thickness，
                        同时在注释示例中提供了一个字符串(names)。
                        '''
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                        if len(det):
                            # Rescale boxes from img_size to im0 size 将预测信息映射到原图
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            '''
                            该代码使用 reversed() 函数以相反的顺序遍历 detections det 列表。
                            对于每次检测，代码提取三个值：xyxy、conf 和 cls。
                            xyxy 可能是一组四个坐标，用于定义图像中对象周围的边界框。
                            conf 是与检测相关的置信度分数，它表示对象属于检测到的类别的可能性有多大。
                            cls 是检测到的对象的预测类标签。
                            该代码使用 int() 函数将 cls 转换为整数。
                            根据 hide_labels 和 hide_conf 的值，代码为检测到的对象创建标签或将标签设置为 None。
                            如果 hide_labels 为 False，则代码使用名称列表查找与 c 对应的类标签的名称。
                            如果 hide_conf 为 False，则代码使用 f 字符串将置信度分数附加到标签。
                            然后代码调用函数 annotator.box_label() 在检测到的对象周围绘制一个边界框，并使用标签变量对其进行标记。 colors() 函数用于根据类标签为边界框选择颜色。
                            总体而言，此代码可能会可视化图像中的对象检测，并可选择显示/隐藏标签和置信度分数。
                            '''

                            for *xyxy, conf, cls in reversed(det):
                                lbl = names[int(cls)]
                                if lbl in ['bicycle', 'car', 'truck', 'motorcycle']:
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (
                                        names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))

                    im0 = annotator.result()
                    self.send_img.emit(im0)

        except Exception as e:
            traceback.print_exc()


class DeepsortThread(QThread):
    send_img2 = pyqtSignal(np.ndarray)  # pyqtSignal是pyqt里非常实用的一个接口，是PyQt5 提供的自定义信号类；

    def __init__(self):
        super(DeepsortThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = ''
        self.conf = 0.3
        self.iou = 0.5
        self.jump_out = False  # 暂停视频
        self.is_continue = True  # continue/pause
        self.config_deepsort = r'D:\AUT\COMP702 笔记\Demo\deep_sort\configs\deep_sort.yaml'

    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=[1, 2, 3, 5, 7],  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        try:
            # initialize deepsort
            cfg = get_config()
            cfg.merge_from_file(self.config_deepsort)
            deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                max_dist=cfg.DEEPSORT.MAX_DIST,
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                use_cuda=True)

            # Initialize yolov5 detection
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            model = attempt_load(self.weights, map_location=device)  # load FP32 model

            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size, 能否整除32, 调整图片

            if half:
                model.half()  # to FP16

            # 加载图片和视频
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
            dataset = iter(dataset)
            prev_trackers = {}
            trajectories = {}

            # 循环每一张图片
            while True:
                if self.jump_out:
                    break

                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

                    if half:
                        model.half()  # to FP16

                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

                    self.current_weight = self.weights

                if self.is_continue:
                    path, img, im0s, vid_cap = next(dataset)  # 读取文件路径, 图片, 视频
                    img = torch.from_numpy(img).to(device)  # 转化Tensor格式
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    pred = model(img, augment=augment, visualize=visualize)[0]

                    # Apply NMS, 非极大抑直, 进行筛选
                    pred = non_max_suppression(pred, self.conf, self.iou, classes, agnostic_nms)

                    #  Process detections
                    for i, det in enumerate(pred):
                        im0, _ = im0s.copy(), getattr(dataset, 'frame', 0)
                        annotator = Annotator(im0, line_width=2, pil=not ascii)

                        # Rescale boxes from img_size to im0 size 将预测信息映射到原图
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
                        dict_box = dict()

                        '''
                        将边界框的坐标从左上角和右下角表示(xyxy格式), 转换为中心点坐标、宽度和高度表示(xywh格式)
                        这个步骤可以使用xyxy2xywh函数实现, det[:, :4]是一个形状为(N, 4)的numpy数组, 包含了N个边界框的坐标,
                        第一列是左上角x坐标, 第二列是左上角y坐标, 第三列是右下角x坐标, 第四列是右下角y坐标
                        '''

                        confs = det[:, 4:5].cpu()  # 获取每个边界框的置信度得分

                        # update()方法用于使用当前帧的最新检测结果更新跟踪器的状态，并输出当前跟踪的目标列表
                        outputs = deepsort.update(bbox_xywh, confs, im0)
                        print(outputs)
                        cmap = plt.get_cmap('tab20b')
                        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                        '''
                        outputs列表包含跟踪目标的信息, confs列表包含每个目标的置信度分数
                        bbox_xywh是一个格式为[x, y, width, height]的边界框坐标列表
                        confs是每个检测到的对象的置信度分数列表, im0是一个图像
                        '''

                        if len(outputs) > 0:
                            for idx, track in enumerate(outputs):
                                x1, y1, x2, y2, id = track

                                if id in prev_trackers:
                                    prev_x1, prev_y1, prev_x2, prev_y2 = prev_trackers[id]
                                    cv2.line(im0, (int((prev_x1 + prev_x2) / 2), int((prev_y1 + prev_y2) / 2)),
                                             (int((x1 + x2) / 2), int((y1 + y2) / 2)), (0, 255, 0), 8)

                                if id not in trajectories:
                                    trajectories[id] = [(int((x1 + x2) / 2), int((y1 + y2) / 2))]
                                else:
                                    trajectories[id].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

                                # Draw trajectory
                                if len(trajectories[id]) > 1:
                                    for i in range(1, len(trajectories[id])):
                                        cv2.line(im0, trajectories[id][i - 1], trajectories[id][i], (0, 255, 0), 8)

                                for j, (output, conf) in enumerate(zip(outputs, confs)):
                                    bboxes = output[0:4]
                                    id = output[4]

                                    color_ = colors[int(id) % len(colors)]
                                    color_ = [i * 255 for i in color_]

                                    label = f'ID:{id}'  # 在目标的边界框上显示目标的ID信息
                                    annotator.box_label(bboxes, label, color=color_)

                        prev_trackers = {track[-1]: track[:-1] for track in outputs}

                    im0 = annotator.result()  # 返回的数据结构可能包括每个目标的ID、位置、速度、加速度等信息, 以及图像上绘制的边界框和标签等视觉化结果
                    self.send_img2.emit(im0)

        except Exception as e:
            traceback.print_exc()


class Window(QWidget):  # 是所有用户界面类的基类，它能接收所有的鼠标、键盘和其他系统窗口事件

    def __init__(self):
        super().__init__()
        # 设置标题(Video Player)
        self.setWindowTitle("Player")
        self.button_Adaptive = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # 从屏幕上(0, 0),位置开始(即为最左上角的点),显示一个1020*600的界面(长1020, 宽600)
        self.setGeometry(0, 0, 1280, 700)

        # 创建播放按钮
        self.playBtn = QPushButton()
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.setEnabled(False)
        self.playBtn.clicked.connect(self.play)

        self.pauseBtn = QPushButton()
        self.pauseBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pauseBtn.setEnabled(False)
        self.pauseBtn.clicked.connect(self.pause)

        self.replayBtn = QPushButton()
        self.replayBtn.setText('Replay')
        self.replayBtn.setEnabled(False)
        self.replayBtn.clicked.connect(self.replay)

        # comboBox自动搜索.pt文件
        self.comboBox = QtWidgets.QComboBox()  # 创建comboBox
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')  # 搜索本地pt文件夹
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]  # 搜索本地pt文件
        self.comboBox.addItems(self.pt_list)  # 添加pt文件
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt)
        self.qtimer_search.start(2000)
        self.model_type = self.comboBox.currentText()
        self.comboBox.currentTextChanged.connect(self.change_model)  # 发送信号

        # 创建选择文件按钮
        self.FileBtn = QPushButton('Select Video')
        self.FileBtn.clicked.connect(self.open_file)

        # 创建label来放置左视频播放
        self.left_label = QLabel()
        self.left_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)  # 设置样式, 不然无法显示
        self.left_label.setFixedWidth(640)
        self.left_label.setFixedHeight(640)

        # 创建label来放置右视频播放
        self.right_label = QLabel()
        self.right_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)  # 设置样式, 不然无法显示
        self.right_label.setFixedWidth(640)
        self.right_label.setFixedHeight(640)

        # 创建分割线
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.left_label)
        self.splitter.addWidget(self.right_label)

        # QHBoxLayout用来水平布局.
        hbox = QHBoxLayout()
        hbox.addWidget(self.FileBtn)
        hbox.addWidget(self.comboBox)
        hbox.addWidget(self.playBtn)
        hbox.addWidget(self.pauseBtn)
        hbox.addWidget(self.replayBtn)

        # QVBoxLayout用来垂直布局.
        vbox = QVBoxLayout()
        vbox.addWidget(self.splitter)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        try:
            # yolov5 thread
            self.yolothread = Yolov5Thread()
            self.yolothread.source = '0'
            self.model = self.comboBox.currentText()
            self.yolothread.weights = "./pt/%s" % self.model
            self.yolothread.send_img.connect(lambda x: self.show_image(x, self.left_label))

            # Deepsort thread
            self.Deepsort_thread = DeepsortThread()
            self.Deepsort_thread.source = '0'
            self.model = self.comboBox.currentText()
            self.Deepsort_thread.weights = "./pt/%s" % self.model
            self.Deepsort_thread.send_img2.connect(lambda x: self.show_image2(x, self.right_label))

        except Exception as e:

            traceback.print_exc()

    def search_pt(self):
        try:
            pt_list = os.listdir('./pt')
            pt_list = [file for file in pt_list if file.endswith('.pt')]
            pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.comboBox.clear()
                self.comboBox.addItems(self.pt_list)

        except Exception as e:

            traceback.print_exc()

    def open_file(self):

        try:
            filename, _ = QFileDialog.getOpenFileName(self, "Select Video")

            if filename != '':
                self.yolothread.source = filename
                self.Deepsort_thread.source = filename

                self.playBtn.setEnabled(True)

        except Exception as e:
            print('%s' % e)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.yolothread.weights = "./pt/%s" % self.model_type

    def show_image(self, img_src, label):  # 在Qlabel上显示检测的图片
        try:
            ih, iw, _ = img_src.shape  # 图片的宽和高
            w = label.geometry().width()  # label的宽
            h = label.geometry().height()  # label的高

            # keep original aspect ratio 保持原始纵横比
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            self.left_label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_image2(self, img_src, label):  # 在Qlabel上显示检测的图片
        try:
            ih, iw, _ = img_src.shape  # 图片的宽和高
            w = label.geometry().width()  # label的宽
            h = label.geometry().height()  # label的高

            # keep original aspect ratio 保持原始纵横比
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            self.right_label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def play(self):
        self.yolothread.jump_out = False
        self.yolothread.is_continue = True

        self.Deepsort_thread.jump_out = False
        self.Deepsort_thread.is_continue = True

        # 按钮激活和关闭
        self.pauseBtn.setEnabled(True)
        self.playBtn.setEnabled(False)
        self.replayBtn.setEnabled(True)

        # 开始进程
        self.yolothread.start()
        self.Deepsort_thread.start()

    def pause(self):
        # 暂停分析
        self.yolothread.is_continue = False
        self.Deepsort_thread.is_continue = False

        # 按钮激活和关闭
        self.pauseBtn.setEnabled(False)
        self.playBtn.setEnabled(True)
        self.replayBtn.setEnabled(True)

    def replay(self):
        # 重新loop
        self.yolothread.jump_out = True
        self.Deepsort_thread.jump_out = True

        # 按钮激活和关闭
        self.replayBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.playBtn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)  # QApplication是PyQt的整个后台管理的关键
    window = Window()
    window.show()
    sys.exit(app.exec_())
