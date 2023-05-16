import sys
import os
import torch
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QLabel, QVBoxLayout, QStyle, \
    QFileDialog, QSplitter, QFrame
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from models.experimental import attempt_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 绝对路径转换为相对路径

from utils.datasets import LoadImages
from utils.general import (check_img_size, cv2, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


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
        self.current_weight = './yolov5s.pt'
        self.is_continue = True  # continue/pause
        self.percent_length = 1000  # progress bar

    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
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
                            xyxy 是一组四个坐标，用于定义图像中对象周围的边界框。
                            conf 是与检测相关的置信度分数，它表示对象属于检测到的类别的可能性有多大。
                            cls 是检测到的对象的预测类标签。
                            该代码使用 int() 函数将 cls 转换为整数。
                            根据 hide_labels 和 hide_conf 的值，代码为检测到的对象创建标签或将标签设置为 None。
                            如果 hide_labels 为 False，则代码使用名称列表查找与 c 对应的类标签的名称。
                            如果 hide_conf 为 False，则代码使用 f 字符串将置信度分数附加到标签。
                            然后代码调用函数 annotator.box_label() 在检测到的对象周围绘制一个边界框，并使用标签变量对其进行标记。 colors() 函数用于根据类标签为边界框选择颜色。
                            总体而言，可视化图像中的对象检测，并可选择显示/隐藏标签和置信度分数。
                            '''

                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))

                    im0 = annotator.result()
                    self.send_img.emit(im0)

        except Exception as e:
            print('%s' % e)


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

        # yolov5 thread
        self.yolothread = Yolov5Thread()
        self.yolothread.source = '0'
        self.model = self.comboBox.currentText()
        self.yolothread.weights = "./pt/%s" % self.model
        self.yolothread.send_img.connect(lambda x: self.show_image(x, self.left_label))

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video")

        if filename != '':
            self.yolothread.source = filename
            self.playBtn.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.yolothread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

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

    def play(self):
        self.yolothread.jump_out = False
        self.yolothread.is_continue = True
        # 按钮激活和关闭
        self.pauseBtn.setEnabled(True)
        self.playBtn.setEnabled(False)
        self.replayBtn.setEnabled(True)
        # 开始进程
        self.yolothread.start()

    def pause(self):
        # 暂停分析
        self.yolothread.is_continue = False
        # 按钮激活和关闭
        self.pauseBtn.setEnabled(False)
        self.playBtn.setEnabled(True)
        self.replayBtn.setEnabled(True)

    def replay(self):
        # 重新loop
        self.yolothread.jump_out = True
        # 按钮激活和关闭
        self.replayBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.playBtn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)  # QApplication是PyQt的整个后台管理的命脉
    window = Window()
    window.show()
    sys.exit(app.exec_())
