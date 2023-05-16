import sys
import os
import time
import torch
import cv2
import numpy as np
import json
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QLabel, QVBoxLayout, QSlider, QStyle, \
    QSizePolicy, QFileDialog, QSplitter, QFrame, QLineEdit
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QUrl, QTimer, pyqtSignal, QThread
from torch.backends import cudnn
from PyQt5.QtGui import QImage, QPixmap
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device
from utils.capnums import Camera

'''
1.Load the YOLOv5 model using PyTorch, and specify the device to use (e.g., CPU or GPU).

2.Read the input video file or capture frames from a camera.

3.Preprocess each frame by resizing it to the required input size of the model, and convert it to a PyTorch tensor.

4.Pass the tensor through the YOLOv5 model and get the predicted bounding boxes, object class labels, and confidence scores.

5. Draw the predicted bounding boxes on the original image.

6.Convert the image to a QPixmap object and display it on a QLabel using the setPixmap() method.
'''


class YoloThread(QThread):
    send_img = pyqtSignal(np.ndarray)  # pyqtSignal是pyqt里非常实用的一个接口，是PyQt5 提供的自定义信号类；
    send_raw = pyqtSignal(np.ndarray)  # NumPy 最重要的一个特点是其N 维数组对象ndarray
    send_statistic = pyqtSignal(dict)
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(YoloThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False  # jump out of the loop
        self.is_continue = True  # continue/pause
        self.percent_length = 1000  # progress bar
        self.rate_check = True  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'

    # yolo source code
    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)

            while True:
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break
                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
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
                    path, img, im0s, self.vid_cap = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    if count % 120 == 0 and count >= 120:
                        fps = int(240 / (time.time() - start_time))
                        self.send_fps.emit('fps：' + str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms,
                                               max_det=max_det)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))

                    if self.rate_check:
                        time.sleep(1 / self.rate)
                    im0 = annotator.result()
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if self.save_fold:
                        os.makedirs(self.save_fold, exist_ok=True)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                   time.localtime()) + '.jpg')
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 120
                                # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold,
                                                         time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                    if percent == self.percent_length:
                        print(count)
                        self.send_percent.emit(0)
                        self.send_msg.emit('finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)


class Window(QWidget):  # 是所有用户界面类的基类，它能接收所有的鼠标、键盘和其他系统窗口事件

    def __init__(self):
        super().__init__()
        # 设置标题(Yolov5 Video Player)
        self.setWindowTitle("Yolov5 Video Player")

        # 从屏幕上(0, 0),位置开始(即为最左上角的点),显示一个1020*600的界面(长1020, 宽600)
        self.setGeometry(0, 0, 1100, 600)

        # 自动搜索pt文件
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        self.videowidget = QVideoWidget()  # QVideoWidget类提供了一个小部件，它显示由媒体对象生成的视频
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # 创建播放按钮
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)

        # comboBox自动搜索.pt文件
        self.comboBox = QtWidgets.QComboBox()
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)
        self.model_type = self.comboBox.currentText()
        self.comboBox.currentTextChanged.connect(self.change_model)

        # 创建选择文件按钮
        self.FileBtn = QPushButton('Select File')
        self.FileBtn.clicked.connect(self.open_file)

        # QSlider 创建滑块
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)  # 设置取值范围
        self.slider.sliderMoved.connect(self.set_position)  # 发送信号

        # 创建label来放置视频播放
        self.video_label = QLabel()
        self.video_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)  # 设置样式, 不然无法显示
        self.video_label.setFixedWidth(550)
        self.video_label.setFixedHeight(550)

        # 创建分割线
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.videowidget)
        self.splitter.addWidget(self.video_label)

        # QHBoxLayout用来水平布局.
        hbox = QHBoxLayout()
        hbox.addWidget(self.FileBtn)
        hbox.addWidget(self.comboBox)
        hbox.addWidget(self.playBtn)
        hbox.addWidget(self.slider)

        # QVBoxLayout用来垂直布局.
        vbox = QVBoxLayout()
        vbox.addWidget(self.splitter)
        vbox.addLayout(hbox)

        self.mediaPlayer.setVideoOutput(self.videowidget)
        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

        self.setLayout(vbox)

        # yolov5 thread
        self.yolothread = YoloThread()
        self.yolothread.source = '0'
        self.model = self.comboBox.currentText()
        self.yolothread.weights = "./pt/%s" % self.model
        self.yolothread.send_img.connect(lambda x: self.show_image(x, self.video_label))

    # 选择本地文件
    def open_file(self):
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        if name:
            self.yolothread.source = name
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(name)))
            self.playBtn.setEnabled(True)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 选择yolov5权重
    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    # combobox切换权重
    def change_model(self):
        self.model_type = self.comboBox.currentText()

    def play_video(self):
        self.yolothread.jump_out = False
        self.yolothread.start()

        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.yolothread.jump_out = True
        else:
            self.mediaPlayer.play()

    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        else:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    # 在Qlabel上显示检测的图片
    def show_image(self, img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
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
            self.video_label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)  # QApplication是PyQt的整个后台管理的命脉
    window = Window()
    window.show()
    sys.exit(app.exec_())
