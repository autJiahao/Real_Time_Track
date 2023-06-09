import sys
import os
import traceback
import shutil
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
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.datasets import LoadImages
from utils.general import (check_img_size, cv2, xyxy2xywh, non_max_suppression, scale_coords, increment_path,
                           strip_optimizer)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from deep_sort.deep_sort import DeepSort


class Yolov5Thread(QThread):
    send_img = pyqtSignal(np.ndarray)

    def __init__(self):
        super(Yolov5Thread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = ''
        self.conf = 0.3
        self.iou = 0.5
        self.jump_out = False
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

        try:
            print("Yolo start")
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            if half:
                model.half()  # to FP16

            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
            # COCO
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            dataset = iter(dataset)

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
                    path, img, im0s, vid_cap = next(dataset)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    pred = model(img, augment=augment, visualize=visualize)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf, self.iou, classes, agnostic_nms)

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

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
    send_img2 = pyqtSignal(np.ndarray)

    def __init__(self):
        super(DeepsortThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = ''
        self.conf = 0.3
        self.iou = 0.5
        self.jump_out = False
        self.is_continue = True  # continue/pause'
        self.config_deepsort = './deep_sort/configs/deep_sort.yaml'

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
            exist_ok=False,  # existing project/name ok, do not increment
            project='runs',  # save results to project/name
            name='exp',  # save results to project/name
            save_txt=True,  # save results to *.txt
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
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Directories
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir

            txt_file_name = self.source.split('/')[-1].split('.')[0]
            txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

            if half:
                model.half()  # to FP16

            # load videos
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
            dataset = iter(dataset)

            prev_trackers = {}
            trajectories = {}
            write_data = []

            # while
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
                    path, img, im0s, vid_cap = next(dataset)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    pred = model(img, augment=augment, visualize=visualize)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf, self.iou, classes, agnostic_nms)

                    #  Process detections
                    for i, det in enumerate(pred):
                        p, s, im0, frame, frame_idx = path, '', im0s.copy(), getattr(dataset, 'frame', 0), 0
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        bbox_xywh = xyxy2xywh(det[:, :4]).cpu()

                        confs = det[:, 4:5].cpu()

                        outputs = deepsort.update(bbox_xywh, confs, im0)

                        cmap = plt.get_cmap('tab20b')
                        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                        txt_path = str(Path(save_dir)) + '/' + '.txt'

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

                                if save_txt:
                                    x = int(((x1 + x2) / 2))
                                    y = int(((y1 + y2) / 2))
                                    write_data.append((id, x, y))

                                for j, (output, conf) in enumerate(zip(outputs, confs)):
                                    bboxes = output[0:4]
                                    id = output[4]

                                    color_ = colors[int(id) % len(colors)]
                                    color_ = [i * 255 for i in color_]

                                    label = f'ID:{id}'
                                    annotator.box_label(bboxes, label, color=color_)

                        prev_trackers = {track[-1]: track[:-1] for track in outputs}

                        im0 = annotator.result()
                        self.send_img2.emit(im0)

                        write_data.sort()
                        # Write MOT compliant results to file
                        with open(txt_path, 'a') as f:
                            for item in write_data:
                                f.write(('%g ' * 3 + '\n') % item)

                lists = []

                with open(txt_path, 'r') as f:
                    for line in f:
                        lists.append(line.strip())

                with open(txt_path, "w") as f:
                    for item in sorted(lists):
                        f.writelines(item)
                        f.writelines('\n')

            if update:
                strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)

        except Exception as e:
            traceback.print_exc()


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Player")
        self.button_Adaptive = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.setGeometry(0, 0, 1280, 700)

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

        self.comboBox = QtWidgets.QComboBox()
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt)
        self.qtimer_search.start(2000)
        self.model_type = self.comboBox.currentText()
        self.comboBox.currentTextChanged.connect(self.change_model)  # 发送信号

        self.FileBtn = QPushButton('Select Video')
        self.FileBtn.clicked.connect(self.open_file)

        self.left_label = QLabel()
        self.left_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.left_label.setFixedWidth(640)
        self.left_label.setFixedHeight(640)

        self.right_label = QLabel()
        self.right_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.right_label.setFixedWidth(640)
        self.right_label.setFixedHeight(640)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.left_label)
        self.splitter.addWidget(self.right_label)

        hbox = QHBoxLayout()
        hbox.addWidget(self.FileBtn)
        hbox.addWidget(self.comboBox)
        hbox.addWidget(self.playBtn)
        hbox.addWidget(self.pauseBtn)
        hbox.addWidget(self.replayBtn)

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

    def show_image(self, img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()

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

    def show_image2(self, img_src, label):
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
            self.right_label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def play(self):
        self.yolothread.jump_out = False
        self.yolothread.is_continue = True

        self.Deepsort_thread.jump_out = False
        self.Deepsort_thread.is_continue = True

        self.pauseBtn.setEnabled(True)
        self.playBtn.setEnabled(False)
        self.replayBtn.setEnabled(True)

        self.yolothread.start()
        self.Deepsort_thread.start()

    def pause(self):
        self.yolothread.is_continue = False
        self.Deepsort_thread.is_continue = False

        self.pauseBtn.setEnabled(False)
        self.playBtn.setEnabled(True)
        self.replayBtn.setEnabled(True)

    def replay(self):
        self.yolothread.jump_out = True
        self.Deepsort_thread.jump_out = True

        self.replayBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.playBtn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
