import torch
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import torch.backends.cudnn as cudnn
from utils.dataloaders import LoadImages
from models.common import DetectMultiBackend
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.general import (check_img_size, cv2, non_max_suppression)


class DisplayImage:
    '''用于展示选择的图片'''

    def __init__(self, master):
        self.master = master  # 设置一个TK对象
        self.image_frame = Frame(master, bd=0, height=600, width=800, bg='white', highlightthickness=2,
                                 highlightbackground='gray', highlightcolor='black')  # 设置展示图片的容器
        self.image_frame.place(x=250, y=100)
        self.Choose_image = Button(master, command=self.choose_pic, text="选择图片", height=2, width=15, default=ACTIVE,
                                   borderwidth=0)  # 设置按钮并绑定选择图片函数
        self.Choose_image.place(x=80, y=220)
        self.Choose_model = Button(master, text="加载模型", command=self.load_model, height=2, width=15, default=ACTIVE,
                                   borderwidth=0)  # 设置按钮并绑定加载模型函数
        self.Choose_model.place(x=80, y=370)
        self.Choose_detect = Button(master, text="检测", command=self.detect_pic, height=2, width=15, default=ACTIVE,
                                    borderwidth=0)  # 设置按钮并绑定检测函数
        self.Choose_detect.place(x=80, y=520)

        self.filenames = []  # 设置存放图片路径的空列表
        self.pic_filelist = []  # 设置存放PIL对象空列表
        self.imgt_list = []  # 设置存放Tkinter兼容图像的列表
        self.image_labellist = []  # 设置存放展示图片的空列表
        self.model_list = []  # 设置存放模型路径的空列表
        self.model_list1 = []  # 设置存放模型的空列表

    def choose_pic(self):
        self.filenames.clear()
        self.filenames += filedialog.askopenfilenames()  # 打开文件

        self.pic_filelist.clear()  # 在重新选择图片时清空原先列表
        self.imgt_list.clear()
        self.image_labellist.clear()

        for widget in self.image_frame.winfo_children():  # 清空框架中的内容
            widget.destroy()

        for i in range(len(self.filenames)):
            self.pic_filelist.append(Image.open(self.filenames[i]))
            self.imgt_list.append(ImageTk.PhotoImage(image=self.pic_filelist[i].resize(
                (int(self.pic_filelist[i].size[0] * 0.6), int(self.pic_filelist[i].size[1] * 0.6)))))
            self.image_labellist.append(Label(self.image_frame, highlightthickness=0, borderwidth=0))
            self.image_labellist[i].configure(image=self.imgt_list[i])
            self.image_labellist[i].pack(side=LEFT, expand=True)

    def load_model(self):
        self.model_list.clear()
        self.model_list1.clear()
        self.model = filedialog.askopenfilenames()  # 选择模型所在文件夹
        self.model_list.append(self.model)
        model = DetectMultiBackend(self.model_list[0][0], device=select_device('0'), dnn=False,
                                   data='data/coco128.yaml', fp16=False)  # 模型载入
        self.model_list1.append(model)

    def detect_pic(self):
        self.imgt_list.clear()  # 在重新检测时清空原先列表
        self.image_labellist.clear()
        for widget in self.image_frame.winfo_children():  # 清空框架中的内容
            widget.destroy()

        stride, names, pt = self.model_list1[0].stride, self.model_list1[0].names, self.model_list1[0].pt  # yolov5源码
        imgsz = check_img_size((640, 640), s=stride)
        cudnn.benchmark = True
        dataset = LoadImages(self.filenames[0], img_size=imgsz, stride=stride, auto=pt)
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(select_device('0'))
            im = im.half() if self.model_list1[0].fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        pred = self.model_list1[0](im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.5, 0.45, None, False, max_det=1000)
        for i, det in enumerate(pred):
            seen += 1
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=2, example=str(names))
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)  # 图片通道转换
            im0 = Image.fromarray(im0)  # 数组格式转为PIL格式
            for m, n in zip(self.pic_filelist, range(len(self.pic_filelist))):  # 布局所选图片展示在容器中
                self.imgt_list.append(
                    ImageTk.PhotoImage(image=im0.resize((int(im0.size[0] * 0.6), int(im0.size[1] * 0.6)))))
                self.image_labellist.append(Label(self.image_frame, highlightthickness=0, borderwidth=0))
                self.image_labellist[n].configure(image=self.imgt_list[n])
                self.image_labellist[n].pack(side=LEFT, expand=True)


def main():
    master = tk.Tk()  # 创建tk对象
    GUI = DisplayImage(master)
    master.title('Detector')  # 窗口命名
    master.geometry('1500x900')  # 窗口大小设置
    master.mainloop()  # 消息循环


if __name__ == '__main__':
    main()
