from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import skimage as color
import matplotlib.pyplot as plt


class InputPalette(QWidget):

    def __init__(self, win_size=256):
        super(InputPalette, self).__init__()
        self.win_size = win_size
        self.img_loaded = False
        self.img_path = None
        self.setFixedSize(win_size, win_size)
        self.use_gray = False
        self.user_edit = True
        self.pos = None
        self.user_points = []
        self.user_points_real = []
        self.user_colors = []
        self.img_l = None
        self.local_input = None


    # 加载图像
    def load(self,name):
        img_path, flag = QFileDialog.getOpenFileName(self, 'load an input image', name,
                                                     ';Image files(*.jpg *.png *.bmp)')
        print(img_path)
        if img_path != '':
            self.img_loaded = True
            # 载入原图像
            self.img_path = img_path
            self.img_ori = cv2.imread(self.img_path, flags=cv2.IMREAD_COLOR)  # 默认读彩色图(0~255)
            self.img_ori = cv2.cvtColor(self.img_ori, cv2.COLOR_BGR2RGB)  # BGR转RGB
            self.img_gray = cv2.cvtColor(self.img_ori, cv2.COLOR_RGB2GRAY)  # RGB转灰度图
            self.img_gray = cv2.cvtColor(self.img_gray, cv2.COLOR_GRAY2RGB)
            # 获取用于窗口显示的图像
            self.height_ori = self.img_ori.shape[0]
            self.width_ori = self.img_ori.shape[1]
            max_width = max([self.height_ori, self.width_ori])
            self.scale = self.win_size / float(max_width)
            self.height_scaled = int(round(self.height_ori * self.scale))
            self.width_scaled = int(round(self.width_ori * self.scale))
            self.dh = int((self.win_size - self.height_scaled) // 2)  # 图像居中需要的位移
            self.dw = int((self.win_size - self.width_scaled) // 2)
            self.ori_win = cv2.resize(self.img_ori, (self.width_scaled, self.height_scaled),
                                      interpolation=cv2.INTER_LANCZOS4)
            self.gray_win = cv2.resize(self.img_gray, (self.width_scaled, self.height_scaled),
                                       interpolation=cv2.INTER_LANCZOS4)
            self.update()
        else:
            QMessageBox.warning(self, 'Warning', '没有选择文件.')

        return self.img_loaded, self.img_path

    # 切换灰度图显示
    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()
    # TODO: 重载事件响应函数
    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(QPaintEvent.rect(), Qt.white)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.img_loaded:
            if self.use_gray:
                im = self.gray_win
            else:
                im = self.ori_win
            qImg = QImage(im.tostring(), self.width_scaled, self.height_scaled, QImage.Format_RGB888)
            painter.drawImage(self.dw, self.dh, qImg)

            if self.user_edit:
                for i in range(len(self.user_colors)):
                    ca = QColor(self.user_colors[i].red(), self.user_colors[i].green(), self.user_colors[i].blue(), 255)
                    painter.setBrush(ca)
                    painter.drawEllipse(self.user_points[i].x(), self.user_points[i].y(), 5, 5)

        painter.end()

    def mousePressEvent(self, QMouseEvent):
        if self.img_loaded and self.user_edit:
            pos = self.valid_point(QMouseEvent.pos())
            if pos is not None:
                self.pos = pos
                if QMouseEvent.button() == Qt.LeftButton:
                    self.add_userPoints(self.pos)

