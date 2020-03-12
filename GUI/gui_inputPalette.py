from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import skimage as color
import matplotlib.pyplot as plt


class InputPalette(QWidget):
    updateUsedColorSignal = pyqtSignal([int, int, int])

    def __init__(self, win_size=256):
        super(InputPalette, self).__init__()
        self.win_size = win_size
        self.img_loaded = False
        self.img_path = None
        self.setFixedSize(win_size, win_size)
        self.use_gray = True
        self.user_edit = True
        self.pos = None
        self.user_points = []
        self.user_points_real = []
        self.user_colors = []
        self.img_l = None
        self.local_input = None

    # 切换灰度图显示
    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()

    # 允许用户输入
    def enable_userEdit(self):
        self.user_edit = not self.user_edit
        self.update()

    # 清除用户输入
    def clear_userEdit(self):
        self.user_points.clear()
        self.user_points_real.clear()
        self.user_colors.clear()

    # 加载图像
    def load(self):
        img_path, flag = QFileDialog.getOpenFileName(self, 'load an input image', './images/',
                                                     'All files(*.*);;Image files(*.jpg *.png *.bmp)')
        if flag is not None:
            self.img_loaded = True
            self.clear_userEdit()
            # 载入原图像
            self.img_path = img_path
            self.img_ori = cv2.imread(self.img_path, flags=cv2.IMREAD_COLOR)    # 默认读彩色图(0~255)
            self.img_ori = cv2.cvtColor(self.img_ori, cv2.COLOR_BGR2RGB)        # BGR转RGB
            self.img_gray = cv2.cvtColor(self.img_ori, cv2.COLOR_RGB2GRAY)      # RGB转灰度图
            self.img_gray = cv2.cvtColor(self.img_gray, cv2.COLOR_GRAY2RGB)

            # 获取用于窗口显示的图像
            self.height_ori = self.img_ori.shape[0]
            self.width_ori = self.img_ori.shape[1]
            max_width = max([self.height_ori, self.width_ori])
            self.scale = self.win_size / float(max_width)
            self.height_scaled = int(round(self.height_ori * self.scale))
            self.width_scaled = int(round(self.width_ori * self.scale))
            self.dh = int((self.win_size - self.height_scaled) // 2)        # 图像居中需要的位移
            self.dw = int((self.win_size - self.width_scaled) // 2)
            self.ori_win = cv2.resize(self.img_ori, (self.width_scaled, self.height_scaled), interpolation=cv2.INTER_LANCZOS4)
            self.gray_win = cv2.resize(self.img_gray, (self.width_scaled, self.height_scaled), interpolation=cv2.INTER_LANCZOS4)

            self.update()
        else:
            print('没有选择文件')

        return self.img_loaded

    # 验证点的位置是否合法
    def valid_point(self, pos):
        if pos is not None:
            if self.dw <= pos.x() < self.win_size - self.dw and \
                    self.dh <= pos.y() < self.win_size - self.dh:
                x = int(np.round(pos.x()))
                y = int(np.round(pos.y()))
                return QPoint(x, y)
            else:
                QMessageBox.warning(self, 'Warning', 'Invalid point position.')
                return None
        else:
            QMessageBox.warning(self, 'Warning', 'Invalid point position.')
            return None

    # 缩放点的坐标
    def scale_point(self, pos):
        x = int((pos.x() - self.dw) / float(self.width_scaled) * self.width_ori)
        y = int((pos.y() - self.dh) / float(self.height_scaled) * self.height_ori)
        return QPoint(x, y)

    # 添加用户编辑点
    def add_userPoints(self, pos=None):
        if pos is not None:
            col = QColorDialog.getColor()
            if col.isValid():
                self.user_points.append(pos)
                pos_real = self.scale_point(pos)
                self.user_points_real.append(pos_real)
                self.user_colors.append(col)
                self.updateUsedColorSignal.emit(col.red(), col.green(), col.blue())
            else:
                print('没有选择颜色')

    # 改变当前点的颜色
    def change_pointColor(self, col):
        if col.isValid() and len(self.user_colors) > 0:
            self.user_colors.pop()
            self.user_colors.append(col)
            self.update()

    # 撤销一个用户编辑点
    def undo_userPoints(self):
        if len(self.user_points_real) > 0:
            self.user_points.pop()
            self.user_points_real.pop()
            self.user_colors.pop()
            self.update()
        else:
            QMessageBox.warning(self, 'Warning', 'There is no user edit points anymore.')

    # 获取用户输入
    def get_localInput(self, used=True):
        if used is True:
            if len(self.user_points_real) > 0:
                # localInput = np.ones((self.height_ori, self.width_ori, 3), dtype=np.uint8) * 255
                localInput = np.zeros((self.height_ori, self.width_ori, 3), dtype=np.uint8)
                localInputMask = np.zeros((self.height_ori, self.width_ori, 1), dtype=np.uint8)
                for i in range(len(self.user_points_real)):
                    x = self.user_points_real[i].x()
                    y = self.user_points_real[i].y()
                    localInput[y, x, :] = [self.user_colors[i].red(), self.user_colors[i].green(), self.user_colors[i].blue()]
                    localInputMask[y, x, 0] = 1
                localInput_lab = color.rgb2lab(localInput)
                localInput_ab = (localInput_lab[:, :, 1:] + 128.0) / 255.0 * 2 - 1
                self.local_input = np.concatenate((localInput_ab, localInputMask), axis=2)
                self.local_input = np.reshape(self.local_input, [1, self.height_ori, self.width_ori, 3])
                print('局部输入', self.local_input.shape)
                return True
            else:
                print('用户输入为空')
                return False
        else:
            if self.img_loaded is True:
                localInput = np.zeros((self.height_ori, self.width_ori, 3), dtype=np.uint8)
                localInputMask = np.zeros((self.height_ori, self.width_ori, 1), dtype=np.uint8)
                localInput_lab = color.rgb2lab(localInput)
                localInput_ab = (localInput_lab[:, :, 1:] + 128.0) / 255.0 * 2 - 1
                self.local_input = np.concatenate((localInput_ab, localInputMask), axis=2)
                self.local_input = np.reshape(self.local_input, [1, self.height_ori, self.width_ori, 3])
                print('生成空白局部输入', self.local_input.shape)
                return True
            return False

    # 获取输入图像
    def get_inputLChannel(self):
        if self.img_loaded is True:
            img_ori_lab = color.rgb2lab(self.img_ori)
            self.img_l = np.reshape(img_ori_lab[:, :, 0], [self.height_ori, self.width_ori, 1])
            self.img_l = self.img_l.astype(np.float32) / 100.0 * 2 - 1
            self.img_l = np.reshape(self.img_l, [1, self.height_ori, self.width_ori, 1])
            return True
        else:
            return False

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

