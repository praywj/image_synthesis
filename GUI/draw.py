import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from GUI import MainWindow
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt

class Mainform(QWidget):
    vis = 0
    def __init__(self, parent=None):
        super( Mainform, self).__init__(parent)
        self.setWindowTitle("edit contour")
        self.resize(300, 350)
        layout = QVBoxLayout()
        self.paint = Winform()
        # draw.addWidget(self.label)
        layout.addWidget(self.paint)

        self.area = QSlider(Qt.Horizontal)
        self.area.setMinimum(20)
        self.area.setMaximum(60)
        self.area.setSingleStep(5)
        self.area.setValue(25)
        # 刻度位置 刻度在下方
        self.area.setTickPosition(QSlider.TicksBelow)
        self.area.setTickInterval(5)
        layout.addWidget(self.area)

        button = QHBoxLayout()
        layout.addLayout(button)
        self.blackButton = QPushButton("draw")
        self.whiteButton = QPushButton("clear")
        self.dragButton = QPushButton("drag")
        self.quitButton = QPushButton("quit")
        button.addStretch(1)
        button.addWidget(self.blackButton)
        button.addStretch(1)
        button.addWidget(self.whiteButton)
        button.addStretch(1)
        button.addWidget(self.dragButton)
        button.addStretch(1)
        layout.addWidget(self.quitButton)
        self.setLayout(layout)

        self.blackButton.clicked.connect(self.black)
        self.whiteButton.clicked.connect(self.white)
        self.quitButton.clicked.connect(self.quit)
        self.dragButton.clicked.connect(self.drag)
        self.area.valueChanged.connect(self.valuechange)
    def valuechange(self):
        a = self.area.value()
        self.paint.getarea(a)
        # print(self.area.value())
    def drag(self):
        self.flag = 0
        self.paint.getflag(self.vis, self.flag)
    def black(self):
        self.vis = 0.0
        self.flag = 1
        self.paint.getflag(self.vis,self.flag)
        self.paint.setPen(QPen(Qt.black))
    def white(self):
        self.vis = 1.0
        self.flag = 1
        self.paint.getflag(self.vis,self.flag)
        self.paint.setPen(QPen(Qt.white))
    def quit(self):
        print('quit')
        reply = QMessageBox.information(self, "Notice!", "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if(reply == 16384):
            # self.main = MainWindow.GUI_mainView()
            # self.main.resultshow('F:\\test\\data\\contour.png')
            self.close()
            # self.main.show2()
    def showdraw(self,name):
        self.show()
        # print(name)
        self.paint.get(name)

class Winform(QWidget):
    def __init__(self):
        super(Winform, self).__init__()
        self.pix = QPixmap()
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.pen = QPen()
        self.pix = QPixmap(256, 256)
        self.cur = 0
        self.k = 25
        self.draw = 0
        self.color = 0
        self.R = np.zeros((self.k, self.k), dtype=np.double)
        self.Rr = np.zeros((self.k, self.k, 3), dtype=np.double)
        self.img = plt.imread('F:\\test\\data\\contour.png')
        pp = QPainter(self.pix)
        pp.drawPixmap(0,0,256,256,QPixmap(r'F:\\test\\data\\contour.png'))

    def getarea(self,a):
        self.k = a
        self.R = np.zeros((self.k, self.k), dtype=np.double)
        self.Rr = np.zeros((self.k, self.k, 3), dtype=np.double)
    def getflag(self,vis,flag):
        self.draw = flag
        self.color = vis
    def setPen(self, p):
        self.pen = p
        self.update()
    def get(self,name):
        self.name = name
        self.image = plt.imread(self.name)
    def paintEvent(self, event):
        pp = QPainter(self.pix)
        pp.setPen(self.pen)
        # 根据鼠标指针前后两个位置绘制直线
        pp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)

    def mousePressEvent(self, event):
        # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            if self.draw == 0 :
                self.cur = self.cur + 1
                if x>=0 and x+self.k <=256 and y>=0 and y+self.k <=256:

                   if self.cur % 2 == 1:

                       self.R[0:self.k,0:self.k] = self.img[y:y+self.k,x:x+self.k]
                       self.Rr[0:self.k, 0:self.k, :] = self.image[y:y+self.k, x:x+self.k, :]
                       self.image[y:y + self.k, x:x + self.k, :] = 0
                       self.img[y:y + self.k, x:x + self.k] = 1.0
                       self.update()
                       # self.img[y, x] = self.cur
                       # plt.imsave('F:\\test\\data\\b.png', self.R, cmap='gray')
                       # plt.imsave('F:\\test\\data\\d.png', self.Rr)
                   else:
                       self.img[y:y+self.k,x:x+self.k] = self.R[0:self.k,0:self.k]
                       self.image[y:y + self.k, x:x + self.k, :] = self.Rr[0:self.k,0:self.k,:]
                       plt.imsave('F:\\test\\data\\contour.png', self.img, cmap='gray')
                       lena = Image.open('F:\\test\\data\\contour.png')
                       lena_L = lena.convert("L")
                       lena_L.save('F:\\test\\data\\contour.png')
                       # print(self.name)
                       plt.imsave(self.name, self.image)
                       lena = Image.open(self.name)
                       lena_L = lena.convert("RGB")
                       lena_L.save(self.name)
                else:
                    QMessageBox.warning(self, 'Warning', '超出编辑范围!.')
            else:
                self.lastPoint = event.pos()
                self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):
        # 鼠标左键按下的同时移动鼠标

        if event.buttons() and Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            if self.draw == 1:
                if x >= 0 and x <= 256 and y >= 0 and y <= 256:
                    self.endPoint = event.pos()
                    self.img[y,x] = self.color
                    plt.imsave('F:\\test\\data\\contour.png', self.img, cmap='gray')
                    lena = Image.open('F:\\test\\data\\contour.png')
                    lena_L = lena.convert("L")
                    lena_L.save('F:\\test\\data\\contour.png')
                else:
                    QMessageBox.warning(self, 'Warning', '超出编辑范围!.')
                # 进行重新绘制
                self.update()

    def mouseReleaseEvent(self, event):
        # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            # self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     form = Mainform()
#     form.show()
#     # send = Mainform()
#     # slot = Winform()
#     # send.sendmsg.connect(slot.get)
#     # send.run()
#     sys.exit(app.exec_())