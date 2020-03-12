from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from GUI import gui_input
from GUI import gui_resultPalette
from GUI import getInput
from models import train
from GUI import draw
class GUI_mainView(QWidget):
    strr = 'Used data'
    def __init__(self):
        super(GUI_mainView, self).__init__()
        self.setWindowTitle('Interactive Image Synthesis Interface')
        self.setFixedSize(1360, 604)
        self.reuse_model = False
        self.background = QtWidgets.QLabel(self)
        self.background.setGeometry(QtCore.QRect(0, 0, 1360, 604))
        self.background.setText("")
        self.background.setPixmap(QtGui.QPixmap("back.png"))
        # 界面
        print('载入界面...')
        self.init_UI()
        print('界面载入成功！\n------------------------------------------------\n')

    # 初始化界面
    def init_UI(self):
        # main layout
        mainLayout = QHBoxLayout()
        self.setLayout(mainLayout)
        # colorPalette layout
        colorPaletteLayout = QVBoxLayout()
        mainLayout.addLayout(colorPaletteLayout)

        colorMenuLayout = QVBoxLayout()
        colorPaletteLayout.addLayout(colorMenuLayout)
        self.colorCheckBox = QCheckBox('select threshold')
        self.label_3 = QtWidgets.QLabel("\n")
        self.colorCheckBox.setChecked(False)
        colorMenuLayout.addWidget(self.colorCheckBox)
        colorMenuLayout.addWidget(self.label_3)

        self.label = QtWidgets.QLabel("min of threshold:")
        self.label.setAlignment(Qt.AlignCenter)
        colorMenuLayout.addWidget(self.label)
        #设置滑动条
        #水平方向
        self.min = QSlider(Qt.Horizontal)
        self.min.setMinimum(0)
        self.min.setMaximum(35)
        self.min.setSingleStep(5)
        self.min.setValue(8)
        #刻度位置 刻度在下方
        self.min.setTickPosition(QSlider.TicksBelow)
        self.min.setTickInterval(5)
        colorMenuLayout.addWidget(self.min)
        self.label_5 = QtWidgets.QLabel("\n")
        colorMenuLayout.addWidget(self.label_5)
        #连接信号槽
        self.min.valueChanged.connect(self.valuechange1)
        self.label_2 = QtWidgets.QLabel("max of threshold:")
        self.label_2.setAlignment(Qt.AlignCenter)
        colorMenuLayout.addWidget(self.label_2)
        self.max = QSlider(Qt.Horizontal)
        self.max.setMinimum(0)
        self.max.setMaximum(40)
        self.max.setSingleStep(5)
        self.max.setValue(10)
        # 刻度位置 刻度在下方
        self.max.setTickPosition(QSlider.TicksBelow)
        self.max.setTickInterval(5)
        colorMenuLayout.addWidget(self.max)
        self.max.valueChanged.connect(self.valuechange2)
        self.label_4 = QtWidgets.QLabel("\n")
        colorMenuLayout.addWidget(self.label_4)

        # 4. 提示信息
        self.msgHint = QPlainTextEdit()
        self.msgHint.setFixedSize(220, 280)
        self.msgHint.setReadOnly(True)
        self.msgHintScroll = QScrollArea()
        self.msgHintScroll.setAutoFillBackground(True)
        self.msgHintScroll.setWidgetResizable(True)
        self.msgHintScroll.setWidget(self.msgHint)
        msgHintLayout = self.addWidget(self.msgHintScroll, 'Info')
        colorPaletteLayout.addLayout(msgHintLayout)
        GUI_mainView.strr = GUI_mainView.strr + '\nmin of threshold is :' + str(self.min.value() * 10)
        GUI_mainView.strr = GUI_mainView.strr + '\nmax of threshold is :' + str(self.max.value() * 10)
        self.msgHint.setPlainText(GUI_mainView.strr)
        self.min.setEnabled(False)
        self.max.setEnabled(False)
        self.msgHint.setEnabled(False)
        # colorThemeLayout = self.addWidget(self, 'Color Theme')
        # colorPaletteLayout.addLayout(colorThemeLayout)

        # inputPalette layout
        inputPaletteLayout = QVBoxLayout()
        mainLayout.addLayout(inputPaletteLayout)
        # 1. 图像显示面板
        self.inputPalette = gui_input.InputPalette(win_size=512)
        inputImageLayout = self.addWidget(self.inputPalette, 'Image')
        inputPaletteLayout.addLayout(inputImageLayout)
        # 2. 菜单栏
        inputMenuLayout = QGridLayout()
        inputPaletteLayout.addLayout(inputMenuLayout)
        self.grayCheckBox = QCheckBox('Gray')
        self.grayCheckBox.setChecked(False)
        self.loadButton = QPushButton('&LoadImage')
        self.contourButton = QPushButton('&contour')
        self.colorButton = QPushButton('&color')
        inputMenuLayout.addWidget(self.grayCheckBox, 0, 0, 1, 1)
        inputMenuLayout.addWidget(self.loadButton, 0, 1, 1, 1)
        inputMenuLayout.addWidget(self.contourButton, 0, 2, 1, 1)
        inputMenuLayout.addWidget(self.colorButton, 0, 3, 1, 1)

        # resultPalete layout
        resultPaletteLayout = QVBoxLayout()
        mainLayout.addLayout(resultPaletteLayout)

        # 1. 图像显示面板
        self.resultPalette = gui_resultPalette.ResultPalette(win_size=512)
        resultImageLayout = self.addWidget(self.resultPalette, 'Result')
        resultPaletteLayout.addLayout(resultImageLayout)
        # 2. 菜单栏
        resultMenuLayout = QHBoxLayout()
        resultPaletteLayout.addLayout(resultMenuLayout)
        self.graButton = QPushButton('梯度')
        self.colorizeButton = QPushButton('&Synthetic')
        self.editButton = QPushButton('&edit')
        self.quitButton = QPushButton('&Refresh')
        # self.gray1CheckBox = QCheckBox('Gray')
        # self.gray1CheckBox.setChecked(False)
        resultMenuLayout.addWidget(self.graButton)
        resultMenuLayout.addWidget(self.colorizeButton)
        resultMenuLayout.addWidget(self.editButton)
        resultMenuLayout.addWidget(self.quitButton)
        # resultMenuLayout.addWidget(self.gray1CheckBox)
        self.contourButton.setEnabled(False)
        self.colorButton.setEnabled(False)
        self.colorizeButton.setEnabled(False)
        self.graButton.setEnabled(False)
        self.editButton.setEnabled(False)
        # 事件响应
        self.colorCheckBox.stateChanged.connect(self.enable_color)
        self.loadButton.clicked.connect(self.load)
        self.contourButton.clicked.connect(self.contour)
        self.colorButton.clicked.connect(self.color)
        # self.new = draw.Mainform()
        self.editButton.clicked.connect(self.edit)
        self.colorizeButton.clicked.connect(self.Synthetic)
        # self.grayCheckBox.stateChanged.connect(self.enable_gray)
        # self.gray1CheckBox.stateChanged.connect(self.enable_gray1)
        self.graButton.clicked.connect(self.gradient)
        self.quitButton.clicked.connect(self.quit)

        # 自定义信号
    def edit(self):
        self.new = draw.Mainform()
        self.new.showdraw(self.img_path)

     # 允许用户修改阈值
    def enable_color(self):
        print('enable_threshold')
        if self.colorCheckBox.isChecked():
            self.min.setEnabled(True)
            self.max.setEnabled(True)
            self.msgHint.setEnabled(True)
        else:
            self.min.setEnabled(False)
            self.max.setEnabled(False)
            self.msgHint.setEnabled(False)
    def enable_gray(self):
        print('enable_gray')
        self.inputPalette.enable_gray()
    def load(self):
        print('LoadImage')
        self.isLoad, self.img_path = self.inputPalette.load('I:\\')
        if self.isLoad:
            self.editButton.setEnabled(True)
            self.colorButton.setEnabled(True)
            self.contourButton.setEnabled(True)
            self.colorizeButton.setEnabled(True)
            self.graButton.setEnabled(True)
        else:
            self.editButton.setEnabled(False)
            self.colorButton.setEnabled(False)
            self.contourButton.setEnabled(False)
            self.colorizeButton.setEnabled(False)
            self.graButton.setEnabled(False)
    def contour(self):
        print('contour %d %d',self.min.value()*10 , self.max.value()*10)
        getInput.getContour(self.img_path, self.min.value() * 10, self.max.value() * 10)
        self.resultPalette.load('F:\\test\\data\\contour.png')
    def color(self):
        print('color')
        getInput.getColor(self.img_path)
        self.resultPalette.load('F:\\test\\data\\color.png')
    def Synthetic(self):
        print(' Synthetic')
        train.eval()
        self.resultPalette.load('F:\\test\\data\\result.png')
    def gradient(self):
        print('gradient')
        getInput.getGradient(self.img_path)
        self.resultPalette.load('F:\\test\\data\\gradient.png')
    def quit(self):
        # print('quit')
        # reply = QMessageBox.information(self, "Notice!", "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        # if(reply == 16384):
        #     self.close()
        QApplication.processEvents()
        self.resultPalette.load('F:\\test\\data\\contour.png')
    def show2(self):
        self.show()

    # 添加group封装的组件
    def addWidget(self, widget, title):
        widgetLayout = QVBoxLayout()
        widgetBox = QGroupBox()
        widgetBox.setTitle(title)
        boxLayout = QVBoxLayout()
        boxLayout.addWidget(widget)
        widgetBox.setLayout(boxLayout)
        widgetLayout.addWidget(widgetBox)
        return widgetLayout
    def valuechange1(self):
        #GUI_mainView.strr = GUI_mainView.strr.join(GUI_mainView.s)
        #print(str(self.min.value()))
        GUI_mainView.strr = GUI_mainView.strr + '\nmin of threshold is :' +str(self.min.value()*10)
        self.msgHint.setPlainText(GUI_mainView.strr)

    def valuechange2(self):
        #GUI_mainView.strr = GUI_mainView.strr.join(GUI_mainView.s)
        #print(str(self.min.value()))
        GUI_mainView.strr = GUI_mainView.strr + '\nmax of threshold is :' +str(self.max.value()*10)
        self.msgHint.setPlainText(GUI_mainView.strr)





#
# if __name__ == '__main__':
#     import sys
#     app = QApplication(sys.argv)
#     widget = QWidget()
#     ui = GUI_mainView()
#     ui.show()
#     # new = draw.Mainform()
#     # ui.editButton.clicked.connect(new.show)
#     sys.exit(app.exec_())