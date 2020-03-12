# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'start.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from GUI import MainWindow

class Ui_Form(QMainWindow):
    def __init__(self):
        super(Ui_Form,self).__init__()
        self.setObjectName("Form")
        self.resize(683, 414)
        self.background = QtWidgets.QLabel(self)
        self.background.setGeometry(QtCore.QRect(0, 0, 683, 414))
        self.background.setText("")
        self.background.setPixmap(QtGui.QPixmap("1.png"))
        self.background.setObjectName("background")
        self.start = QtWidgets.QPushButton(self)
        self.start.setGeometry(QtCore.QRect(492, 360, 61, 28))
        self.start.setStyleSheet("font: 75 10pt \"Arial\";")
        self.start.setObjectName("start")
        self.retranslateUi(self)
        # self.start.clicked.connect(self.getstart)
        QtCore.QMetaObject.connectSlotsByName(self)

    # def getstart(self):
    #     new = contour()
    #     self.close()
    #     new.show2()

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "主界面"))
        self.start.setText(_translate("Form", "start"))
    def close1(self):
        self.close()

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    ui = Ui_Form()
    new = MainWindow.GUI_mainView()
    ui.show()
    ui.start.clicked.connect(new.show2)
    ui.start.clicked.connect(ui.close1)
    sys.exit(app.exec_())