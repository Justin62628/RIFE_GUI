# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SVFI_about.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(620, 541)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea = QtWidgets.QScrollArea(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 600, 492))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setScaledContents(True)
        self.label_7.setWordWrap(True)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_3.addWidget(self.label_7)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_7.setText(_translate("Dialog", "<html><head/><body><p><span style=\"\n"
"                                            font-size:12pt; color:#ffffff;\">本软件 </span><span style=\"\n"
"                                            font-size:12pt; font-style:italic; color:#ffffff;\">Squirrel Video\n"
"                                            Frame Interpolation</span></p><p><span style=\"\n"
"                                            color:#ffffff;\">是基于</span><span style=\"\n"
"                                            font-weight:600; color:#ffffff;\">RIFE: Real-Time Intermediate Flow\n"
"                                            Estimation for Video Frame Interpolation</span><span style=\"\n"
"                                            color:#ffffff;\"> AI补帧算法的可视化图形界面集成.</span></p><p><br/></p><p><span\n"
"                                            style=\" font-weight:600; color:#ffffff;\">RIFE算法作者</span><span\n"
"                                            style=\" color:#ffffff;\"> Zhewei Huang, Tianyuan Zhang, Wen\n"
"                                            Heng, Boxin Shi, Shuchang Zhou </span></p><p><span\n"
"                                            style=\" color:#ffffff;\">https://github.com/hzwer/arXiv2020-RIFE</span></p><p><span\n"
"                                            style=\" font-weight:600; color:#ffffff;\">SVFI作者</span><span\n"
"                                            style=\" color:#ffffff;\"> YiWeiHuang-stack, Justin62628, 穆氏,\n"
"                                            ABlyh-LEO, NULL204</span></p><p><span style=\"\n"
"                                            color:#ffffff;\">https://github.com/YiWeiHuang-stack/Squirrel-RIFE</span></p><p><span\n"
"                                            style=\" color:#ffffff;\">https://github.com/Justin62628/RIFE_GUI</span></p><p><span\n"
"                                            style=\" text-decoration: underline; color:#ffffff;\">其中RIFE\n"
"                                            GUI已与SVFI版本合并</span></p><p><span style=\"\n"
"                                            color:#ffffff;\">市面上的SVFI 1.x, RIFE GUI 6.2.x均为旧版本</span></p></body></html>\n"
"                                        "))
