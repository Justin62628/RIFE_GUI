# -*- coding: utf-8 -*-
import json
import random

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from Utils.utils import Tools


class MyListWidgetItem(QWidget):
    dupSignal = pyqtSignal(dict)
    remSignal = pyqtSignal(dict)

    def __init__(self, parent=None):
        """
        Custom ListWidgetItem to display RIFE Task
        :param parent:
        """
        super().__init__(parent)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.filename = QtWidgets.QLabel(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filename.sizePolicy().hasHeightForWidth())
        self.filename.setSizePolicy(sizePolicy)
        self.filename.setMinimumSize(QSize(400, 0))
        self.filename.setObjectName("filename")
        self.horizontalLayout.addWidget(self.filename)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.line = QtWidgets.QFrame(self)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.RemoveItemButton = QtWidgets.QPushButton(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RemoveItemButton.sizePolicy().hasHeightForWidth())
        self.task_id_display = QtWidgets.QLabel(self)
        self.task_id_display.setSizePolicy(sizePolicy)
        self.task_id_display.setObjectName("task_id_display")
        self.horizontalLayout.addWidget(self.task_id_display)
        self.RemoveItemButton.setSizePolicy(sizePolicy)
        self.RemoveItemButton.setObjectName("RemoveItemButton")
        self.horizontalLayout.addWidget(self.RemoveItemButton)
        self.DuplicateItemButton = QtWidgets.QPushButton(self)
        self.DuplicateItemButton.setSizePolicy(sizePolicy)
        self.DuplicateItemButton.setObjectName("DuplicateItemButton")
        self.horizontalLayout.addWidget(self.DuplicateItemButton)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.RemoveItemButton.setText("    -    ")
        self.DuplicateItemButton.setText("    +    ")
        self.RemoveItemButton.clicked.connect(self.on_RemoveItemButton_clicked)
        self.DuplicateItemButton.clicked.connect(self.on_DuplicateItemButton_clicked)
        """Item Data Settings"""
        self.task_id = None
        self.input_path = None

    def setTask(self, input_path: str, task_id: str):
        self.task_id = task_id
        self.input_path = input_path
        len_cut = 100
        if len(self.input_path) > len_cut:
            self.filename.setText(self.input_path[:len_cut] + "...")
        else:
            self.filename.setText(self.input_path)
        self.task_id_display.setText(f"  id: {self.task_id}  ")

    def on_DuplicateItemButton_clicked(self, e):
        """
        Duplicate Item Button clicked
        action:
            1: duplicate
            0: remove
        :param e:
        :return:
        """
        self.dupSignal.emit({"task_id": self.task_id, "input_path": self.input_path, "action": 1})
        pass

    def on_RemoveItemButton_clicked(self, e):
        self.dupSignal.emit({"task_id": self.task_id, "input_path": self.input_path, "action": 0})
        pass


class MyLineWidget(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            url = e.mimeData().urls()[0]
            if not len(self.text()):
                self.setText(url.toLocalFile())
        else:
            e.ignore()


class MyListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.task_dict = list()
        self.setDragDropMode(QAbstractItemView.InternalMove)

    def generateTaskId(self, input_path: str):
        path_md5 = Tools.md5(input_path)[:6]
        while True:
            path_id = random.randrange(100000, 999999)
            if path_id not in self.task_dict:
                self.task_dict.append(path_id)
                break
        task_id = f"{path_md5}_{path_id}"
        return task_id

    def saveTasks(self):
        """
        return tasks information in strings of json format
        :return: {"inputs": [{"task_id": self.task_id, "input_path": self.input_path}]}
        """
        data = list()
        for item in self.get_items():
            widget = self.itemWidget(item)
            item_data = {"task_id": widget.task_id, "input_path": widget.input_path}
            data.append(item_data)
        return json.dumps({"inputs": data})

    def dropEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            for url in e.mimeData().urls():
                item = url.toLocalFile()
                self.addFileItem(item)
        else:
            e.ignore()

    def dragEnterEvent(self, e):
        self.dropEvent(e)

    def get_widget_data(self, item) -> dict:
        """
        Get widget data from item's widget
        :param item: item
        :return:
        """
        widget = self.itemWidget(item)
        item_data = {"task_id": widget.task_id, "input_path": widget.input_path, "row": self.row(item)}
        return item_data

    def get_items(self):
        """
        获取listwidget中条目数
        :return: list
        """
        widgetres = []
        count = self.count()
        # 遍历listwidget中的内容
        for i in range(count):
            widgetres.append(self.item(i))
        return widgetres

    def refreshTasks(self):
        items = self.get_items()
        self.clear()
        try:
            for item in items:
                widget = self.itemWidget(item)
                self.addFileItem(widget.input_path)
        except RuntimeError:
            pass

    def addFileItem(self, input_path: str, task_id=None):
        input_path = input_path.strip('"')
        taskListItem = MyListWidgetItem()
        if task_id is None:
            task_id = self.generateTaskId(input_path)
        taskListItem.setTask(input_path, task_id)
        taskListItem.dupSignal.connect(self.itemActionResponse)
        taskListItem.remSignal.connect(self.itemActionResponse)
        # Create QListWidgetItem
        taskListWidgetItem = QListWidgetItem(self)
        # Set size hint
        taskListWidgetItem.setSizeHint(taskListItem.sizeHint())
        # Add QListWidgetItem into QListWidget
        self.addItem(taskListWidgetItem)
        self.setItemWidget(taskListWidgetItem, taskListItem)

    def itemActionResponse(self, e: dict):
        """
        Respond to item's action(click on buttons)
        :param e:
        :return:
        """
        """
        self.dupSignal.emit({"task_id": self.task_id, "input_path": self.input_path, "action": 1})
        """
        task_id = e.get('task_id')
        target_item = None
        for item in self.get_items():
            if self.itemWidget(item).task_id == task_id:
                target_item = item
                break
        if target_item is None:
            return
        if e.get("action") == 1:  # dupSignal
            input_path = self.itemWidget(target_item).input_path
            self.addFileItem(input_path)
            pass
        elif e.get("action") == 0:
            self.takeItem(self.row(item))

    def keyPressEvent(self, e):
        current_item = self.currentItem()
        if current_item is None:
            e.ignore()
            return
        # if e.key() == Qt.Key_Delete:
        #     self.removeItemWidget(current_item)


class MyTextWidget(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dropEvent(self, event):
        try:
            if event.mimeData().hasUrls:
                url = event.mimeData().urls()[0]
                self.setText(f"{url.toLocalFile()}")
            else:
                event.ignore()
        except Exception as e:
            print(e)


class MyComboBox(QComboBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()


class MySpinBox(QSpinBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()


class MyDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()
