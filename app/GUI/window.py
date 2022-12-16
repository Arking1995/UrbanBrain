import os
import sys
from os.path import join
import math

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QCheckBox, QAction, QFileDialog, QLabel, QPushButton, \
    QSlider, QGridLayout, QGroupBox, QListWidget
from PyQt5.QtGui import QIcon, QPixmap, QImage

import numpy as np
import matplotlib.pyplot as plt
from Utils.utils import *
from UrbanCtrl import UrbanCtrl


class Urban_Ctrl_UI(QMainWindow):
    def __init__(self):
        super(Urban_Ctrl_UI, self).__init__()
        self.initialize_params()
        self.init_ui()

    def init_ui(self):
        self.initialize_widgets()
        self.intialize_layouts()

        # DBG
        test_img = read_img('Imgs/NY.jpg')
        self.layout_img = UrbanCtrl.get_layout_img()

        update_img_widget(self.latent_widget, test_img)
        update_img_widget(self.layout_widget, self.layout_img)

        self.show()

    def initialize_params(self):
        self.title = 'Urban Planner'
        self.left, self.top = 400, 400
        self.width, self.height = 1720, 1080

        params = {
            "pretrained_dir": 'UrbanCtrl/pretrained_model',
            "results_dir": 'UrbanCtrl/results',
        }
        UrbanCtrl.Initialize(params)

    def initialize_widgets(self):
        self.setWindowTitle(self.title)
        self.setAcceptDrops(True)

        self.center_widget = QWidget(self)
        self.setCentralWidget(self.center_widget)

        """ Canvas Widgets
        """
        self.latent_widget = QLabel(self)
        self.layout_widget = QLabel(self)



    def intialize_layouts(self):
        """ Canvas group
        """
        canvas_layout = QtWidgets.QHBoxLayout()
        canvas_layout.addWidget(self.latent_widget)
        canvas_layout.addWidget(self.layout_widget)

        self.canvas_group = QGroupBox("canvas", self)
        self.canvas_group.setLayout(canvas_layout)

        """ Root Layout
        """
        grid = QGridLayout()
        grid.addWidget(self.canvas_group, 0, 0)
        self.center_widget.setLayout(grid)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            quit()

        if event.key() == Qt.Key_D:
            print('Pressed D')


if __name__ == '__main__':
    scale = True

    if scale:
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        app = QApplication(sys.argv)
        app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    else:
        app = QApplication(sys.argv)

    gui = Urban_Ctrl_UI()
    sys.exit(app.exec_())
