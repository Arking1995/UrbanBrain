import os
import sys
from os.path import join

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QAction, QFileDialog, QLabel, QPushButton, QSlider, \
    QGridLayout, QGroupBox, QListWidget
from PyQt5.QtGui import QIcon, QPixmap, QImage

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


def resize(img, max_size):
    old_shape = len(img.shape)
    h, w = img.shape[:2]
    if h > w:
        newh, neww = max_size, int(max_size * w / h)
    else:
        newh, neww = int(max_size * h / w), max_size
    ret = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
    if old_shape != len(ret.shape):
        return ret[..., np.newaxis]
    return ret


def set_qt_img(img, label):
    pixmap = QPixmap(img)
    label.setPixmap(pixmap)
    label.adjustSize()


def to_qt_img(np_img):
    if np_img.dtype != np.uint8:
        np_img = np.clip(np_img, 0.0, 1.0)
        np_img = np_img * 255.0
        np_img = np_img.astype(np.uint8)

    if len(np_img.shape) == 2:
        np_img = np_img[..., np.newaxis].repeat(3, axis=2)

    h, w, c = np_img.shape
    # bytesPerLine = 3 * w
    return QImage(np_img.data, w, h, 3 * w, QImage.Format_RGB888)


def update_img_widget(widget, img):
    set_qt_img(to_qt_img(img), widget)


def overlap_replace(a, b, start_pos):
    """ overlapping a and b
    """
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    a_ = a.copy()

    sh, shh, sw, sww = start_pos[0], start_pos[0] + hb, start_pos[1], start_pos[1] + wb
    clipped_h, clipped_hh, clipped_w, clipped_ww = np.clip(sh, 0, ha), np.clip(shh, 0, ha), np.clip(sw, 0, wa), np.clip(
        sww, 0, wa)
    h, w = clipped_hh - clipped_h, clipped_ww - clipped_w
    a_[clipped_h:clipped_hh, clipped_w:clipped_ww] = b[clipped_h - sh:clipped_h - sh + h,
                                                     clipped_w - sw:clipped_w - sw + w]
    return a_


def composite(a, b, bmask):
    return (1.0 - bmask) * a + bmask * b


def overlap_comp(a, b, bmask, start_pos):
    """ overlap and composite a and b with bmask
    """
    acopy = a.copy()
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]

    sh, shh, sw, sww = start_pos[0], start_pos[0] + hb, start_pos[1], start_pos[1] + wb
    clipped_h, clipped_hh, clipped_w, clipped_ww = np.clip(sh, 0, ha), np.clip(shh, 0, ha), np.clip(sw, 0, wa), np.clip(
        sww, 0, wa)
    h, w = clipped_hh - clipped_h, clipped_ww - clipped_w
    acopy[clipped_h:clipped_hh, clipped_w:clipped_ww] = composite(a[clipped_h:clipped_hh, clipped_w:clipped_ww],
                                                                  b[clipped_h - sh:clipped_h - sh + h,
                                                                  clipped_w - sw:clipped_w - sw + w],
                                                                  bmask[clipped_h - sh:clipped_h - sh + h,
                                                                  clipped_w - sw:clipped_w - sw + w])
    return acopy


def read_img(fname, fmt='RGB'):
    return np.array(Image.open(fname).convert(fmt)) / 255.0


def save_img(fname, img):
    plt.imsave(fname, np.clip(img, 0.0, 1.0))
    print('{} file saved'.format(fname))


def draw_line(img, p0, p1, color='red'):
    pil_img = Image.fromarray((img * 255.0).astype(np.uint8))
    img_draw = ImageDraw.Draw(pil_img)
    img_draw.line((p0, p1), fill=color, width=2)
    return np.array(pil_img) / 255.0


def draw_point(img, p, size=5, color='red'):
    pil_img = Image.fromarray((img * 255.0).astype(np.uint8))
    img_draw = ImageDraw.Draw(pil_img)
    img_draw.ellipse((p[0], p[1], p[0] + size, p[1] + size), fill=color)
    return np.array(pil_img) / 255.0
