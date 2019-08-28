"""
This file implements a visualizer for the laser video data.
"""

from PyQt5 import QtCore, QtGui

import pyqtgraph as pg
import pyqtgraph.ptime as ptime

import play_video_ui
import play_double_video_ui

import numpy as np
import matplotlib.cm as cm

import observations_set
import sys

def _add_labels_status_bar(statusbar):
    status_left = QtGui.QLabel()
    status_center_left = QtGui.QLabel()
    status_center_left.setAlignment(QtCore.Qt.AlignCenter)
    status_center_right = QtGui.QLabel()
    status_center_right.setAlignment(QtCore.Qt.AlignCenter)
    status_right = QtGui.QLabel()
    status_right.setAlignment(QtCore.Qt.AlignRight)

    statusbar.insertPermanentWidget(0, status_left, stretch=1)
    statusbar.insertPermanentWidget(1, status_center_left, stretch=1)
    statusbar.insertPermanentWidget(2, status_center_right, stretch=1)
    statusbar.insertPermanentWidget(3, status_right, stretch=1)

    return status_left, status_center_left, status_center_right, status_right

class VideoViewer(QtGui.QMainWindow):
    """
    Implements the visualization of a laser data video. It implements several characteristics:

    - Playback control: pause, resume, faster/slower speed, play forward/backwards, fast-forward 10 and 500 frames,
                        manual frame advance.
    - Color control: change and include new colormaps. Change the "levels" (min and max values in the data).
    """
    def __init__(self, video_data, cmap='inferno'):
        """
        Inits the visualization window to show the video `video_data`.
        :param video_data: A 3-D or 4-D video with shape (n_frames, rows, columns) or
        (n_frames, rows, columns, rgb or rgba).
        :param cmap: Default colormap. All the colormaps in matplotlib are available.
        """
        super(VideoViewer, self).__init__()
        self.ui = play_video_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.status_left, self.status_center_left, \
        self.status_center_right, self.status_right = _add_labels_status_bar(self.ui.statusbar)
        self.ui.graphicsView.setBackground([255, 255, 255, 255])

        self.video_data = video_data

        # The GraphicsView rotates the video, so it must be rotated with this line
        self.rotated_video = video_data.swapaxes(1,2)

        self.ui.graphicsView.enableMouse(True)
        self.ui.graphicsView.autoPixelRange = True
        self.ui.graphicsView.sigSceneMouseMoved.connect(self.mouseMoved)

        self.view_box = pg.ViewBox()
        self.ui.graphicsView.setCentralItem(self.view_box)
        self.view_box.setAspectLocked()

        self.imitem = pg.ImageItem(self.rotated_video[0])

        self.colormaps = ['inferno', 'viridis', 'plasma', 'jet']
        if cmap in self.colormaps:
            self.idx_colormap = self.colormaps.index(cmap)
        else:
            self.colormaps.append(cmap)
            self.idx_colormap = len(self.colormaps) - 1
        self.set_cmap(cmap)

        self.view_box.addItem(self.imitem)

        self.view_box.setRange(QtCore.QRectF(0,0,self.rotated_video.shape[1],self.rotated_video.shape[2]))

        self.idx = 0

        self.currentPos = QtCore.QPointF(0.0, 0.0)

        self.levels = laser_levels(self.video_data)
        self.idx_levels = 0
        self.imitem.setLevels(self.levels[self.idx_levels])

        self.status_left.setText(''.join(['Frame ', str(self.idx), " / ", str(self.video_data.shape[0])]))
        self.status_center_right.setText(''.join(['Pixel range: [', str(self.levels[self.idx_levels, 0]), ", ",
                                                  str(self.levels[self.idx_levels, 1]), ']']))


        self.forward = True
        self.running = True

        self.sleepFrames = [2000, 1000, 500, 300, 100, 70, 50, 30, 0.00001]
        self.stepIndex = 5

        self.lastTime = ptime.time()
        self.fps = None

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0.000001)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.running = not self.running
            if self.running:
                self.timer.start(self.sleepFrames[self.stepIndex])
            else:
                self.timer.stop()
        elif event.key() == QtCore.Qt.Key_Left:
            self.forward = False
        elif event.key() == QtCore.Qt.Key_Right:
            self.forward = True
        elif event.key() == QtCore.Qt.Key_Up:
            self.stepIndex = min(self.stepIndex + 1, len(self.sleepFrames) - 1)
            self.timer.start(self.sleepFrames[self.stepIndex])
        elif event.key() == QtCore.Qt.Key_Down:
            self.stepIndex = max(self.stepIndex - 1, 0)
            self.timer.start(self.sleepFrames[self.stepIndex])
        elif event.key() == QtCore.Qt.Key_R:
            self.view_box.setRange(QtCore.QRectF(0, 0, self.rotated_video.shape[1], self.rotated_video.shape[2]))
        elif event.key() == QtCore.Qt.Key_E:
            self.update()
        elif event.key() == QtCore.Qt.Key_Q:
            self.idx = self.idx - 500
            if self.idx < 0:
                self.idx = self.video_data.shape[0] + self.idx
            self.update_frame()
        elif event.key() == QtCore.Qt.Key_W:
            self.idx = self.idx + 500
            if self.idx >= self.video_data.shape[0]:
                self.idx = self.idx - self.video_data.shape[0]
            self.update_frame()
        elif event.key() == QtCore.Qt.Key_N:
            self.idx_levels = (self.idx_levels + 1) % self.levels.shape[0]
            self.status_center_right.setText(''.join(['Pixel range: [', str(self.levels[self.idx_levels, 0]), ", ",
                                                      str(self.levels[self.idx_levels, 1]), ']']))
            self.update_frame()
        elif event.key() == QtCore.Qt.Key_C:
            self.idx_colormap = (self.idx_colormap + 1) % len(self.colormaps)
            self.set_cmap(self.colormaps[self.idx_colormap])
        elif event.key() == QtCore.Qt.Key_A:
            self.idx = self.idx - 10
            if self.idx < 0:
                self.idx = self.video_data.shape[0] + self.idx
            self.update_frame()
        elif event.key() == QtCore.Qt.Key_S:
            self.idx = self.idx + 10
            if self.idx >= self.video_data.shape[0]:
                self.idx = self.idx - self.video_data.shape[0]
            self.update_frame()
        elif event.key() == QtCore.Qt.Key_T:
            self.timer.stop()
            self.idx = 0
            self.update_frame()

    def update(self):
        if self.forward:
            self.idx = self.idx + 1
        else:
            self.idx = self.idx - 1

        if self.idx == self.video_data.shape[0]:
            self.idx = 0
        elif self.idx == -1:
            self.idx = self.video_data.shape[0] - 1

        now = ptime.time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0/dt
        else:
            s = np.clip(dt*3., 0, 1)
            self.fps = self.fps * (1-s) + (1.0/dt) * s

        self.update_frame()

    def update_status_bar(self):
        self.status_left.setText(''.join(['Frame ', str(self.idx), " / ", str(self.video_data.shape[0])]))
        self.status_center_left.setText('(%.02f, %0.2f) [%.02f]' % (self.currentPos.y(), self.currentPos.x(),
                                                                 self.video_data[
                                                                     self.idx, int(self.currentPos.y()), int(
                                                                         self.currentPos.x())]))
        self.status_right.setText('FPS: %.02f' % self.fps)

    def update_frame(self):
        self.imitem.setImage(self.rotated_video[self.idx], levels=self.levels[self.idx_levels])
        self.update_status_bar()

    def mouseMoved(self, evt):
        if self.imitem.sceneBoundingRect().contains(evt):
            self.currentPos = self.view_box.mapSceneToView(evt)
            self.status_center_left.setText('(%.02f, %0.2f) [%.02f]' % (self.currentPos.y(), self.currentPos.x(),
                                                                     self.video_data[
                                                                         self.idx, int(self.currentPos.y()), int(
                                                                             self.currentPos.x())]))


    # From here https://github.com/pyqtgraph/pyqtgraph/issues/561#issuecomment-329904839
    def set_cmap(self, cmap):
        colormap = cm.get_cmap(cmap)
        colormap._init()
        self.imitem.setLookupTable(colormap._lut[:-3] * 255)


class DoubleVideoViewer(QtGui.QMainWindow):
    """
    As VideoViewer, but showing two different videos. One in the left and the other in the right. Both videos must
    have the same number of frames.

    Same functionality and controls as VideoViewer but it allows different "levels" for each video.
    """
    def __init__(self, left_video, right_video, left_levels=None, right_levels=None, cmap='inferno'):
        """
        Shows two videos with its respective levels.

        See the functions laser_levels(), laser_levels_common(), laser_levels_only_common() for help to compute the
        levels.

        :param left_video: A 3-D or 4-D video with shape (n_frames, rows, columns) or
        (n_frames, rows, columns, rgb or rgba).
        :param right_video: A 3-D or 4-D video with shape (n_frames, rows, columns) or
        (n_frames, rows, columns, rgb or rgba).
        :param left_levels: Levels of the left video. If None, the levels are computed automatically with the limits
            in the data.
        :param right_levels: Levels of the right video. If None, the levels are computed automatically with the limits
            in the data.
        :param cmap: Default colormap. All the colormaps in matplotlib are available.
        """
        super(DoubleVideoViewer, self).__init__()

        if left_video.shape[0] != right_video.shape[0]:
            raise ValueError("Left and right video have different number of frames.")

        self.left_grayscale = self.check_grayscale(left_video)
        self.right_grayscale = self.check_grayscale(right_video)

        self.left_video = left_video
        self.right_video = right_video

        # The GraphicsView rotates the video, so it must be rotated with this line
        self.left_rotated = self.left_video.swapaxes(1,2)
        self.right_rotated = self.right_video.swapaxes(1,2)

        if self.left_grayscale and self.right_grayscale:
            self.check_levels_common(left_levels, right_levels)
        elif self.left_grayscale:
            self.left_levels = self.check_levels_single(left_levels, left_video)
        elif self.right_grayscale:
            self.right_levels = self.check_levels_single(right_levels, right_video)

        self.ui = play_double_video_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.status_left, self.status_center_left, \
        self.status_center_right, self.status_right = _add_labels_status_bar(self.ui.statusbar)
        self.ui.graphicsView_left.setBackground([255, 255, 255, 255])
        self.ui.graphicsView_right.setBackground([255, 255, 255, 255])

        self.ui.graphicsView_left.enableMouse(True)
        self.ui.graphicsView_right.enableMouse(True)
        self.ui.graphicsView_left.autoPixelRange = True
        self.ui.graphicsView_right.autoPixelRange = True
        self.ui.graphicsView_left.sigSceneMouseMoved.connect(self.mouseMoved_left)
        self.ui.graphicsView_right.sigSceneMouseMoved.connect(self.mouseMoved_right)

        self.left_view_box = pg.ViewBox()
        self.right_view_box = pg.ViewBox()
        self.ui.graphicsView_left.setCentralItem(self.left_view_box)
        self.ui.graphicsView_right.setCentralItem(self.right_view_box)
        self.left_view_box.setAspectLocked()
        self.right_view_box.setAspectLocked()

        self.left_imitem = pg.ImageItem(self.left_rotated[0])
        self.right_imitem = pg.ImageItem(self.right_rotated[0])

        self.colormaps = ['inferno', 'viridis', 'plasma', 'jet']
        if cmap in self.colormaps:
            self.idx_colormap = self.colormaps.index(cmap)
        else:
            self.colormaps.append(cmap)
            self.idx_colormap = len(self.colormaps) - 1

        self.set_cmap(cmap)
        if self.left_grayscale:
            self.idx_left_level = 0
            self.left_imitem.setLevels(self.left_levels[self.idx_left_level])
        if self.right_grayscale:
            self.idx_right_level = 0
            self.right_imitem.setLevels(self.right_levels[self.idx_right_level])

        self.left_view_box.addItem(self.left_imitem)
        self.right_view_box.addItem(self.right_imitem)

        self.left_view_box.setRange(QtCore.QRectF(0, 0, self.left_rotated.shape[1], self.left_rotated.shape[2]))
        self.right_view_box.setRange(QtCore.QRectF(0, 0, self.right_rotated.shape[1], self.right_rotated.shape[2]))

        self.idx = 0
        self.currentPos = QtCore.QPointF(0.0, 0.0)
        self.leftPos = True

        self.status_left.setText(''.join(['Frame ', str(self.idx), " / ", str(self.left_video.shape[0])]))
        self.status_center_right.setText(''.join([
            ''.join(['Left pixel range: [', str(self.left_levels[self.idx_left_level, 0]), ", ",
                     str(self.left_levels[self.idx_left_level, 1]), '].  ']) if self.left_grayscale else '',
            ''.join(['Right pixel range: [', str(self.right_levels[self.idx_right_level, 0]), ", ",
                                                  str(self.right_levels[self.idx_right_level, 1]), '].']) if self.right_grayscale else '']))


        self.forward = True
        self.running = True

        self.sleepFrames = [2000, 1000, 500, 300, 100, 70, 50, 30, 0.00001]
        self.stepIndex = 5

        self.lastTime = ptime.time()
        self.fps = None

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0.000001)

    def check_grayscale(self, video):
        if video.ndim == 3:
            return True
        elif video.ndim == 4 and (video.shape[3] == 3 or video.shape[3] == 4):
            return False
        else:
            raise ValueError("The video should be a 3D array (grayscale) or a 4D array (color) with RGB or RGB values in"
                             "the last axis.")

    def check_levels_single(self, levels, video):
        if levels is None:
            return laser_levels(video)
        else:
            return np.atleast_2d(levels)

    def check_levels_common(self, left_levels, right_levels):
        if left_levels is None and right_levels is None:
            self.left_levels, self.right_levels = laser_levels_common(self.left_video, self.right_video)
        elif not left_levels is None and right_levels is None:
            left_levels = np.atleast_2d(left_levels)
            if left_levels.shape[1] != 2:
                raise ValueError("left_levels has wrong dimension in axis 1: %d. Expected dimension: 2" % left_levels.shape[1])
            self.left_levels = left_levels
            self.right_levels = laser_levels(self.right_video)
        elif left_levels is None and not right_levels is None:
            right_levels = np.atleast_2d(right_levels)
            if right_levels.shape[1] != 2:
                raise ValueError(
                    "right_levels has wrong dimension in axis 1: %d. Expected dimension: 2" % right_levels.shape[1])
            self.right_levels = right_levels
            self.left_levels = laser_levels(self.left_video)
        else:
            left_levels = np.atleast_2d(left_levels)
            right_levels = np.atleast_2d(right_levels)
            if left_levels.shape[1] != right_levels.shape[1] != 2:
                raise ValueError("Wrong dimension in axis 1. left_levels: %d, right_levels: %d. Expected dimension: "
                                 % (left_levels.shape[1], right_levels.shape[1]))
            self.left_levels = left_levels
            self.right_levels = right_levels

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.running = not self.running
            if self.running:
                self.timer.start(self.sleepFrames[self.stepIndex])
            else:
                self.timer.stop()
        elif event.key() == QtCore.Qt.Key_Left:
            self.forward = False
        elif event.key() == QtCore.Qt.Key_Right:
            self.forward = True
        elif event.key() == QtCore.Qt.Key_Up:
            self.stepIndex = min(self.stepIndex + 1, len(self.sleepFrames) - 1)
            self.timer.start(self.sleepFrames[self.stepIndex])
        elif event.key() == QtCore.Qt.Key_Down:
            self.stepIndex = max(self.stepIndex - 1, 0)
            self.timer.start(self.sleepFrames[self.stepIndex])
        elif event.key() == QtCore.Qt.Key_R:
            self.left_view_box.setRange(QtCore.QRectF(0, 0, self.left_rotated.shape[1], self.left_rotated.shape[2]))
            self.right_view_box.setRange(QtCore.QRectF(0, 0, self.right_rotated.shape[1], self.right_rotated.shape[2]))
        elif event.key() == QtCore.Qt.Key_E:
            self.update()
        elif event.key() == QtCore.Qt.Key_Q:
            self.idx = self.idx - 500
            if self.idx < 0:
                self.idx = self.left_video.shape[0] + self.idx
            self.update_frame()
        elif event.key() == QtCore.Qt.Key_W:
            self.idx = self.idx + 500
            if self.idx >= self.left_video.shape[0]:
                self.idx = self.idx - self.left_video.shape[0]
            self.update_frame()
        elif event.key() == QtCore.Qt.Key_N and self.left_grayscale:
            self.idx_left_level = (self.idx_left_level + 1) % self.left_levels.shape[0]
            self.left_imitem.setLevels(self.left_levels[self.idx_left_level])
            self.status_center_right.setText(''.join([
                ''.join(['Left pixel range: [', str(self.left_levels[self.idx_left_level, 0]), ", ",
                         str(self.left_levels[self.idx_left_level, 1]), '].  ']) if self.left_grayscale else '',
                ''.join(['Right pixel range: [', str(self.right_levels[self.idx_right_level, 0]), ", ",
                         str(self.right_levels[self.idx_right_level, 1]), '].']) if self.right_grayscale else '']))

            self.update_frame()
        elif event.key() == QtCore.Qt.Key_M and self.right_grayscale:
            self.idx_right_level = (self.idx_right_level + 1) % self.right_levels.shape[0]
            self.right_imitem.setLevels(self.right_levels[self.idx_right_level])
            self.status_center_right.setText(''.join([
                ''.join(['Left pixel range: [', str(self.left_levels[self.idx_left_level, 0]), ", ",
                         str(self.left_levels[self.idx_left_level, 1]), '].  ']) if self.left_grayscale else '',
                ''.join(['Right pixel range: [', str(self.right_levels[self.idx_right_level, 0]), ", ",
                         str(self.right_levels[self.idx_right_level, 1]), '].']) if self.right_grayscale else '']))

            self.update_frame()
        elif event.key() == QtCore.Qt.Key_C:
            self.idx_colormap = (self.idx_colormap + 1) % len(self.colormaps)
            self.set_cmap(self.colormaps[self.idx_colormap])
        elif event.key() == QtCore.Qt.Key_A:
            self.idx = self.idx - 10
            if self.idx < 0:
                self.idx = self.left_video.shape[0] + self.idx
            self.update_frame()
        elif event.key() == QtCore.Qt.Key_S:
            self.idx = self.idx + 10
            if self.idx >= self.left_video.shape[0]:
                self.idx = self.idx - self.left_video.shape[0]
            self.update_frame()
        elif event.key() == QtCore.Qt.Key_T:
            self.timer.stop()
            self.idx = 0
            self.update_frame()

    def update(self):
        if self.forward:
            self.idx = self.idx + 1
        else:
            self.idx = self.idx - 1

        if self.idx == self.left_video.shape[0]:
            self.idx = 0
        elif self.idx == -1:
            self.idx = self.left_video.shape[0] - 1

        now = ptime.time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0/dt
        else:
            s = np.clip(dt*3., 0, 1)
            self.fps = self.fps * (1-s) + (1.0/dt) * s

        self.update_frame()

    def update_status_bar(self):
        self.status_left.setText(''.join(['Frame ', str(self.idx), " / ", str(self.left_video.shape[0])]))

        if self.leftPos:
            pixelvalue = self.left_video[self.idx, int(self.currentPos.y()), int(self.currentPos.x())]
            single_value = self.left_grayscale
        else:
            pixelvalue = self.right_video[self.idx, int(self.currentPos.y()), int(self.currentPos.x())]
            single_value = self.right_grayscale

        if single_value:
            self.status_center_left.setText('(%.02f, %0.2f) [%.02f]' % (self.currentPos.y(), self.currentPos.x(), pixelvalue))
        else:
            self.status_center_left.setText(
                '(%.02f, %0.2f) [%.02f, %.02f, %.02f]' % (self.currentPos.y(), self.currentPos.x(),
                                                          pixelvalue[0], pixelvalue[1], pixelvalue[2]))
        self.status_right.setText('FPS: %.02f' % self.fps)

    def update_frame(self):
        self.left_imitem.setImage(self.left_rotated[self.idx], levels=self.left_levels[self.idx_left_level])
        self.right_imitem.setImage(self.right_rotated[self.idx], levels=self.right_levels[self.idx_right_level])
        self.update_status_bar()

    def mouseMoved_left(self, evt):
        if self.left_imitem.sceneBoundingRect().contains(evt):
            self.currentPos = self.left_view_box.mapSceneToView(evt)
            self.leftPos = True
            pixelvalue = self.left_video[self.idx, int(self.currentPos.y()), int(self.currentPos.x())]
            if self.left_grayscale:
                self.status_center_left.setText(
                    '(%.02f, %0.2f) [%.02f]' % (self.currentPos.y(), self.currentPos.x(), pixelvalue))
            else:
                self.status_center_left.setText(
                    '(%.02f, %0.2f) [%.02f, %.02f, %.02f]' % (self.currentPos.y(), self.currentPos.x(),
                                                              pixelvalue[0], pixelvalue[1], pixelvalue[2]))
    def mouseMoved_right(self, evt):
        if self.right_imitem.sceneBoundingRect().contains(evt):
            self.currentPos = self.right_view_box.mapSceneToView(evt)
            self.leftPos = False
            pixelvalue = self.right_video[self.idx, int(self.currentPos.y()), int(self.currentPos.x())]
            if self.right_grayscale:
                self.status_center_left.setText(
                    '(%.02f, %0.2f) [%.02f]' % (self.currentPos.y(), self.currentPos.x(), pixelvalue))
            else:
                self.status_center_left.setText(
                    '(%.02f, %0.2f) [%.02f, %.02f, %.02f]' % (self.currentPos.y(), self.currentPos.x(),
                                                              pixelvalue[0], pixelvalue[1], pixelvalue[2]))

    # From here https://github.com/pyqtgraph/pyqtgraph/issues/561#issuecomment-329904839
    def set_cmap(self, cmap):
        colormap = cm.get_cmap(cmap)
        colormap._init()
        lut = colormap._lut[:-3] * 255
        if self.left_grayscale:
            self.left_imitem.setLookupTable(lut)
        if self.right_grayscale:
            self.right_imitem.setLookupTable(lut)

def laser_levels(video):
    """
    Computes the levels for a single laser video.
    :param video: Video to compute the levels
    :return: A unique set of levels with limits.
    """
    return np.unique(np.asarray([[0, 1023], [0, video.max()], [video.min(), video.max()]]), axis=0)

def laser_levels_only_common(video1, video2):
    """
    Computes the levels for two laser videos, including common min/max value for both videos.

    :param video1: A video to compute the levels.
    :param video2: Other video to compute the levels.
    :return: Left unique video levels, Right unique video levels.
    """
    min_value = min(video1.min(), video2.min())
    max_value = max(video1.max(), video2.max())
    return np.unique(np.asarray([[0, 1023], [0, max_value], [min_value, max_value]]), axis=0)

def laser_levels_common(video1, video2):
    """
    Computes the levels for two laser videos, including common min/max value for both videos and the local min/max
    values for each video.

    :param video1: A video to compute the levels.
    :param video2: Other video to compute the levels.
    :return: Left unique video levels, Right unique video levels.
    """
    min_value = min(video1.min(), video2.min())
    max_value = max(video1.max(), video2.max())

    left_levels = np.asarray([[0, 1023], [0, max_value], [min_value, max_value], [video1.min(), video1.max()], [0, video1.max()]])
    right_levels = np.asarray([[0, 1023], [0, max_value], [min_value, max_value], [video2.min(), video2.max()], [0, video2.max()]])

    return np.unique(left_levels, axis=0), np.unique(right_levels, axis=0)


if QtGui.QApplication.instance() is None:
    g_app = QtGui.QApplication(sys.argv)

def play_video(video, cmap='inferno'):
    """
    Plays a given video with a default colormap.

    :param video: A 3-D or 4-D video with shape (n_frames, rows, columns) or
    (n_frames, rows, columns, rgb or rgba).
    :param cmap: Default colormap. All the colormaps in matplotlib are available.
    :return:
    """
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtGui.QApplication.instance()
        myapp = VideoViewer(video, cmap)
        myapp.show()
        app.exec_()

def play_double_video(video1, video2, levels1=None, levels2=None, cmap='inferno'):
    """
    Plays two videos with the selected levels and a default colormap.

    :param video1: A 3-D or 4-D video with shape (n_frames, rows, columns) or
    (n_frames, rows, columns, rgb or rgba).
    :param video2: A 3-D or 4-D video with shape (n_frames, rows, columns) or
    (n_frames, rows, columns, rgb or rgba).
    :param levels1: Levels of the left video. If None, the levels are computed automatically with the limits
        in the data.
    :param levels2: Levels of the right video. If None, the levels are computed automatically with the limits
        in the data.
    :param cmap: Default colormap. All the colormaps in matplotlib are available.
    :return:
    """
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtGui.QApplication.instance()
        myapp = DoubleVideoViewer(video1, video2, levels1, levels2, cmap)
        myapp.show()
        app.exec_()

def load_video(filename):
    """
    Load the 3D array of a laser video.
    :param filename: Name of the file.
    :return:
    """
    return np.load(filename)['image']

if __name__ == '__main__':

    obs_set = observations_set.ObservationROISet.fromfolder('data/Type1')
    video = load_video('data/Type1/0001.npz')
    video2 = load_video('data/Type1/0002.npz')

    min_length = min(video.shape[0], video2.shape[0])
    left_levels, right_levels = laser_levels_common(video, video2)

    play_double_video(video[:min_length], video2[:min_length], left_levels, right_levels)

    play_video(video2)