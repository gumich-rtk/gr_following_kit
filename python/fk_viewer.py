#!/usr/bin/python3
"""
Gumich Following Kit Demo Viewer
"""

import sys
import signal
import socket
import struct
import can
import numpy as np
from scipy.optimize import least_squares
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui, QtNetwork


ANCHORS_NUMBER = 4

CAN_IFACE = "can1"
CAN_BITRATE = 250000
CAN_ID_BASE = 0x400

VISIBLE_DIST_M = 5.0
ANCHOR_RECT_WIDTH_M = 0.64
ANCHOR_RECT_LENGTH_M = 0.65

DIST_FILTER_LEN = 4

ANCHOR_COLORS = [(255,   0,   0),
                 (255, 255,   0),
                 (0,   255,   0),
                 (128,   0, 255)]
ANCHOR_SINGLE_COLOR = (255, 255, 255)

DEVICE_RADIUS_PX = 5

def anchor_get_colors(anchor_number):
    """
    Generate color for a anchor.
    """
    if anchor_number <= 4:
        return ANCHOR_COLORS[:anchor_number]
    return [ANCHOR_SINGLE_COLOR] * anchor_number


def anchor_calculate_coord_simple_rect(width, length):
    """
    Generate coordinates in case of 4 anchors placed in corners of a rectangle.
    """
    return list(np.array([
        [+(length / 2), +(width / 2)],
        [+(length / 2), -(width / 2)],
        [-(length / 2), -(width / 2)],
        [-(length / 2), +(width / 2)],
    ]))


# pylint: disable=too-few-public-methods
class MeanFilter:
    """
    Simple averaging filter.
    """

    def __init__(self, size):
        self.buf = [0.0 for _ in range(size)]
        self.pos = 0

    def __call__(self, value):
        self.buf[self.pos] = value
        self.pos = (self.pos + 1) % len(self.buf)
        return sum(self.buf) / len(self.buf)


def mlat_solve(distances, coordinates, initial_guess):
    """
    Solve the multilateration task.
    """
    def residue(solution):
        return [(np.linalg.norm(solution - coordinates[i]) - distances[i])
                for i in range(len(coordinates))]
    return least_squares(residue, initial_guess).x


def index_from_can_id(can_id, id_base, n_anchors):
    """
    Get the index in array from the CAN ID.
    """
    index = can_id - id_base - 1
    if not 0 <= index < n_anchors:
        return None
    return index


class CanThread(QtCore.QThread):
    """
    Background thread that receives can messages.
    """
    data_ready = QtCore.pyqtSignal(int, float, int)

    def __init__(self, channel, bitrate, id_base, n_devices):
        super().__init__()
        self.channel = channel
        self.bitrate = bitrate
        self.id_base = id_base
        self.n_devices = n_devices
        self.extended = False
        self.stop = False

    def run(self):
        """
        Main loop.
        """
        def get_id_mask(n_devices):
            id_mask = 0xFFF
            while n_devices:
                id_mask = id_mask << 1
                n_devices = n_devices >> 1
            return id_mask
        id_mask = get_id_mask(self.n_devices)

        # pylint: disable=abstract-class-instantiated
        bus = can.interface.Bus(bustype='socketcan', channel=self.channel,
                                bitrate=self.bitrate)
        filters = [{"can_id": self.id_base, "can_mask": id_mask,
                    "extended": self.extended}]
        bus.set_filters(filters)

        while True:
            msg = bus.recv(0.2)
            if self.isInterruptionRequested():
                break
            if msg is None:
                continue
            index = index_from_can_id(msg.arbitration_id, self.id_base,
                                      ANCHORS_NUMBER)
            if index is None:
                continue
            dist_data = msg.data[0:4]
            buttons_data = msg.data[4:]
            dist_mm = struct.unpack('<i', dist_data)[0]
            dist_m = float(dist_mm) / 1000
            self.data_ready.emit(index, dist_m, buttons_data)


class MainWindow(QtWidgets.QMainWindow):
    """
    The Qt main window object.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if ANCHORS_NUMBER == 4:
            self.anchor_coord_m = anchor_calculate_coord_simple_rect(
                    ANCHOR_RECT_WIDTH_M, ANCHOR_RECT_LENGTH_M)
        else:
            raise NotImplementedError(
                    f"unsupported number of anchors: {ANCHORS_NUMBER}")
        self.anchor_colors = anchor_get_colors(ANCHORS_NUMBER)

        self.tag_dist_filters = [
                MeanFilter(DIST_FILTER_LEN) for _ in range(ANCHORS_NUMBER)]

        cm_m = np.sum(self.anchor_coord_m) / len(self.anchor_coord_m)
        self.tag_dist_m = [np.linalg.norm(cm_m - self.anchor_coord_m)
                           for i in range(ANCHORS_NUMBER)]
        self.tag_coord_m = [0.0, 0.0]
        self.button_status = 0

        central_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QHBoxLayout()
        central_layout.setSpacing(0)
        central_layout.setContentsMargins(0, 0, 0, 0)

        self.label = QtWidgets.QLabel()
        self.label.installEventFilter(self)
        self.painter = QtGui.QPainter()
        la_sp = self.label.sizePolicy()
        la_sp.setHorizontalStretch(1)
        la_sp.setHorizontalPolicy(QtWidgets.QSizePolicy.GrowFlag |
                                  QtWidgets.QSizePolicy.ShrinkFlag |
                                  QtWidgets.QSizePolicy.IgnoreFlag)
        la_sp.setVerticalPolicy(QtWidgets.QSizePolicy.GrowFlag |
                                QtWidgets.QSizePolicy.ShrinkFlag |
                                QtWidgets.QSizePolicy.IgnoreFlag)
        self.label.setSizePolicy(la_sp)
        central_layout.addWidget(self.label)

        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('k')
        self.graph_widget.hideAxis('bottom')
        self.graph_widget.showGrid(x=False, y=True, alpha=0.5)
        self.graph_widget.enableAutoRange(axis='y')
        self.graph_widget.setAutoVisible(y=True)

        gw_sp = self.graph_widget.sizePolicy()
        gw_sp.setHorizontalStretch(1)
        self.graph_widget.setSizePolicy(gw_sp)
        central_layout.addWidget(self.graph_widget)

        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        self.values_x = list(range(1000))
        self.values_y = [[0 for _ in range(1000)] for _ in range(ANCHORS_NUMBER)]

        pens = [pg.mkPen(color=self.anchor_colors[i], width=2)
                for i in range(ANCHORS_NUMBER)]
        self.plot = [self.graph_widget.plot(self.values_x, self.values_y[i],
                                            pen=pens[i])
                     for i in range(ANCHORS_NUMBER)]

        canvas = QtGui.QPixmap(self.label.width(), self.label.height())
        self.label.setPixmap(canvas)

    def draw(self):
        """
        Redraw the widget graphics.
        """
        canvas = self.label.pixmap()
        canvas.fill(QtGui.QColor(0, 0, 0))
        self.painter.begin(canvas)

        device_pen = QtGui.QPen()
        device_pen.setWidth(1)
        device_brush = QtGui.QBrush()
        device_brush.setStyle(QtCore.Qt.SolidPattern)
        line_pen = QtGui.QPen()
        line_pen.setWidth(1)
        line_brush = QtGui.QBrush()

        cw_px = canvas.width()
        ch_px = canvas.height()
        o_px = [cw_px // 2, ch_px // 2]
        d_px = min(o_px)
        px_per_m = float(d_px) / VISIBLE_DIST_M

        def m2px(scalar_m):
            return int(float(scalar_m) * px_per_m)

        def v_m2px(vector_m):
            return [o_px[0] - m2px(vector_m[1]), o_px[1] - m2px(vector_m[0])]

        for i in range(ANCHORS_NUMBER):
            a_px = v_m2px(self.anchor_coord_m[i])
            r_px = m2px(self.tag_dist_m[i])

            device_pen.setColor(QtGui.QColor(*self.anchor_colors[i]))
            device_brush.setColor(QtGui.QColor(*self.anchor_colors[i]))
            line_pen.setColor(QtGui.QColor(*self.anchor_colors[i]))
            self.painter.setPen(device_pen)
            self.painter.setBrush(device_brush)
            self.painter.drawEllipse(QtCore.QPoint(*a_px),
                                     DEVICE_RADIUS_PX, DEVICE_RADIUS_PX)
            self.painter.setPen(line_pen)
            self.painter.setBrush(line_brush)
            self.painter.drawEllipse(QtCore.QPoint(*a_px), r_px, r_px)

        line_pen.setColor(QtGui.QColor(255, 255, 255))
        self.painter.setPen(line_pen)
        self.painter.setBrush(line_brush)
        self.painter.drawLine(QtCore.QPoint(*o_px),
                              QtCore.QPoint(*v_m2px([0.50, 0.00])))
        self.painter.drawLine(QtCore.QPoint(*o_px),
                              QtCore.QPoint(*v_m2px([0.00, 0.25])))

        t_px = v_m2px(self.tag_coord_m)
        self.painter.drawLine(QtCore.QPoint(*o_px), QtCore.QPoint(*t_px))

        device_pen.setColor(QtGui.QColor(255, 255, 255))
        device_brush.setColor(QtGui.QColor(255, 255, 255))
        self.painter.setPen(device_pen)
        self.painter.setBrush(device_brush)
        self.painter.drawEllipse(QtCore.QPoint(*t_px), 5, 5)
        self.painter.end()

    # pylint: disable=invalid-name
    def eventFilter(self, obj, event):
        """
        Qt widget event callback.
        """
        if obj == self.label and event.type() == QtCore.QEvent.Resize:
            canvas = QtGui.QPixmap(self.label.width(), self.label.height())
            self.label.setPixmap(canvas)
        if obj == self.label and event.type() == QtCore.QEvent.Paint:
            self.draw()
        return super().eventFilter(obj, event)

    def calculate_coord(self):
        """
        Calculate the coordinates of the tag.
        """
        self.tag_coord_m = mlat_solve(
                self.tag_dist_m, self.anchor_coord_m, self.tag_coord_m)

    def process_buttons(self):
        """
        Handle button events.
        """

    @QtCore.pyqtSlot(int, float, int)
    def update_data(self, index, distance_m, button_status):
        """
        Receive the data from the CAN thread.
        """
        self.values_y[index] = self.values_y[index][1:]
        self.values_y[index].append(distance_m)
        self.plot[index].setData(self.values_x, self.values_y[index])

        self.button_status = button_status
        self.tag_dist_m[index] = self.tag_dist_filters[index](distance_m)
        self.calculate_coord()
        self.process_buttons()
        self.label.update()


class SignalWakeupHandler(QtNetwork.QAbstractSocket):
    """
    A helper object that utilizes the set_wakeup_fd function to make the Qt
    application responsive to command line keyboard shortcuts (Ctrl+C).
    """
    def __init__(self, parent=None):
        super().__init__(QtNetwork.QAbstractSocket.UdpSocket, parent)
        self.fd_prev = None
        self.c_sock, self.py_sock = socket.socketpair(type=socket.SOCK_DGRAM)
        self.setSocketDescriptor(self.py_sock.fileno())
        self.c_sock.setblocking(False)
        self.fd_prev = signal.set_wakeup_fd(self.c_sock.fileno())
        self.readyRead.connect(self.read_signal)

    def __del__(self):
        if self.fd_prev is not None:
            signal.set_wakeup_fd(self.fd_prev)

    def read_signal(self):
        """
        Read a signal number.
        """
        self.readData(1)


app = QtWidgets.QApplication(sys.argv)
SignalWakeupHandler(app)
signal.signal(signal.SIGINT, lambda sig, _: app.quit())
main_window = MainWindow()
device_thread = CanThread(CAN_IFACE, CAN_BITRATE, CAN_ID_BASE, ANCHORS_NUMBER)
device_thread.data_ready.connect(main_window.update_data)
app.aboutToQuit.connect(device_thread.requestInterruption)
device_thread.start()
main_window.show()
sys.exit(app.exec_())
