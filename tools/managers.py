"""管理器"""

import cv2
import numpy as np
import time


class CaptureManager(object):
    """视频捕获管理"""

    def __init__(self, capture, previewWindowManger=None,
                 shouldMirrorPreview=False):
        self.previewWindowManger = previewWindowManger  # 预览窗口
        self.shouldMirrorPreview = shouldMirrorPreview  # 是否镜像查看

        self._capture = capture  # 视频读写类
        self._channel = 0
        self._enteredFrame = False  # 是否成功从设备捕获下一帧
        self._frame = None  # 缓存捕获到的一帧
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None  # 捕获开始的时间
        self._frameElapsed = 0  # 从开始捕获到现在已经过去的帧
        self._fpsEstimate = None  # 帧率估计

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """如果有的话捕获下一帧"""

        # 检查是否已捕获完所有帧
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'
        if not self._enteredFrame:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """在窗口绘制，写入文件，释放帧"""

        # 检查可抓取的帧是否已读入
        # 读取并缓存帧
        if self.frame is None:
            self._enteredFrame = False
            return

        # 更新帧率估计和相关变量
        if self._frameElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._frameElapsed / timeElapsed
        self._frameElapsed += 1

        # 如果有绘到窗口
        if self.previewWindowManger is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManger.show(mirroredFrame)
            else:
                self.previewWindowManger.show(self._frame)
        # 写入图片文件
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # 写入视频文件
        self._writeVideoFrame()

        # 释放帧
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """把下一个退出的帧写入图片文件"""
        self._imageFilename = filename

    def startWritingVideo(self, filename,
                          encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        """开始将一个退出帧些人视频文件"""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """停止将一个退出帧些人视频文件"""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        """创建或向视频文件追加内容"""
        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # 捕获的帧率不确定，使用估计
                if self._frameElapsed < 20:
                    # 等待更多的帧过去，以便估算更稳定
                    return
                else:
                    fps = self._fpsEstimate

            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename,
                                                self._videoEncoding, fps, size)
        self._videoWriter.write(self._frame)


class WindowManager(object):
    """窗口管理"""

    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback  # 键盘检测回调函数

        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvent(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # 丢弃GTK编码的所有非ASCII信息
            keycode &= 0xFF
            self.keypressCallback(keycode)


