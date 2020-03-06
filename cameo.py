import cv2

from tools.managers import CaptureManager, WindowManager
import tools.face_detection as face
import tools.filters as filters

cascade = {"face": "./haarcascade/haarcascade_frontalface_default.xml",
           "eye": "./haarcascade/haarcascade_eye.xml"}


class Cameo(object):
    def __init__(self):
        self._windowManger = WindowManager('Cameo', self.onKeypress)

        self._captureManger = CaptureManager(cv2.VideoCapture(0),
                                             self._windowManger, True)
        self._curveFilter = filters.EmbossFilter()
        self._faceDetect = face.Detect(faceCascaPath=cascade["face"],
                                       eyeCascaPath=cascade["eye"])

    def run(self):
        """运行捕获摄像头图像"""
        self._windowManger.createWindow()
        while self._windowManger.isWindowCreated:
            self._captureManger.enterFrame()
            frame = self._captureManger.frame

            # TODO 滤波
            # return返回的是引用（地址），这里滤波结果返回frame即类内改变_frame
            # filters.strokeEdges(frame, frame)  # 边缘检测
            # cv2.Canny(frame, 200, 300)
            # self._curveFilter.apply(frame, frame)  # 特效
            # filters.contourDetection(frame, style="contours")  # 轮廓检测

            # 人脸识别
            self._faceDetect.faceDetect(frame)

            self._captureManger.exitFrame()
            self._windowManger.processEvent()

    def onKeypress(self, keycode):
        """处理按键"""

        if keycode == 32:  # space
            self._captureManger.writeImage('./data/cache/screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManger.isWritingVideo:
                self._captureManger.startWritingVideo(
                    './data/cache/screencast.avi')
            else:
                self._captureManger.stopWritingVideo()
        elif keycode == 27:  # ESC
            self._windowManger.destroyWindow()


if __name__ == "__main__":
    cameo = Cameo()
    cameo.run()
