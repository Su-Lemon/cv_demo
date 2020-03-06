"""人脸检测"""

import cv2


class Detect(object):
    def __init__(self, **kwargs):
        if "faceCascaPath" in kwargs:
            self.faceCascade = cv2.CascadeClassifier(kwargs["faceCascaPath"])  # 人脸检测器
        if "eyeCascaPath" in kwargs:
            self.eyeCascade = cv2.CascadeClassifier(kwargs["eyeCascaPath"])  # 眼睛检测器

        self.faceAreas = None
        self.eyeAreas = None

    def faceDetect(self, src):
        try:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            self.faceAreas = self.faceCascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in self.faceAreas:
                cv2.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 2)

                roi = src[y:y+h, x:x+w]
                self.eyeDetect(roi)
        except Exception as e:
            print(e)

    def eyeDetect(self, src):
        try:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            self.eyeAreas = self.eyeCascade.detectMultiScale(gray, 1.03, 5, 0, (40, 40))

            for (x, y, w, h) in self.eyeAreas:
                cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
        except Exception as e:
            print(e)

