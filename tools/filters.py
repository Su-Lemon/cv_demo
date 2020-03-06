"""滤波器"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
# from . import utils


def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    """
    边缘检测
    :param src:源图像
    :param dst:目标图像
    :param blurKsize:模糊滤波器的滤波核（奇数）
    :param edgeKsize:边缘检测滤波器的滤波核（奇数）
    """
    if blurKsize >= 3:
        bluredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(bluredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0/255)*(255-graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel*normalizedInverseAlpha
    cv2.merge(channels, dst)


def contourDetection(src, threshold=50, epsilon=0.005, **kwargs):
    """
    轮廓检测，在源图像src绘制
    :param src: 源图像
    :param threshold: 二值化阈值
    :param epsilon: 轮廓近似多边形与原轮廓的差别
    :param kwargs: 轮廓检测方法 ex: style="contours"
    :return:
    """
    ret, thresh = cv2.threshold(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY),
                                threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if kwargs["style"] == "contours":
        cv2.drawContours(src, contours, -1, (0, 0, 255), 1)
    else:
        for c in contours:
            # 目标轮廓的外接矩形
            if kwargs["style"] == "boundingRect":
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 1)

            # 包围目标的最小矩形
            elif kwargs["style"] == "minAreaRect":
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(src, [box], 0, (255, 0, 0), 1)

            # 包围目标的最小闭圆
            elif kwargs["style"] == "minEnclosingCircle":
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                radius = int(radius)
                src = cv2.circle(src, center, radius, (0, 255, 0), 1)

            # 包围目标的近似多边形
            elif kwargs["style"] == "approxPolyDP":
                diff = epsilon * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, diff, True)
                cv2.drawContours(src, [approx], -1, (0, 0, 255), 1)

            # 包围目标的近似多边形
            elif kwargs["style"] == "convexHull":
                hull = cv2.convexHull(c)
                cv2.drawContours(src, [hull], -1, (0, 0, 255), 1)


class VConvolutionFilter(object):
    """自定义核的通用卷积滤波器"""

    def __init__(self, kernel):
        self._kernel = kernel  # 卷积核

    def apply(self, src, dst):
        """将滤波器应用到BGR或gary源图像上得到目标图像"""

        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    """半径为1-pixel的锐化滤波器"""

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    """半径为2像素的模糊滤波器"""

    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [-0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """半径为1像素的浮雕滤波器"""

    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)


if __name__ == "__main__":
    img = cv2.imread("../data/test/3.jpg")
    contourDetection(img, threshold=50, style="contours")
    # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.namedWindow("1", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("1", img)
    cv2.waitKey()
