"""图像分割"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def segGrabCut(src, roiRect=None):
    """
    使用GrabCut算法进行图像分割
    :param src: 源图像
    :param roiRect: 矩形区域 ex: roiRect=(0, 0, img.shape[0]-1, img.shape[1]-1)
    :return:
    """
    # 创建掩模
    mask = np.zeros(src.shape[:2], np.uint8)

    # 创建前景背景模型
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(src, mask, roiRect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    ret = src*mask2[:, :, np.newaxis]
    return ret


if __name__ == "__main__":
    img = cv2.imread("../data/test/3.jpg")
    segImg = segGrabCut(img, roiRect=(0, 0, img.shape[0]-1, img.shape[1]-1))
    plt.imshow(segImg, cmap="gray"), plt.colorbar(), plt.show()
