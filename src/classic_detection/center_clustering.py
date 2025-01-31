import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os


def create_tb(params, name='track bar'):
    """Creates track bars based on params dict"""
    def get_tb_func(var_name):
        def tb_func(value):
            global params
            params[var_name][0] = value

        return tb_func

    cv2.namedWindow(name)
    cv2.resizeWindow(name, 400, 1000)
    for key in params.keys():
        cv2.createTrackbar(key, name, params[key][0], params[key][2], get_tb_func(key))
        cv2.setTrackbarMin(key, name, params[key][1])
        cv2.setTrackbarPos(key, name, params[key][0])


def check_contour(contour):
    """Check whether contour is appropriate"""
    s = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    if s > params['s'][0] and h < params['max h'][0]:
        return True

    return False


def check_rough_bbox(bbox):
    x, y, w, h = bbox
    if (params['min bbox height'][0] < h < params['max bbox height'][0] and
            w / h > params['min bbox aspect ratio'][0] and w > params['min bbox width'][0]):
        return True
    return False

def get_rough_bboxes(contours):
    """
    Calculates line bboxes with DBSCAN applied to centers of specified contours
    :param contours: contours of letters
    :return: bboxes (x, y, w, h)
    """
    data_int = [[], []]
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if len(cnt) > params['len'][0]:
            data_int[0].extend([cy] * 3)

            x, y, w, h = cv2.boundingRect(cnt)
            data_int[1].append(x)
            data_int[1].append(x + w - 1)
            data_int[1].append(cx)
        else:
            data_int[0].append(cy)
            data_int[1].append(cx)
    data_int = [np.array(x, dtype=np.uint16) for x in data_int]

    ### DBSCAN
    rects = []
    data = [
        data_int[0].astype(dtype=np.float64) * (params['k'][0]),
        data_int[1].astype(dtype=np.float64),
    ]

    if len(data[0]) > 0:
        clustering = DBSCAN(eps=params['eps'][0], min_samples=params['min samples'][0], metric='l2')
        clustering.fit(np.transpose(data))
        labels = np.unique(clustering.labels_)
        for label in labels:
            idxs = np.nonzero(clustering.labels_ == label)

            x1 = np.min(data_int[1][idxs])
            x2 = np.max(data_int[1][idxs])
            y1 = np.min(data_int[0][idxs]) - 7
            y2 = np.max(data_int[0][idxs]) + 7

            rects.append((x1, y1, x2 - x1, y2 - y1))

    return rects

def find_bboxes(img):
    """
    Finds bboxes (x, y, w, h)
    :param img: numpy image (openCV-like: BGR)
    :return: list of bboxes
    """

    # resolution
    IMG_HEIGTH = 1280
    IMG_WIDTH = 960

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGTH))
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, params['bs'][0]*2+1, params['C'][0])

    conts, _ = cv2.findContours(thresh, 0, cv2.CHAIN_APPROX_SIMPLE)
    appr_conts = list(filter(check_contour, conts))

    # update thresh
    if appr_conts is not None:
        thresh = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(thresh, appr_conts, -1, (255, 255, 255), 1)

    rects = get_rough_bboxes(appr_conts)
    appr_rects = list(filter(check_rough_bbox, rects))


    cv2.imshow('inner', thresh)
    cv2.waitKey(1)
    ###
    return appr_rects


if __name__ == "__main__":
    img_path = r'..\..\data\images'
    color = (255, 150, 100)
    params = {
        'C': [27, 0, 50],
        'bs': [5, 1, 40],
        'speed': [0, 0, 50],
        's': [5, 0, 20],
        'len': [0, 0, 20],
        'eps': [34, 1, 1000],
        'min samples': [1, 1, 100],
        'k': [2, 1, 150],
        'max h': [70, 0, 70],
        'first dilate x': [3, 0, 5],
        'first dilate y': [3, 0, 5],
        'letter shift': [15, 0, 40],
        'min bbox aspect ratio': [1, 0, 10],
        'min bbox height': [10, 0, 20],
        'max bbox height': [100, 0, 400],
        'min bbox width': [50, 0, 300],
    }
    create_tb(params)

    images = os.listdir(img_path)
    for img_name in images:
        image = cv2.imread(os.path.join(img_path, img_name))
        IMG_HEIGTH = 1280
        IMG_WIDTH = 960
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGTH))
        bboxes = find_bboxes(image)
        while (params['speed'][0] == 0):
            bboxes = find_bboxes(image)
            image_2 = image.copy()
            for x, y, w, h in bboxes:
                cv2.rectangle(image_2, (x, y), (x + w, y + h), color, 2)
            cv2.imshow('res', image_2)
            cv2.waitKey(1)
        from math import ceil

        for x, y, w, h in bboxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.imshow('res', image)
        cv2.waitKey(ceil(1000 / params['speed'][0]))

