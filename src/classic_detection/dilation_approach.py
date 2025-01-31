import os

import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


def predict_text(images):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    processor = TrOCRProcessor.from_pretrained('kazars24/trocr-base-handwritten-ru')
    model = VisionEncoderDecoderModel.from_pretrained('kazars24/trocr-base-handwritten-ru').to(device=device)

    pix_val = processor(images=images, return_tensors='pt').pixel_values.to(device=device)
    gen_ids = model.generate(pix_val)
    texts = processor.batch_decode(gen_ids, skip_special_tokens=True)

    return texts
############ TRACK BARS
def get_tb_func(var_name):
    def tb_func(value):
        global params
        params[var_name][0] = value
    return tb_func
def create_tb(params, name='track bar'):
    cv2.namedWindow(name)
    for key in params.keys():
        cv2.createTrackbar(key, name, params[key][0], params[key][2], get_tb_func(key))
        cv2.setTrackbarMin(key, name, params[key][1])
        cv2.setTrackbarPos(key, name, params[key][0])
######
def find_bboxes(img):
    """
    Finds bboxes (x, y, w, h)
    :param img: numpy image (openCV-like: BGR)
    :return: list of bboxes
    """

    # resolution
    IMG_HEIGTH = 1280
    IMG_WIDTH = 960

    # contours acquisition
    LETTER_SHIFT = params['letter shift'][0]
    # contours selection
    MIN_AREA = 150
    MIN_CONV_RATIO = 0.45

    # bboxes generation
    BOT_PADDING = 5
    TOP_PADDING = 18
    LEFT_PADDING = 10
    RIGHT_PADDING = 20

    # bboxes selection
    ROI_H_RATIO = 0.99
    ROI_W_RATIO = 0.99

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGTH))
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, params['bs'][0]*2+1, params['C'][0])

    ### select cobtours
    ### show all images
    res = [thresh]
    conts, _ = cv2.findContours(thresh, 0, cv2.CHAIN_APPROX_SIMPLE)
    appr_conts = []
    if conts is not None:
        for cont in conts:
            s = cv2.contourArea(cont)
            if len(cont) > params['len'][0] and s > params['s'][0]:
                appr_conts.append(cont)
        thresh = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(thresh, appr_conts, -1, (255, 255, 255), 1)
    ###

    # dilate
    if params['first dilate x'][0] != 0 and params['first dilate y'][0] != 0:
        thresh = cv2.dilate(thresh, np.ones([params['first dilate y'][0], params['first dilate x'][0]], dtype=np.uint8))
    res.append(thresh)
    # merge letters to right
    kernel = np.ones([1, 2*LETTER_SHIFT+1], dtype=np.uint8)
    kernel[:, LETTER_SHIFT+1:2*LETTER_SHIFT+1] = np.zeros([1, LETTER_SHIFT])
    thresh = cv2.dilate(thresh, kernel)
    res.append(thresh)
    # fill spaces
    thresh = cv2.dilate(thresh, np.ones([3,3], dtype=np.uint8))
    res.append(thresh)
    # erode
    thresh = cv2.erode(thresh, np.ones([1, 30], dtype=np.uint8))
    res.append(thresh)

    # find appropriate contours
    conts, _ = cv2.findContours(thresh, 0, cv2.CHAIN_APPROX_SIMPLE)
    appr_conts = []
    if conts is not None:
        for cont in conts:
            convex = cv2.convexHull(cont)
            area = cv2.contourArea(cont, False)
            conv_area = cv2.contourArea(cont, False)
            if area > MIN_AREA and area / conv_area > MIN_CONV_RATIO:
                appr_conts.append(convex)
    else:
        print('No conts')

    # select appropriate bboxes
    rects = []
    for cont in appr_conts:
        # crop by mask
        x, y, w, h = cv2.boundingRect(cont)
        x -= LEFT_PADDING + LETTER_SHIFT
        y -= TOP_PADDING
        w += LEFT_PADDING + RIGHT_PADDING
        h += TOP_PADDING + BOT_PADDING

        # check if bbox (rect) is appropriate
        if h > w:
            continue
        if y < IMG_HEIGTH * ((1 - ROI_H_RATIO)//2):
            continue
        if y + h > IMG_HEIGTH * ((1 - ROI_H_RATIO)//2 + ROI_H_RATIO):
            continue
        if x < IMG_WIDTH * ((1 - ROI_W_RATIO) // 2):
            continue
        if x + w > IMG_WIDTH * ((1 - ROI_W_RATIO) // 2 + ROI_W_RATIO):
            continue

        rects.append((x, y, w, h))
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), 2)
    ### show res
    res.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    res_1 = np.concatenate(res[:3], axis=1)
    res_2 = np.concatenate(res[3:], axis=1)
    res = np.concatenate([res_1, res_2], axis=0)
    res = cv2.resize(res, (1080, 960))
    cv2.imshow('steps', res)
    cv2.waitKey(1)
    ###
    return rects


if __name__ == "__main__":
    video_path = r'data/images/video.MOV'
    img_path = r'..\..\data\images'
    showVideo = False
    color = (255, 150, 100)
    thickness = 1
    params = {
        'C': [27, 0, 50],
        'bs': [5, 1, 40], # 15
        'speed': [0, 0, 50],
        'first dilate x': [3, 0, 5],
        'first dilate y': [3, 0, 5],
        's': [5, 0, 20],
        'len': [0, 0, 20],
        'letter shift': [15, 0, 40],
    }
    create_tb(params)
    if showVideo:
        while(1):
            cap = cv2.VideoCapture(video_path)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cur_frame = 0
            while (cap.isOpened()):
                ret, frame = cap.read()

                if ret:
                    IMG_HEIGTH = 1280
                    IMG_WIDTH = 960
                    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGTH))
                    bboxes = find_bboxes(frame)
                    while (params['speed'][0] == 0):
                        bboxes = find_bboxes(frame)
                        cv2.waitKey(1)
                    from math import ceil
                    for x, y, w, h in bboxes:
                        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                    cv2.imshow('res', frame)
                    cv2.waitKey(ceil(50/params['speed'][0]))

                    cur_frame += 1
                    if cur_frame == total_frames - 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap.release()
            cv2.destroyAllWindows()
    else:
        while(1):
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
