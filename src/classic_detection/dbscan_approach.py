import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from sklearn.cluster import DBSCAN
from math import ceil
import os


def predict_text(images):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    processor = TrOCRProcessor.from_pretrained('kazars24/trocr-base-handwritten-ru')
    model = VisionEncoderDecoderModel.from_pretrained('kazars24/trocr-base-handwritten-ru').to(device=device)

    pix_val = processor(images=images, return_tensors='pt').pixel_values.to(device=device)
    gen_ids = model.generate(pix_val)
    texts = processor.batch_decode(gen_ids, skip_special_tokens=True)

    return texts


def check_bbox(bbox):
    x, y, w, h = bbox
    if (params['min bbox height'][0] < h < params['max bbox height'][0]
            and params['max bbox aspect ratio'][0] > w / h > params['min bbox aspect ratio'][0]
            and w > params['min bbox width'][0]):
        return True
    return False


############ TRACK BARS
def get_tb_func(var_name):
    def tb_func(value):
        global params
        params[var_name][0] = value
    return tb_func
def create_tb(params, name='track bar'):
    cv2.namedWindow(name)
    cv2.resizeWindow(name, 400, 1000)
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

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGTH))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, params['bs'][0] * 2 + 1,
                                     params['C'][0])

    ### select cobtours
    conts, _ = cv2.findContours(thresh, 0, cv2.CHAIN_APPROX_SIMPLE)
    if len(conts) > params['cont thresh'][0]:
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, params['bs alt'][0] * 2 + 1,
                                       params['C alt'][0])
    appr_conts = []
    if conts is not None:
        for cont in conts:
            s = cv2.contourArea(cont)
            x, y, w, h = cv2.boundingRect(cont)
            if len(cont) > params['len'][0] and s > params['s'][0] and h < params['max h'][0]:
                appr_conts.append(cont)
        thresh = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(thresh, appr_conts, -1, (255, 255, 255), 1)
        cv2.imshow('inner', thresh)
        cv2.waitKey(1)

    ### DBSCAN
    rects = []
    data_int = list(np.nonzero(thresh))
    data = [
        data_int[0].astype(np.float64) * (params['k'][0] / 100),
        data_int[1].astype(np.float64),
    ]

    if len(data[0]) > 0:
        clustering = DBSCAN(eps=params['eps'][0] / 100, min_samples=params['min samples'][0], metric='l2')
        clustering.fit(np.transpose(data))
        labels = np.unique(clustering.labels_)

        ###
        mask = np.zeros(image.shape, dtype=np.uint8)
        delta = 180 / len(labels)
        hue_values = [delta * i for i in range(len(labels))]
        colors_hsv = np.array([[[hue, 255, 255] for hue in hue_values]], dtype=np.uint8)
        colors = cv2.cvtColor(colors_hsv, cv2.COLOR_HSV2BGR)[0]
        np.random.seed(42)
        ###
        for i, label in enumerate(labels):
            idxs = np.nonzero(clustering.labels_ == label)

            ###
            mask[data_int[0][idxs], data_int[1][idxs], :] = colors[np.random.randint(0, len(colors))]
            ###

            x1 = np.min(data_int[1][idxs])
            x2 = np.max(data_int[1][idxs])
            y1 = np.min(data_int[0][idxs])
            y2 = np.max(data_int[0][idxs])

            rects.append((x1, y1, x2 - x1, y2 - y1))
        cv2.imshow('mask', mask)
        cv2.waitKey(1)
    rects = list(filter(check_bbox, rects))

    return rects


if __name__ == "__main__":
    video_path = r'data/images/video.MOV'
    img_path = r'..\..\data\images'
    showVideo = False
    color = (255, 150, 100)
    thickness = 1
    params = {
        'C': [27, 0, 50],
        'bs': [4, 1, 40],
        'C alt': [27, 0, 50],
        'bs alt': [4, 1, 40],
        'speed': [0, 0, 50],
        's': [5, 0, 20],
        'cont thresh': [10000, 0, 10000],
        'len': [0, 0, 20],
        'eps': [1934, 1, 2500],
        'min samples': [8, 1, 100],
        'k': [1564, 1, 5000],
        'max h': [70, 0, 70],
        'min bbox aspect ratio': [1, 0, 10],
        'max bbox aspect ratio': [13, 0, 30],
        'min bbox height': [10, 0, 20],
        'max bbox height': [200, 0, 400],
        'min bbox width': [50, 0, 300],
    }
    create_tb(params)
    if (showVideo):
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
                        frame_2 = frame.copy()
                        for x, y, w, h in bboxes:
                            cv2.rectangle(frame_2, (x, y), (x + w, y + h), color, 2)
                        cv2.imshow('res', frame_2)
                        cv2.waitKey(1)
                    if bboxes is not None:
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
