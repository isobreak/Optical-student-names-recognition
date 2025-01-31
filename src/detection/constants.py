import os

DATA_PATH = r'..\..\data\train_segmentation'
ANNOT_PATH = os.path.join(DATA_PATH, 'annotations.json')
ANNOT_EXT_PATH = os.path.join(DATA_PATH, 'annotations_extended.json')
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
DETECTION_RESOLUTION = (800, 800)
TEST_CROP_RESOLUTION = (1960, 1960)
BBOX_PARAMS = {
    'min_visibility': 0.25,
}
MIN_MAX_HEIGHT = (1600, 3000)
POSTPROCESSING_DBSCAN_PARAMS = {
    'eps': 8,
    'min_samples': 1,
    'metric': 'l1',
}
POSTPROCESSING_MERGE_PARAMS = {
    'x_thresh': 1,
}