import cv2
import numpy as np
import torch
from torchvision.transforms import v2
from sklearn.cluster import DBSCAN
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from constants import *
from Levenshtein import distance

def find_bboxes(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    """
        Utilizes given model for prediction
    Args:
        model: torch model to be used
        image: RGB image with shape (800, 800, 3)

    Returns:
        bounding boxes
    """
    # preprocessing
    preprocess  = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(DETECTION_RESOLUTION),
    ])
    input = preprocess(image)
    input = input.unsqueeze(0)

    # prediction
    model.eval()
    prediction = model(input)
    boxes = prediction[0]['boxes'].detach().to('cpu').numpy()

    return boxes


def find_clusters(bboxes: np.ndarray) -> list[np.ndarray]:
    """
    Finds groups of bboxes representing the same student name based on avg_y of bbox
    Args:
        bboxes: numpy array of shape (n_bboxes, 4) in XYXY format

    Returns:
        clusters
    """
    clustering = DBSCAN(**POSTPROCESSING_DBSCAN_PARAMS)
    y1 = bboxes[:, 1]
    y2 = bboxes[:, 3]
    pos = (y1 + y2) / 2

    clustering.fit(pos.reshape(-1, 1))
    boxes_clusters = []
    for label in np.unique(clustering.labels_):
        boxes_i = bboxes[clustering.labels_ == label, :]

        # sorting bboxes based on x1 value
        indices = boxes_i[:, 0].argsort()
        sorted_bboxes = boxes_i[indices, :]
        boxes_clusters.append(sorted_bboxes)

    return boxes_clusters


def merge_bboxes(bboxes: np.ndarray) -> np.ndarray:
    """
    Merge boxes in a given list based on their position
    Args:
        bboxes: list of bboxes of shape (n_samples, 4) in XYXY format

    Returns:
        merged bboxes (n_after_merge, 4)
    """

    def are_neighbours(a: np.ndarray, b: np.ndarray, thresh: int) -> bool:
        """Check whether a and b should be merged during postprocessing stage"""
        if a[0] < b[0]:
            left = a
            right = b
        else:
            left = b
            right = a

        if right[0] - left[2] < thresh:
            return True

        return False

    def get_merged_box(a):
        """Returns merged bbox based on a given list of bboxes"""
        x1 = min([bbox[0] for bbox in a])
        x2 = max([bbox[2] for bbox in a])
        y1 = sum([bbox[1] for bbox in a]) / len(a)
        y2 = sum([bbox[3] for bbox in a]) / len(a)

        return np.array([x1, y1, x2, y2], dtype=np.uint32)

    unprocessed = [bboxes[i] for i in range(len(bboxes))]
    processed = []
    while len(unprocessed) > 1:
        neighbours = [unprocessed[0]]
        for i in range(len(unprocessed) - 1, 0, -1):
            if are_neighbours(unprocessed[0], unprocessed[i], POSTPROCESSING_MERGE_PARAMS['x_thresh']):
                neighbours.append(unprocessed[i])
                del unprocessed[i]

        if len(neighbours) > 1:
            merged = get_merged_box(neighbours)
            unprocessed.append(merged)
        else:
            processed.append(unprocessed[0])
        del unprocessed[0]

    processed.append(unprocessed[0])
    res = np.vstack(processed)

    return res


def recognise_text(image: np.ndarray, bboxes: np.ndarray, processor, model, device) -> list[str]:
    """
    Processes given image in specified areas using given model
    Args:
        image: RGB image of shape (H, W, 3) with any resolution
        bboxes: numpy array of shape (n_samples, 4)
        processor: processor to be used for pre/post processing
        model: model to be used for recognition
        device: device for computation

    Returns:
        list of recognised texts
    """
    # scaling bboxes
    IMG_HEIGHT, IMG_WIDTH, _ = image.shape
    BBOX_IMG_H, BBOX_IMG_W = DETECTION_RESOLUTION
    x_scale = IMG_WIDTH / BBOX_IMG_W
    y_scale = IMG_HEIGHT / BBOX_IMG_H
    scale_vector = [x_scale, y_scale, x_scale, y_scale]
    scaled_bboxes = bboxes * scale_vector

    # crop ROIs from original image based on scaled bboxes
    images = []
    for bbox in scaled_bboxes:
        x1, y1, x2, y2 = [int(el) for el in bbox]
        images.append(image[y1:y2, x1:x2, :])

    pix_val = processor(images=images, return_tensors='pt').pixel_values.to(device=device)
    gen_ids = model.generate(pix_val)
    gen_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)

    return gen_texts


def recognise_cluster(cluster_texts: list[str], possible_names: list[list[str]], name_parts_considered: int = 1,
                      max_wordwise_distance: float = 10000.0) -> [str, int, list[int], float]:
    """
    Recognise student name corresponding to given cluster based on possible names
    Args:
        cluster_texts: list of recognised words to be considered
        possible_names: list with possible names (lowercase)
        name_parts_considered: number of name parts which should be considered
        max_wordwise_distance: maximum appropriate Levenshtein_distance / len(name_part) between each word-name_part pair

    Returns: index corresponding to recognised student,
    list with indexes of used bboxes (of name_parts_considered length), mean wordwise distance
    """
    min_total_distance = 10000.0
    cluster_texts = list(map(lambda x: x.lower(), cluster_texts))

    res = None
    res_idx = None
    usage_info = [None] * name_parts_considered
    for k, full_name in enumerate(possible_names):
        cur_total_distance = 0
        rejected = False
        word_pool = cluster_texts.copy()

        for j, name_part in enumerate(full_name[:name_parts_considered]):
            min_word_distance = 10000.0
            i_min = None
            for i, word in enumerate(word_pool):
                cur_word_distance = distance(word, name_part) / len(name_part)
                if cur_word_distance < min_word_distance:
                    min_word_distance = cur_word_distance
                    i_min = i

            if min_word_distance < max_wordwise_distance:
                del word_pool[i_min]
                cur_total_distance += min_word_distance
                usage_info[j] = i_min
            else:
                rejected = True
                break

        if cur_total_distance < min_total_distance and not rejected:
            min_total_distance = cur_total_distance
            res = ' '.join(full_name)
            res_idx = k

    return res, res_idx, usage_info, min_total_distance / name_parts_considered


def visualize(image, clusters, used_infos, texts = None):
    """Visualization for bbox clusters"""
    image = cv2.resize(image, (800, 800))
    img_1 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_2 = np.zeros(image.shape, dtype=np.uint8)

    delta = 180 / len(clusters)
    hue_values = [delta * i for i in range(len(clusters))]
    colors_hsv = np.array([[[hue, 255, 255] for hue in hue_values]], dtype=np.uint8)
    colors = cv2.cvtColor(colors_hsv, cv2.COLOR_HSV2BGR)[0]
    for i, cluster in enumerate(clusters):
        for j, box in enumerate(cluster):
            if j not in used_infos[i]:
                continue
            x1, y1, x2, y2 = [int(el) for el in box]
            cv2.rectangle(img_1, (x1, y1), (x2, y2), colors[i].tolist(), 1)
            if texts is not None:
                cv2.putText(img_2, texts[i][j], (x1, y1 + 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, colors[i].tolist(), 1,
                            lineType=cv2.LINE_AA)

    res = np.concatenate([img_1, img_2], axis=1)
    cv2.imshow('res', res)
    cv2.waitKey(1)

    return res


def main():
    path = r'..\..\data\images2'

    # prepare detection
    det_model = torch.load('../../data/training/models/opt_model.pt', weights_only=False).to('cpu')

    # prepare recognition
    device = 'cuda:0'
    processor = TrOCRProcessor.from_pretrained('raxtemur/trocr-base-ru')
    rec_model = VisionEncoderDecoderModel.from_pretrained('raxtemur/trocr-base-ru').to(device=device)

    # prepare database info
    with open('../../data/student_names.txt', 'r', encoding='utf-8') as f:
        database = f.read()
        database = [[x.lower() for x in full_name.split()] for full_name in database.split('\n')]


    for img_name in os.listdir(path):
        image = cv2.imread(os.path.join(path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # processing
        bboxes = find_bboxes(det_model, image)
        clusters = find_clusters(bboxes)
        clusters = [merge_bboxes(cluster) for cluster in clusters]
        texts = []
        for cluster in clusters:
            text = recognise_text(image, cluster, processor, rec_model, device)
            texts.append(text)

        # student recognition
        rec_students = []
        for text in texts:
            student, idx, usage_info, d = recognise_cluster(text, database, 3, 0.5)
            rec_students.append(student)

        print('Recognised students:', rec_students)

        # visualization
        visualize(image, clusters, texts)


if __name__ == "__main__":
    main()
