# ##
# image2 = (image.copy() * 255).astype(np.uint8)
# for box in boxes:
#     x1, y1, x2, y2 = [int(x) for x in box]
#     cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 0, 255), 2)
# img = cv2.imread(r'C:\kirill\training\projects\ocr_lastname\text-recognition\data\images\5440470954456245625.jpg')
# res = np.concat([image2, cv2.resize(img, (800, 800))], axis=1)
# cv2.imshow('after transforms', cv2.resize(res, (1600, 800)))
# cv2.waitKey(1)
# ##
# import logging
# import sys
# import optuna
# from plotly.io import show
# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# study_name = "OCR experiment 1"
# storage_name = "sqlite:///{}.db".format(study_name)
# training = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')
# print(training.best_params)
# # print(training.get_trials())
# # print(training.best_trial)
# fig = optuna.visualization.plot_param_importances(training)
# show(fig)
from Levenshtein import distance

with open('corpus.txt', 'r') as f:
    corpus = f.read()
    corpus = [[cluster.split(';;;') for cluster in image.split(';;;;')] for image in corpus.split(';;;;;')]

with open('../../data/student_names.txt', 'r', encoding='utf-8') as f:
    database = f.read()
    database = [[x.lower() for x in full_name.split()] for full_name in database.split('\n')]


def word_is_appr(word: str, thresh: float):
    count = 0
    for char in word:
        if char.isalpha():
            count += 1
    if count / len(word) > thresh:
        return True
    return False

word_thresh = 0.9
name_parts_num = 2
name_parts_nums = [2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
for i, page in enumerate(corpus[:len(name_parts_nums)]):
    print(name_parts_nums[i])
    results = []
    tot_dists = []

    names_pull = database.copy()
    for cluster in page:
        j_min = None
        min_total_distance = 10000.0

        cluster = list(filter(lambda x: word_is_appr(x, 0.5), cluster))
        cluster = list(map(lambda x: x.lower(), cluster))

        res = None
        for j, full_name in enumerate(names_pull):
            cur_total_distance = 0
            rejected = False
            word_pool = cluster.copy()

            for name_part in full_name[:name_parts_nums[i]]:
                min_word_distance = 10000.0
                i_min = None
                for i, word in enumerate(word_pool):
                    cur_word_distance = distance(word, name_part) / len(name_part)
                    if cur_word_distance < min_word_distance:
                        min_word_distance = cur_word_distance
                        i_min = i

                if min_word_distance < word_thresh:
                    del word_pool[i_min]
                    cur_total_distance += min_word_distance
                else:
                    rejected = True
                    break

            if cur_total_distance < min_total_distance and not rejected:
                min_total_distance = cur_total_distance
                res = full_name
                j_min = j
        results.append(res)
        tot_dists.append(min_total_distance)
        if j_min is not None:
            del names_pull[j_min]

    import numpy as np
    idxs = np.argsort(tot_dists)
    for i in idxs:
        print(f'{tot_dists[i]:.4}', results[i], list(filter(lambda x: word_is_appr(x, 0.5), page[i])))
