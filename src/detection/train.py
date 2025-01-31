import logging
import sys

import cv2
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset
from torchmetrics.detection import MeanAveragePrecision
from cjm_pytorch_utils.core import set_seed
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import optuna

from src.detection.constants import *
from models import get_faster_rcnn


class DetectionDataset(Dataset):
    """
    Dataset for segmentation model training. Requires json COCO-like annotations path and images path
    """
    def __init__(self, img_path: str, annot_path: str, transform = None):
        """
        Args:
            img_path: path to images
            annot_path: path to COCO-like annotations
            transform: transforms to be applied
        """
        self.transform = transform
        self.img_path = img_path

        with open(annot_path, 'r') as f:
            data_js = json.load(f)
            data = {image['id']: {'img_name': image['file_name'],
                                  'height': image['height'],
                                  'width': image['width'],
                                  'bboxes': [],
                                  } for image in data_js['images']}
            for annot in data_js['annotations']:
                x,y,w,h = annot['bbox']
                img_w = data[annot['image_id']]['width']
                img_h = data[annot['image_id']]['height']
                if w * DETECTION_RESOLUTION[0] / img_w > 1 and h * DETECTION_RESOLUTION[1] / img_h > 1:
                    coordinates = x, y, x + w, y + h
                    data[annot['image_id']]['bboxes'].append(np.array(coordinates, dtype=np.float32))

            keys = sorted(list(data.keys()))
            self.data = {i: data[keys[i]] for i in range(len(keys))}
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # image
        image = cv2.imread(str(os.path.join(self.img_path, self.data[idx]['img_name'])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # boxes
        boxes = self.data[idx]['bboxes']
        boxes = np.stack(boxes, axis=0)

        # transformations
        transformed = self.transform(image=image, bboxes=boxes, labels=np.ones(len(boxes)))
        image = transformed['image']
        boxes = transformed['bboxes']

        image = torch.tensor(np.transpose(image, [2, 0, 1]))
        target_dict = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.ones(len(boxes), dtype=torch.int64),
        }

        return image, target_dict


def collate_fn(batch):
    """Collate function used in dataloader"""
    return tuple(zip(*batch))


def run_epoch(model, dataloader, device = 'cpu', is_training = False,
              optimizer = None, epoch_number: int = None) -> float:
    """
    Function to run a single training or evaluation epoch.
    Args:
        model: A PyTorch model to train or evaluate.
        dataloader: A PyTorch DataLoader providing the data.
        device: The device (CPU or GPU) to run the model on.
        is_training: Boolean flag indicating whether the model is in training or evaluation mode.
        optimizer: The optimizer to use for training the model (used in training mode only)

    Returns:
        The average batch loss for train mode or batch mAP (AP) for test mode
    """

    if is_training:
        model.train()
        sum_loss = 0
    else:
        model.eval()
        epoch_AP = MeanAveragePrecision()

    desc = ("Train" if is_training else "Eval") + (f' {epoch_number}' if epoch_number is not None else '')
    progress_bar = tqdm(total=len(dataloader), desc=desc)

    for batch_id, (inputs, targets) in enumerate(dataloader, 1):
        inputs = torch.stack(inputs).to(device)

        # copy data for AP computation
        if not is_training:
            targets_copied = []
            for target in targets:
                target_copied = {}
                for key in target.keys():
                    target_copied[key] = target[key].clone().detach()
                targets_copied.append(target_copied)

        # move data
        inputs = inputs.to(device)
        for target in targets:
            for key in target.keys():
                target[key] = target[key].to(device)

        if is_training:
            losses = model(inputs, targets)
            loss = sum(loss for loss in losses.values())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_item = loss.item()
            sum_loss += loss_item
            progress_bar.set_postfix({
                'avg batch loss': sum_loss / batch_id,
                'last batch loss': loss_item,
            })
            progress_bar.update()
        else:
            with torch.no_grad():
                preds = model(inputs, targets)

                for pred in preds:
                    for key in pred.keys():
                        pred[key] = pred[key].to('cpu')

                epoch_AP.update(preds, targets_copied)
                progress_bar.update()

    if is_training:
        ret_val = sum_loss / batch_id
    else:
        ret_val = epoch_AP.compute()['map'].item()
        progress_bar.set_postfix({
            'AP': ret_val,
        })

    progress_bar.close()

    return ret_val


class EarlyStopper:
    def __init__(self, patience, min_increment):
        self.patience = patience
        self.min_increment = min_increment
        self.counter = 0
        self.max_val = 0
    def __call__(self, current_val: float):
        if current_val - self.max_val > self.min_increment:
            self.max_val = current_val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == self.patience:
                return True

        return False


def train(train_ratio: float = 0.9, learning_rate: float = 1e-6, epochs_number: int = 50, **kwargs) -> float:
    """
    Trains Faster R-CNN with specified params
    Args:
        train_ratio:
        learning_rate:
        epochs_number:
        **kwargs:

    Returns:
        maximal test Average Precision
    """

    batch_size = 6
    num_workers = 1
    pin_memory = True
    seed = 42

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    set_seed(seed)
    model = get_faster_rcnn().to(device)

    train_transform = A.Compose([
        A.RandomSizedCrop(min_max_height=(kwargs['min_height'], kwargs['max_height']) if
        'min_height' and 'max_height' in kwargs else MIN_MAX_HEIGHT, size=DETECTION_RESOLUTION),
        A.HorizontalFlip(p=kwargs['HorizontalFlip'] if 'HorizontalFlip' in kwargs else 0),
        A.VerticalFlip(p=kwargs['VerticalFlip'] if 'VerticalFlip' in kwargs else 0),
        A.Illumination(p=kwargs['Illumination'] if 'Illumination' in kwargs else 0),
        A.MotionBlur(p=kwargs['MotionBlur'] if 'MotionBlur' in kwargs else 0),
        A.RandomBrightnessContrast(p=kwargs['RandomBrightnessContrast'] if 'RandomBrightnessContrast' in kwargs else 0),
        A.ColorJitter(p=kwargs['ColorJitter'] if 'ColorJitter' in kwargs else 0),
        A.GaussNoise(p=kwargs['GaussNoise'] if 'GaussNoise' in kwargs else 0),
        A.AdditiveNoise(p=kwargs['AdditiveNoise'] if 'AdditiveNoise' in kwargs else 0),
        A.Normalize(*(kwargs['Normalize'] if 'Normalize' in kwargs else ((0, 0, 0), (1, 1, 1))), max_pixel_value=255),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], **BBOX_PARAMS))

    test_transform = A.Compose([
        A.CenterCrop(*TEST_CROP_RESOLUTION),
        A.Resize(*DETECTION_RESOLUTION),
        A.Normalize(*(kwargs['Normalize'] if 'Normalize' in kwargs else ((0, 0, 0), (1, 1, 1))), max_pixel_value=255),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], **BBOX_PARAMS))

    train_dataset = DetectionDataset(img_path=IMAGES_PATH, annot_path=ANNOT_PATH, transform=train_transform)
    test_dataset = DetectionDataset(img_path=IMAGES_PATH, annot_path=ANNOT_PATH, transform=test_transform)
    dataset_length = len(train_dataset.data)
    train_samples_number = int(dataset_length * train_ratio)
    indices = np.random.permutation(range(dataset_length))
    train_indices, test_indices = indices[:train_samples_number], indices[train_samples_number:]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_APs = []

    early_stopper = EarlyStopper(patience=kwargs['patience'] if 'patience' in kwargs else 10,
                                 min_increment=kwargs['min_increment'] if 'min_increment' in kwargs else 0)

    try:
        for epoch in range(epochs_number):
            train_loss = run_epoch(model, data_loader_train, device, True, optimizer, epoch)
            train_losses.append(train_loss)
            test_AP = run_epoch(model, data_loader_test, device, False, epoch_number=epoch)
            test_APs.append(test_AP)

            if early_stopper(test_AP):
                break
    finally:
        if 'plot_save_path' in kwargs:
            figure, (train_ax, test_ax) = plt.subplots(2, 1, figsize=(5, 6))
            train_ax.set_title('Train Losses')
            test_ax.set_title('Test Average Precision')
            train_ax.plot(train_losses)
            test_ax.plot(test_APs)
            plt.savefig(kwargs['plot_save_path'])
            print(f'Plot has been saved as {kwargs["plot_save_path"]}')

        if 'model_save_path' in kwargs:
            if not 'model_save_thresh' in kwargs or max(test_APs) > kwargs['model_save_thresh']:
                torch.save(model, kwargs['model_save_path'])
                print(f'Model has been saved at {kwargs["model_save_path"]}')

    return max(test_APs)


def objective(trial) -> float:
    """Objective function for optuna"""
    n = len(os.listdir(r'../../data/training/plots'))
    plot_name = f'plot_{n}.png'
    model_name = f'model_{n}.pt'

    params = {
        # base
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-4),
        'epochs_number': 60,
        'patience': 20,
        'min_increment': 0.05,
        'min_height': trial.suggest_int('min_height', 800, 2000, step=100),
        'max_height': trial.suggest_int('max_height', 2000, 3000, step=100),
        # safe (?)
        'HorizontalFlip': trial.suggest_float('HorizontalFlip', 0, 1),
        'VerticalFlip': trial.suggest_float('VerticalFlip', 0, 1),
        'Illumination': trial.suggest_float('Illumination', 0, 1),
        'MotionBlur': trial.suggest_float('MotionBlur', 0, 1),
        # experimental
        'RandomBrightnessContrast': trial.suggest_float('RandomBrightnessContrast', 0, 1),
        'ColorJitter': trial.suggest_float('ColorJitter', 0, 1),
        'GaussNoise': trial.suggest_float('GaussNoise', 0, 1),
        'AdditiveNoise': trial.suggest_float('AdditiveNoise', 0, 1),
        # base
        'Normalize': (
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            if trial.suggest_categorical('Normalize', ['standart', 'min_max']) == 'standart'
            else ((0, 0, 0), (1, 1, 1)),
        # save information
        'plot_save_path': f'../../data/training/plots/{plot_name}',
        'model_save_path': f'../../data/training/models/{model_name}',
        'model_save_thresh': 0.6,
    }
    max_AP = train(**params)

    return max_AP


def conduct_study():
    """Conduct training using optuna"""
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "OCR experiment 1"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')
    study.optimize(objective, n_trials=50)


if __name__ == "__main__":
    params = {
        # base
        'learning_rate': 6.5e-05,
        'epochs_number': 100,
        'patience': 100,
        'min_increment': 0.10,
        'min_height': 1700,
        'max_height': 2300,
        # safe (?)
        'MotionBlur': 0.5,
        'ColorJitter': 0.25,
        'AdditiveNoise': 0.9,
        # save information
        # 'plot_save_path': f'../../data/training/plots/opt_model.png',
        # 'model_save_path': f'../../data/training/models/opt_model.pt',
    }
    max_AP = train(**params)
