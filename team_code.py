import joblib
import numpy as np
import os
import random
import shutil
import os
import shutil
import joblib
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from collections import OrderedDict
from data_loader import get_training_and_validation_loaders
from functools import partial
from helper_code import *
from matplotlib import pyplot as plt
from cnn import VGGMODEL
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from typing import Callable, Optional

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_DISTANCE_TO_MAX_THRESHOLD = 0.1
LIST_OF_ALL_LABELS = ['NORM', 'Acute MI', 'Old MI', 'STTC', 'CD', 'HYP', 'PAC', 'PVC', 'AFIB/AFL', 'TACHY', 'BRADY']
RESIZE_TEST_IMAGES = (425, 550)
OPTIM_LR = 1e-4
OPTIM_WEIGHT_DECAY = 1e-4
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.1
GRAD_CLIP = 10


def train_models(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Loading the data...')

    classification_images = list()
    classification_labels = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        record_parent_folder = os.path.dirname(record)

        labels = load_labels(record)
        if labels:
            nonempty_labels = [l for l in labels if l != '']
            if nonempty_labels != []:
                images = get_image_files(record)
                classification_images.append(os.path.join(record_parent_folder, images[0]))
                classification_labels.append(nonempty_labels)

    if not classification_labels:
        raise Exception('There are no labels for the data.')

    num_classes = len(LIST_OF_ALL_LABELS)

    if verbose:
        print('Training the models on the data...')

    training_loader, validation_loader = get_training_and_validation_loaders(
        LIST_OF_ALL_LABELS, classification_images, classification_labels)

    classification_model = VGGMODEL(LIST_OF_ALL_LABELS).to(DEVICE)
    for param in classification_model.parameters():
        param.requires_grad = True

    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(classification_model.parameters(), lr=OPTIM_LR, weight_decay=OPTIM_WEIGHT_DECAY)
    scheduler = StepLR(opt, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    N_loss = []
    N_loss_valid = []
    train_auprc = []
    valid_auprc = []
    train_auroc = []
    valid_auroc = []
    f1_train = []
    f1_valid = []

    plot_folder = os.path.join(model_folder, "training_figures")
    os.makedirs(plot_folder, exist_ok=True)

    final_weights = None

    for epoch in range(EPOCHS):
        N_item_sum = 0
        N_item_sum_valid = 0
        targets_train = []
        outputs_train = []
        targets_valid = []
        outputs_valid = []

        if verbose:
            print(f"============================[{epoch}]============================")
        classification_model.train()
        for i, (image, label) in enumerate(training_loader):
            opt.zero_grad()

            image = image.float().to(DEVICE)
            label = label.float().to(DEVICE)
            prediction = classification_model(image)

            N = loss(prediction, label)
            N.backward()
            N_item = N.item()
            N_item_sum += N_item

            torch.nn.utils.clip_grad_norm_(classification_model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            if verbose:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {N_item}")

            targets_train.append(label.data.cpu().numpy())
            outputs_train.append(prediction.data.cpu().numpy())

        classification_model.eval()
        with torch.no_grad():
            for j, (image, label) in enumerate(validation_loader):
                image = image.float().to(DEVICE)
                label = label.float().to(DEVICE)
                prediction = classification_model(image)

                N = loss(prediction, label)
                N_item = N.item()
                N_item_sum_valid += N_item

                targets_valid.append(label.data.cpu().numpy())
                outputs_valid.append(prediction.data.cpu().numpy())
                print(f"Epoch: {epoch}, Valid Iteration: {j}, Loss: {N_item}")

        scheduler.step()

        targets_train = np.concatenate(targets_train, axis=0).T
        outputs_train = np.concatenate(outputs_train, axis=0).T
        targets_valid = np.concatenate(targets_valid, axis=0).T
        outputs_valid = np.concatenate(outputs_valid, axis=0).T

        auprc_t = average_precision_score(y_true=targets_train, y_score=outputs_train)
        auroc_t = roc_auc_score(y_true=targets_train, y_score=outputs_train)
        auprc_v = average_precision_score(y_true=targets_valid, y_score=outputs_valid)
        auroc_v = roc_auc_score(y_true=targets_valid, y_score=outputs_valid)

        train_auprc.append(auprc_t)
        train_auroc.append(auroc_t)
        valid_auprc.append(auprc_v)
        valid_auroc.append(auroc_v)

        N_loss.append(N_item_sum / len(training_loader))
        N_loss_valid.append(N_item_sum_valid / len(validation_loader))

        fig = plt.figure()
        plt.plot(N_loss, label="train")
        plt.plot(N_loss_valid, label="valid")
        plt.title("Loss function")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(plot_folder, "loss.png"))
        plt.close()

        fig = plt.figure()
        plt.plot(train_auprc, label="train auprc")
        plt.plot(valid_auprc, label="valid auprc")
        plt.plot(train_auroc, label="train auroc")
        plt.plot(valid_auroc, label="valid auroc")

        plt.title("AUPRC and AUROC")
        plt.xlabel('epoch')
        plt.ylabel('Performance')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(plot_folder, "auroc_auprc.png"))
        plt.close()

        file_path = os.path.join(model_folder, "model_weights_" + str(epoch) + ".pth")
        torch.save(classification_model.state_dict(), file_path)

        final_weights = file_path

    os.makedirs(model_folder, exist_ok=True)

    save_classification_model(model_folder, LIST_OF_ALL_LABELS, final_weights)

    if verbose:
        print('Done.')
        print()

def load_models(model_folder, verbose):
    digitization_model = None

    classes_filename = os.path.join(model_folder, 'classes.txt')
    classes = joblib.load(classes_filename)

    classification_model = VGGMODEL(classes).to(DEVICE)
    classification_filename = os.path.join(model_folder, "classification_model.pth")
    classification_model.load_state_dict(torch.load(classification_filename))

    return digitization_model, classification_model

def run_models(record, digitization_model, classification_model, verbose):
    signal = None

    classes = classification_model.list_of_classes

    record_parent_folder = os.path.dirname(record)
    image_files = get_image_files(record)
    image_path = os.path.join(record_parent_folder, image_files[0])
    img = Image.open(image_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = transforms.Resize(RESIZE_TEST_IMAGES)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    img = img.unsqueeze(0)

    img = img.float().to(DEVICE)

    classification_model.eval()
    with torch.no_grad():
        probabilities = torch.squeeze(classification_model(img), 0).tolist()
        predictions = []
        for i in range(len(classes)):
            if probabilities[i] >= CLASSIFICATION_THRESHOLD:
                predictions.append(classes[i])

    if predictions == []:
        highest_probability = max(probabilities)
        for i in range(len(classes)):
            if abs(highest_probability - probabilities[i]) <= CLASSIFICATION_DISTANCE_TO_MAX_THRESHOLD:
                predictions.append(classes[i])

    return signal, predictions

def extract_features(record):
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np


# Save your trained models.
def save_classification_model(model_folder, list_of_classes=None, final_weights=None):
    # Ensure the model folder exists
    os.makedirs(model_folder, exist_ok=True)    
    # Check if the final weights file is provided
    if final_weights is not None:
        # Save the list of classes
        if list_of_classes is not None:
            classes_filename = os.path.join(model_folder, 'classes.txt')
            joblib.dump(list_of_classes, classes_filename, protocol=0)
        else:
            raise ValueError("list_of_classes must be provided when saving the model.")
        # Copy the file with the final weights to the model path
        model_filename = os.path.join(model_folder, "classification_model.pth")
        
        if os.path.isfile(final_weights):
            shutil.copyfile(final_weights, model_filename)
        else:
            raise FileNotFoundError(f"File {final_weights} does not exist. Cannot copy to {model_filename}.")
    else:
        raise ValueError("final_weights must be provided when saving the model.")
