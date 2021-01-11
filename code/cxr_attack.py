import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader

from XrayDataset import XrayDataset
from utils import train, test

import argparse
import visdom

import os

import tqdm

import foolbox
import eagerpy as ep

import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser(description='Generate adversarial CXR images')

arg_parser.add_argument('--batch_size', '-b', help='Batch size to train with', type=int, default=64)
arg_parser.add_argument('--num_batches', '-n', help='Number of batches to compute', type=int, default=1)
arg_parser.add_argument('--mention', '-m', help='Positive (1) or negative labels (0)', type=int, default=1)
arg_parser.add_argument('--vis_adv', '-v', help='Plot adversarials and the difference', action='store_true')
arg_parser.add_argument('--save', '-s', help='Path to save adversarial images to', type=str, default=None)
arg_parser.add_argument('--only', '-o', help='Only look at images with this label', type=int, default=None)
arg_parser.add_argument('--ignore_unperturbed', help='Only consider images with have had to have been perturbed',
                        action='store_true')

arg_parser.add_argument('--attack', '-a', help='Attack to carry out (default: pgd)', type=str,
                        choices=['pgd', 'fgsm', 'cw'], default='pgd')

arg_parser.add_argument('model', help='Path to model to attack', type=str)
arg_parser.add_argument('label', help='Label to classify', type=str)
arg_parser.add_argument('data_path', help='Path to root of MIMIC-CXR data', type=str)
arg_parser.add_argument('collate_path', help='Path to collated labels CSV file', type=str)

args = arg_parser.parse_args()

# Sanity checks on the CLI arguments
if args.mention not in [0, 1]:
    raise Exception('Can only detect positive (1) or negative (0) presences in images')

# Load collated CSV file for the headers
collated_csv = pd.read_csv(args.collate_path, nrows=0)
labels = list(collated_csv.columns)[2:]

if args.label not in labels:
    raise Exception('Label must be one of those predicted by CheXpert')

# Must use CPU for foolbox to work
device = torch.device("cpu")
print("======================= Using device:", device)

# Load model
model = torch.load(args.model, map_location=device)
model.eval()
print('======================= Model loaded')

# Get a batch of images
include_filenames = False if args.save is None else True
dataset = XrayDataset(args.data_path, args.collate_path, args.label, args.mention, normalise=False,
                      include_filenames=include_filenames, only=args.only)
dataloader = iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=True))

for batch_num in range(args.num_batches):
    if include_filenames:
        images, labels, filenames = next(dataloader)
    else:
        images, labels = next(dataloader)

    images = images.to(device)
    print('\n================================= Image batch {} loaded ({} images)'.format(batch_num + 1,
                                                                                         args.batch_size))

    # Test accuracy of model on these images
    model = model.to(device)
    model = model.eval()
    outputs = model(images)
    preds = F.softmax(outputs, 1)
    _, preds = torch.max(preds, 1)

    print(preds)
    acc = torch.sum(preds.cpu() == labels).item() / args.batch_size
    print('======================= Accuracy on image batch:', acc)

    # Model to do adversarial attack on. bounds = pixel value bounds, preprocessing for DenseNet
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    foolbox_model = foolbox.models.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing, num_classes=2)

    # Apply the attack
    if args.attack == 'pgd':
        attack = foolbox.attacks.PGD(model=foolbox_model, distance=foolbox.distances.Linfinity)
    elif args.attack == 'cw':
        attack = foolbox.attacks.CarliniWagnerL2Attack(model=foolbox_model)
    else:
        attack = foolbox.attacks.FGSM(model=foolbox_model)

    epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]

    adversarials = attack(images.numpy(), labels.numpy())
    adversarials_class = attack(images.numpy(), labels.numpy(), unpack=False)
    print('======================= Adversaries generated')

    adv_preds = torch.from_numpy(foolbox_model.forward(adversarials))
    _, adv_preds = torch.max(F.softmax(adv_preds, 1), 1)

    adv_acc = torch.sum(adv_preds == labels).item() / args.batch_size

    # Because of the way Foolbox works, this should always be 0
    print('======================= Accuracy on adversarial batch:', adv_acc)

    if args.ignore_unperturbed:
        perturbed_adv = []
        for i in range(len(adversarials)):
            if adversarials_class[i].distance.value != 0:
                perturbed_adv.append(adversarials[i])

        print('Perturbed {}/{} images in this class'.format(len(perturbed_adv), len(adversarials_class)))
        adversarials = perturbed_adv

    if args.vis_adv:
        for i in range(len(adversarials)):
            img = images[i]
            img = np.transpose(img, (1, 2, 0))

            adv = adversarials[i]
            adv = np.transpose(adv, (1, 2, 0))

            print(type(img))
            print(type(adv))

            diff = abs(img - torch.from_numpy(adv))

            plt.subplot(2, 2, 1)
            plt.gca().set_title('Original Image')
            plt.imshow(img)

            plt.subplot(2, 2, 2)
            plt.gca().set_title('Adversarial Image')
            plt.imshow(adv)

            plt.subplot(2, 2, 3)
            plt.gca().set_title('Difference')
            plt.imshow(diff)

            plt.show()

    # Save the images if we want
    if args.save is not None:
        print('======================= Saving adversaries to', args.save)

        for i in range(len(adversarials)):

            # We want to save a jpeg not a dcm file
            filename = filenames[i][:-3] + 'jpg'
            adv = adversarials[i]
            adv = np.transpose(adv, (1, 2, 0))

            path = os.path.join(args.save, filename)
            plt.imsave(path, adv)