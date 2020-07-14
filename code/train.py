import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from visdom import Visdom

from utils import AverageMeter, VisdomLinePlotter

from ShapDataset import ShapDataset
from models import BinaryClassModel

import argparse

global plotter
plotter = VisdomLinePlotter('attack-classifier')

arg_parser = argparse.ArgumentParser(description='Use Captum to explain samples from RETAIN')

arg_parser.add_argument('--learning_rate', '-lr', help='Learning rate for optimiser', default=0.0001, type=float)
arg_parser.add_argument('--epochs', '-e', help='Number of epochs to train for', default=10, type=int)
arg_parser.add_argument('--batch_size', '-b', help='Batch size for training', default=32, type=int)
arg_parser.add_argument('--hidden_dim', '-d', help='Size of hidden layer', default=500, type=int)

arg_parser.add_argument('--test', '-t', help='Test the dataset', action='store_true')
arg_parser.add_argument('--save', '-s', help='Save trained model', action='store_true')

args = arg_parser.parse_args()

def train_epoch(model, train_loader, epoch, criterion, optimiser, device, verbose=False):
    losses = AverageMeter()
    model.train()

    # For each batch
    for i, (data, labels) in enumerate(train_loader):
        data = data.requires_grad_()
        labels = labels.type(torch.FloatTensor)

        # Send the data to the device (GPU hopefully!)
        data = data.to(device)
        labels = labels.to(device)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward,
        # dont want to cummulate gradients
        optimiser.zero_grad()
        outputs = model(data.float())

        if verbose:
            print(outputs)
            print(labels)

        loss = criterion(outputs, labels)
        losses.update(loss.data.cpu().numpy(), len(labels))

        loss.backward()
        optimiser.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
               epoch, i, len(train_loader), 100. * i / len(train_loader), loss=losses))

    plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)


def val_epoch(model, val_loader, epoch, criterion, device):
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for i, (data, labels) in enumerate(val_loader):
            data = data.requires_grad_()
            labels = labels.type(torch.FloatTensor)

            # Send to device (GPU hopefully!)
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data.float())
            loss = criterion(outputs, labels)
            losses.update(loss.data.cpu().numpy(), labels.size(0))

            predicted = F.softmax(outputs)

            # We need to create the array on the first batch, append to it afterwards
            if i == 0:
                out = predicted.data.cpu().numpy()
                label = labels.cpu().numpy()
            else:
                out = np.concatenate((out, predicted.data.cpu().numpy()), axis=0)
                label = np.concatenate((label, labels.cpu().numpy()), axis=0)

            for j in range(labels.size(0)):
                if predicted[j] == labels[j]:
                    correct += 1

            total += labels.size(0)

        acc = (correct / total) * 100

        print('Validation set: Average loss: {:.4f}\tAccuracy {acc}'.format(losses.avg, acc=acc))

        plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
        plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)

        # Return acc as the validation outcome
        return acc

# See if we can use CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("======================= Using device:", device)

print("======================= Loading Dataset")
dataset = ShapDataset('normal_shap.pkl', 'adv_shap.pkl')

if args.test:
    print('======================= Testing Dataset')
    total_all_0 = 0
    for shap, label in iter(dataset):
        count = 0
        for v in shap:
            if v != 0:
                count += 1

        if count == 0:
            print('======================= All 0 SHAP values!')
            total_all_0 += 1

    print(total_all_0, 'samples have all 0 values')


# Randomly create training and validation datasets (for now)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
indices = list(range(len(dataset)))

np.random.shuffle(indices)
train_indices, val_indices = indices[:train_size], indices[train_size:]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

batch_size = args.batch_size
epochs = args.epochs

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
print("======================= Dataset Loaded!")

input_dim = dataset.num_columns
hidden_dim = args.hidden_dim

# Initialise our model
model = BinaryClassModel(input_dim, hidden_dim)
model.float()
model.to(device)

best = 0

lr = args.learning_rate
optimiser = torch.optim.SGD(model.parameters(), lr)

for epoch in range(epochs):
    print('======================= Epoch', epoch)
    train_epoch(model, train_loader, epoch, nn.BCELoss(), optimiser, device=device)

    loss_val = val_epoch(model, val_loader, epoch, nn.BCELoss(), device=device)

    best = max(loss_val, best)
    print('** Validation: %f (best) - %f (current)' % (best, loss_val))

print('============================================== Training complete!')

if args.save:
    torch.save(model, './mimic-classifier.pt')
