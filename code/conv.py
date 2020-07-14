import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ray import tune
import ray

import visdom
from ShapDataset import ShapDatasetFly

import argparse
from utils import AverageMeter, VisdomLinePlotter

import numpy as np

import tqdm
import os

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('normal_path', help='Location to saved normal images', type=str)
    arg_parser.add_argument('adversarial_path', help='Location to saved adversarial images', type=str)
    arg_parser.add_argument('model', help='Path to model', type=str)
    arg_parser.add_argument('collate_path', help='Path to collated_labels.csv', type=str)

    arg_parser.add_argument('--plot', help='Name of visdom plot', type=str, default='adv-mlp')

    arg_parser.add_argument('--optimise', '-o', help='Perform hyperparameter optimisation', action='store_true')

    arg_parser.add_argument('--lr', help='Learning rate during training', type=float, default=0.0001)
    arg_parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=100)
    arg_parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    arg_parser.add_argument('--save', help='Path to save best model to', type=str, default=None)
    arg_parser.add_argument('--name', '-n', help='Name of raytune experiment', type=str, default='train_conv')
    arg_parser.add_argument('--opt', help='Use optimisation', action='store_true')

    args = arg_parser.parse_args()

    print('=================== Loading dataset')
    dataset = ShapDatasetFly(args.normal_path, args.adversarial_path, args.collate_path, args.model)

    global plotter
    plotter = VisdomLinePlotter(args.plot)


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 53 * 53, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return F.sigmoid(x)

def train_epoch(model, train_loader, epoch, criterion, optimiser, device):
    model.train()
    losses = AverageMeter()

    # For each batch
    with tqdm.tqdm(total=len(train_loader)) as progress:
        for i, (data, labels) in enumerate(train_loader):
            data = data.requires_grad_()
            labels = labels.type(torch.FloatTensor).flatten()

            # Send the data to the device (GPU hopefully!)
            data = data.to(device)
            labels = labels.to(device)

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward,
            # dont want to cummulate gradients
            optimiser.zero_grad()
            outputs = model(data)
            print(outputs)

            loss = criterion(outputs.flatten(), labels)
            losses.update(loss.data, len(labels))

            loss.backward()
            optimiser.step()

            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            #       epoch, i, len(train_loader), 100. * i / len(train_loader), loss=losses))

            progress.update(1)

    plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg.cpu())


def val_epoch(model, val_loader, epoch, criterion, acc_func, device, verbose=False, get_prop=False):
    losses = AverageMeter()
    model.eval()

    correct = 0
    total = 0

    with tqdm.tqdm(total=len(val_loader)) as progress:
        for i, (data, labels) in enumerate(val_loader):
            data = data.requires_grad_()
            labels = labels.type(torch.FloatTensor).flatten()

            # Send to device (GPU hopefully!)
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs.flatten(), labels)
            losses.update(loss.data.cpu().numpy(), labels.size(0))

            # convert output probabilities to predicted class
            predicted = torch.round(outputs.flatten())

            print('labels:', labels)
            print('pred:', predicted)

            if verbose:
                print("================== Labels\n\n")
                print(labels)
                print("\n\n===================== Predicted\n\n")
                print(predicted)
                print("\n\n")

            # We need to create the array on the first batch, append to it afterwards
            if i == 0:
                out = predicted.data.cpu().numpy()
                label = labels.cpu().numpy()
            else:
                out = np.concatenate((out, predicted.data.cpu().numpy()), axis=0)
                label = np.concatenate((label, labels.cpu().numpy()), axis=0)

            for j in range(len(labels)):
                if predicted[j] == labels[j]:
                    correct += 1

            total += labels.size(0)

    acc = (correct / total) * 100

    print('Validation set: Average loss: {:.4f}\tAccuracy {acc}'.format(losses.avg, acc=acc))


    plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
    plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)

    # Return acc
    return acc

def train_conv(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("======================= Using device:", device)

    best = 0
    criterion = torch.nn.BCELoss()
    epochs = args.epochs
    batch_size = config['batch_size']

    net = Conv()
    #net = net.double()

    opt = torch.optim.Adam(net.parameters(), lr=config['lr'], betas=(0.9, 0.999))

    net = net.to(device)

    # Randomly create training and validation datasets (for now)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    indices = list(range(len(dataset)))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # If we want to fill NaNs with means, we need to do the training and testing sets separately

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    for epoch in range(epochs):
        train_epoch(net, train_loader, epoch, criterion, opt, device=device)

        acc = val_epoch(net, val_loader, epoch, criterion, criterion, device=device, verbose=False)

        best = max(acc, best)
        print('** Validation: %f (best) - %f (current)' % (best, acc))
    
        if args.save is not None and best == acc:
            torch.save(net, args.save)

class TrainConv(tune.Trainable):
    def _setup(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("======================= Using device:", self.device)

        self.best = 0
        self.criterion = torch.nn.BCELoss()
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']

        self.net = Conv()
        # net = net.double()

        self.opt = torch.optim.Adam(self.net.parameters(), lr=config['lr'])

        self.net = self.net.to(self.device)

        # Randomly create training and validation datasets (for now)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        indices = list(range(len(dataset)))

        np.random.shuffle(indices)
        train_indices, val_indices = indices[:train_size], indices[train_size:]

        # If we want to fill NaNs with means, we need to do the training and testing sets separately

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
        self.val_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=val_sampler)

    def _train(self):
        for epoch in range(self.epochs):
            train_epoch(self.net, self.train_loader, epoch, self.criterion, self.opt, device=self.device)

            acc = val_epoch(self.net, self.val_loader, epoch, self.criterion, self.criterion, device=self.device,
                            verbose=False)

            self.best = max(acc, self.best)
            print('** Validation: %f (best) - %f (current)' % (self.best, acc))

        return {'mean_acc': self.best}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.net.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.net.load_state_dict(torch.load(checkpoint_path))


if __name__ == '__main__':
    if args.opt:
        ray.init(webui_host="0.0.0.0", num_gpus=1)

        space = {"lr": tune.grid_search([0.001, 0.01, 0.0001]),
                 "epochs": tune.grid_search([5, 10, 15, 20]),
                 "batch_size": tune.grid_search([16, 32, 64, 128])
                 }

        analysis = tune.run(
            TrainConv,
            config=space,
            resources_per_trial={"cpu": 1, "gpu": 0.2},
            name=args.name,
            checkpoint_at_end=True,
            checkpoint_freq=3
        )

        print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))
    else:
        train_conv({'batch_size': args.batch_size, 'lr': args.lr})
