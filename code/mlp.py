import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ray import tune
import ray

import visdom
from ShapDataset import ShapDatasetTop

import argparse
from utils import AverageMeter, VisdomLinePlotter

import numpy as np

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('normal_path', help='Location to saved normal SHAP values', type=str)
    arg_parser.add_argument('adversarial_path', help='Location to saved adversarial SHAP values', type=str)

    arg_parser.add_argument('--plot', help='Name of visdom plot', type=str, default='adv-mlp')

    arg_parser.add_argument('--optimise', '-o', help='Perform hyperparameter optimisation', action='store_true')

    arg_parser.add_argument('--lr', help='Learning rate during training', type=float, default=0.0001)
    arg_parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=100)
    arg_parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    arg_parser.add_argument('--save', help='Path to save best model to', type=str, default=None)
    arg_parser.add_argument('--hidden_dim', help='Dimension of hidden layer', type=int, default=60)
    arg_parser.add_argument('--name', '-n', help='Name of raytune experiment', type=str, default='train_mlp')

    args = arg_parser.parse_args()

    print('=================== Loading dataset')
    dataset = ShapDatasetTop(args.normal_path, args.adversarial_path)

    global plotter
    plotter = VisdomLinePlotter(args.plot)

class Net(nn.Module):
    def __init__(self, hidden_dim=60):
        super(Net, self).__init__()

        self.l1 = torch.nn.Linear(100, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.l1(x)
        x= self.l2(x)

        return F.sigmoid(x)

class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sigmoid=True):
        super(mlp, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

        self.sigmoid = sigmoid

    def forward(self, inputs):
        x = self.relu(self.layer1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        if self.sigmoid:
            return F.sigmoid(x)
        else:
            return x

#net = mlp(dataset.num_columns, args.hidden_dim, 1)

def train_epoch(model, train_loader, epoch, criterion, optimiser, device):
    model.train()
    losses = AverageMeter()

    # For each batch
    for i, (data, labels) in enumerate(train_loader):
        data = data.requires_grad_()
        labels = labels.type(torch.DoubleTensor).flatten()

        # Send the data to the device (GPU hopefully!)
        data = data.to(device)
        labels = labels.to(device)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward,
        # dont want to cummulate gradients
        optimiser.zero_grad()
        outputs = model(data).flatten()

        loss = criterion(outputs, labels)
        losses.update(loss.data, len(labels))

        loss.backward()
        optimiser.step()

        #print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
         #     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
         #      epoch, i, len(train_loader), 100. * i / len(train_loader), loss=losses))

    #plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg.cpu())


def val_epoch(model, val_loader, epoch, criterion, acc_func, device, verbose=False, get_prop=False):
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0


        for i, (data, labels) in enumerate(val_loader):
            data = data.requires_grad_()
            labels = labels.type(torch.DoubleTensor).flatten()

            # Send to device (GPU hopefully!)
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs.flatten(), labels)
            losses.update(loss.data.cpu().numpy(), labels.size(0))

            # convert output probabilities to predicted class
            predicted = torch.round(outputs.flatten())
            print(outputs)
            print(predicted)

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

            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    correct += 1

            total += labels.size(0)

        acc = (correct / total) * 100

        print('Validation set: Average loss: {:.4f}\tAccuracy {acc}'.format(losses.avg, acc=acc))


        #plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
        #plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)

        # Return acc as the validation outcome
        return acc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("======================= Using device:", device)

# Send it to the device (hopefully a GPU!)

def train_mlp(config):
    hidden_dim = config['hidden_dim']
    epochs = config['epochs']
    lr = config['lr']
    batch_size = config['batch_size']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best = 0
    criterion = torch.nn.BCELoss()

    net = mlp(100, hidden_dim, 1)
    net = net.double()

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    net.to(device)

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

        if config['save'] is not None and acc > best:
            print('======================= Saving model to {}'.format(args.save))
            torch.save(net, args.save)

        best = max(acc, best)
        print('** Validation: %f (best) - %f (current)' % (best, acc))

        try:
            tune.track.log(mean_accuracy=acc)
        except AttributeError:
            continue

if __name__ == '__main__':
    ray.init(webui_host="0.0.0.0", num_gpus=1)

    if args.optimise:
        space = {"lr": tune.grid_search([0.001, 0.01, 0.0001]),
                 "hidden_dim": tune.grid_search([20, 40, 60, 80, 100, 120, 140, 160, 180, 200]),
                 "epochs": args.epochs,
                 "batch_size": args.batch_size,
                 "save": None}

        analysis = tune.run(
            train_mlp,
            config=space,
            resources_per_trial={"cpu": 1, "gpu": 0.2},
            name=args.name
        )

        dfs = analysis.trial_dataframes
        print(dfs)

        print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))
    else:
        space = {"lr": args.lr,
                 "hidden_dim": args.hidden_dim,
                 "epochs": args.epochs,
                 "batch_size": args.batch_size,
                 "save": True}

        train_mlp(space)
