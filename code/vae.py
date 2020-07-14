import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import visdom
from ShapDataset import ShapDatasetTop

import argparse
from utils import AverageMeter, VisdomLinePlotter

import numpy as np
import pandas as pd

import pickle
import random
import math

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('normal_path', help='Location to saved normal SHAP values', type=str)
arg_parser.add_argument('adversarial_path', help='Location to saved adversarial SHAP values', type=str)

arg_parser.add_argument('--plot', help='Name of visdom plot', type=str, default='adv-vae')

arg_parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
arg_parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')

arg_parser.add_argument('--save', help='Location to save model to', type=str, default=None)
arg_parser.add_argument('--load', help='Location to load model from', type=str, default=None)

arg_parser.add_argument('--verbose', help='Verbose output', action='store_true')

args = arg_parser.parse_args()

print('=================== Loading dataset')
dataset = ShapDatasetTop(args.normal_path, args.adversarial_path, normal_only=True, normalise=True)
dataset_all = ShapDatasetTop(args.normal_path, args.adversarial_path, normal_only=False, normalise=True)

# Get adversarial samples only, bit of a cheat way
adv_samples = pd.DataFrame(dataset_all.adversarial)
adv_samples = adv_samples.fillna(0)

normal_samples = pd.DataFrame(dataset.normal)
normal_samples = normal_samples.fillna(0)

global plotter
plotter = VisdomLinePlotter(args.plot)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.l1 = nn.Linear(100, 50)
        self.l21 = nn.Linear(50, 5)
        self.l22 = nn.Linear(50, 5)
        self.l3 = nn.Linear(5, 50)
        self.l4 = nn.Linear(50, 100)

    def encode(self, x):
        h1 = F.relu(self.l1(x))
        return self.l21(h1), self.l22(h1)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.l3(z))
        return torch.sigmoid(self.l4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction loss: MSE + KL divergence
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


model = VAE()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("======================= Using device:", device)

model = model.to(device)

opt = optim.Adam(model.parameters(), lr=0.001)

print("====================== Loading data")

# Randomly create training and validation datasets (for now)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
indices = list(range(len(dataset)))

np.random.shuffle(indices)
train_indices, val_indices = indices[:train_size], indices[train_size:]

# If we want to fill NaNs with means, we need to do the training and testing sets separately

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

batch_size = args.batch_size
epochs = args.epochs

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

model = model.train()
model = model.double()

# No need to train if we're loading a model

if args.load is None:
    losses = AverageMeter()
    for epoch in range(epochs):
        for data, _ in train_loader:
            data = data.requires_grad_()
            data = data.to(device)

            opt.zero_grad()

            output, mu, logvar = model(data)

            loss_obj = loss_function(output, data, mu, logvar)

            loss_obj.backward()
            opt.step()

            losses.update(loss_obj.item(), len(data))

        plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)
        print('Epoch {} loss: {}'.format(epoch, losses.avg))

    if args.save is not None:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
else:
    with open(args.load, 'rb') as f:
        model = torch.load(f)

model = model.eval()

# Go through our adversarial samples and get error
total_err = 0
for i in range(len(adv_samples)):
    data = adv_samples.iloc[i]
    data = data.to_numpy()
    data = torch.from_numpy(data)
    data = data.to(device)

    if data.sum().item() != 0:
        data = (data - data.mean()) / data.std()

    output, _, _  = model(data)
    error = torch.sum((data - output) * (data - output))

    total_err += error

    print('Error for adv. item {} was {}'.format(i, error))

avg_adv_err = total_err / len(adv_samples)

# Do the same but for the normal data
total_err = 0
max_err = 0
errors = []
for i in range(len(normal_samples)):
    data = normal_samples.iloc[i]
    data = data.to_numpy()
    data = torch.from_numpy(data)
    data = data.to(device)

    if data.sum().item() != 0:
        data = (data - data.mean()) / data.std()

    output, _, _ = model(data)
    error = torch.sum((data - output) * (data - output))

    if error > max_err:
        max_err = error

    total_err += error

    errors.append(error)

    print('Error for normal item {} was {}'.format(i, error))

avg_normal_err = total_err / len(normal_samples)

std_dev_sum = [(x - avg_normal_err) * (x - avg_normal_err) for x in errors][0]
std_dev = math.sqrt(std_dev_sum / len(normal_samples))

print('Normal error was {}\nAdv. error was {}'.format(avg_normal_err, avg_adv_err))
print('Std. dev. of normal error was {}'.format(std_dev))

dataloader = DataLoader(dataset_all, batch_size=1)

# Try going through a test set and seeing if we can find the adv. samples
adv_correct = 0
normal_correct = 0

adv_total = 0
normal_total = 0
for data, labels in dataloader:
    data = data.to(device)
    labels = labels.to(device)

    output, _, _ = model(data)

    error = torch.sum((data[0] - output[0]) * (data[0] - output[0]))

    if args.verbose:
        print('================ original')
        print(data)
        print('================ reconstructed')
        print(output)

    if error > (avg_normal_err + std_dev):
        adv = True
    else:
        adv = False

    if labels[0] == 1:
        if adv:
            adv_correct += 1

        adv_total += 1
    else:
        if not adv:
            normal_correct += 1

        normal_total += 1

print('Acc. on normal data: {}\nAcc. on adv. data: {}'.format(normal_correct / normal_total, adv_correct / adv_total))

"""max_error = 0
# Go through test data, find largest error
for data, labels in test_loader:
    data = data.to(device)
    labels = labels.to(device)

    output = model(data)

    for j in range(len(output)):
        error = torch.sum((data[j] - output[j]) * (data[j] - output[j]))

        if error > max_error:
            max_error = error
            print('Found max error at index {}'.format(j))"""