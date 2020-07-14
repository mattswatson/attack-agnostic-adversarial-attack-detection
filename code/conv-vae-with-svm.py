import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import visdom
from ShapDataset import ShapDatasetTop, ShapDatasetFly

# Ignore warnings from sklearn
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing

import argparse
from utils import AverageMeter, VisdomLinePlotter,  LogisticRegression, train_logistic_regression_epoch, \
    test_logistic_regression_epoch, ReconErrorDataset

import numpy as np
import pandas as pd

import pickle
import random
import math
import tqdm

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('normal_path', help='Location to saved normal SHAP values', type=str)
arg_parser.add_argument('adversarial_path', help='Location to saved adversarial SHAP values', type=str)
arg_parser.add_argument('model', help='Path to model', type=str)
arg_parser.add_argument('collate_path', help='Path to collated_labels.csv', type=str)

arg_parser.add_argument('--plot', help='Name of visdom plot', type=str, default='ae_with_svm')

arg_parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
arg_parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
arg_parser.add_argument('--svm', help='Use SVM on reconstruction error (default: use LR)', action='store_true')

arg_parser.add_argument('--save', help='Location to save model to', type=str, default=None)
arg_parser.add_argument('--load', help='Location to load model from', type=str, default=None)
arg_parser.add_argument('--save_recon_error', help='Location to save reconstruction errors to', type=str, default=None)
arg_parser.add_argument('--save_svm', help='Location to save recon. err. SVM to', type=str, default=None)

arg_parser.add_argument('--n_jobs', help='Number of parallel instances for grid search', type=int, default=None)

args = arg_parser.parse_args()

print('=================== Loading dataset')
#dataset = ShapDatasetTop(args.normal_path, args.adversarial_path, normal_only=False, normalise=True)
#dataset_all = ShapDatasetTop(args.normal_path, args.adversarial_path, normal_only=False, normalise=True)
dataset = ShapDatasetFly(args.normal_path, args.adversarial_path, args.collate_path, args.model, normal_only=True,
                         large_normal=True)
dataset_all = ShapDatasetFly(args.normal_path, args.adversarial_path, args.collate_path, args.model)

global plotter
plotter = VisdomLinePlotter(args.plot)

# https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/april/test-run-neural-anomaly-detection-using-pytorch

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


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(56, 28)
        self.fc2 = nn.Linear(56, 28)
        self.fc3 = nn.Linear(28, 56)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.ReLU()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        esp = esp.to(device)

        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar

# Reconstruction loss: MSE + KL divergence
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


model = ConvVAE()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("======================= Using device:", device)

model = model.to(device)

opt = optim.Adam(model.parameters(), lr=0.01)

print("====================== Loading data")

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
test_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)

model = model.train()

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

print('====================== Calculating final reconstruction loss for all train data')
# Get the normal and adv. samples from our train data
dataloader = DataLoader(dataset_all, batch_size=1)

all_train_recon_errors = []
all_train_labels = []
with tqdm.tqdm(total=len(dataloader)) as progress:
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        output = model(inputs)

        error = torch.mean((inputs[0] - output[0]) * (inputs[0] - output[0]))

        all_train_recon_errors.append(error.data.item())
        all_train_labels.append(labels[0].data.item())

        progress.update(1)

# Make the errors into a pandas dataframe so we can easily feed it into an SVM
# Make the errors into a pandas dataframe so we can easily feed it into an SVM
error_df = pd.DataFrame(all_train_recon_errors)
error_labels_df = pd.DataFrame(all_train_labels)

# Because we have such imabalanced classes, we need more adv. samples so that we don't just classify all normal samples
# Correctly and be done with it
# This isn't the best way to do this, but for now we will see if it works

num_normal = len(error_labels_df[error_labels_df[0] == 0])
num_adv = len(error_labels_df[error_labels_df[0] == 1])

# How many times do we have to repeat the adv. set to get around the same number of samples?
repeat = int(num_normal / num_adv)

# Get the adv. samples
locs = error_labels_df.index[error_labels_df[0] == 1].tolist()

adv_samples = error_df.loc[locs]

for i in range(repeat):
    error_df = error_df.append(adv_samples, ignore_index=True)
    error_labels_df = error_labels_df.append(error_labels_df.loc[locs], ignore_index=True)

err_dataset = ReconErrorDataset(error_df, error_labels_df)

if not args.svm:
    print('====================== Training Logistic Regression on reconstruction errors')
    train_size = int(0.8 * len(err_dataset))
    val_size = len(err_dataset) - train_size
    indices = list(range(len(err_dataset)))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    err_train_loader = DataLoader(err_dataset, batch_size=32, sampler=train_sampler)
    err_val_loader = DataLoader(err_dataset, batch_size=1, sampler=val_sampler)

    model = LogisticRegression(1, 1)
    model = model.double()
    model = model.to(device)
    loss = nn.BCELoss()
    opt = optim.SGD(model.parameters(), 0.001)

    best = 0
    for epoch in range(100):
        train_logistic_regression_epoch(model, loss, opt, err_train_loader, plotter, device, epoch)

        acc = test_logistic_regression_epoch(model, err_val_loader, loss, plotter, device, epoch)

        best = max(acc, best)
        print('** Validation: %f (best) - %f (current)' % (best, acc))
else:
    print('====================== Training SVM on reconstruction errors')

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    split = sss.split(error_df, error_labels_df)

    for train_indices, test_indices in split:
        data_train, labels_train = error_df.iloc[train_indices], error_labels_df.iloc[train_indices]
        data_test, labels_test = error_df.iloc[test_indices], error_labels_df.iloc[test_indices]

    data_train = preprocessing.scale(data_train)
    data_test = preprocessing.scale(data_test)

    # Hyperparameters we want to search over
    param_grid = [{'kernel': ['rbf'], 'C': [1.0], 'gamma': ['auto'], 'shrinking': [True]}]

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=args.n_jobs)
    grid.fit(data_train, labels_train.values.ravel())

    # svm = SVC(kernel='rbf', shrinking=True)
    # svm.fit(data_train, labels_train)
    print('======================  SVM trained')
    print('====================== Best parameters found were:\n')
    print(grid.best_params_)

    if args.save_svm is not None:
        print('====================== Saving SVM to {}'.format(args.save_svm))
        with open(args.save_svm, 'wb') as f:
            pickle.dump(grid, f)
        print('====================== SVM saved')

    # Try going through a test set and seeing if we can find the adv. samples
    normal_correct = 0
    adv_correct = 0

    normal_total = 0
    adv_total = 0
    print('====================== Testing SVM')
    with tqdm.tqdm(total=len(data_test)) as progress:
        for i in range(len(data_test)):
            error = data_test[i]
            label = labels_test.iloc[i].values[0]

            pred = grid.predict([error])

            print('Error of sample was {}, label of sample was {}, predicted label was {}'.format(error,
                                                                                                  label,
                                                                                                  pred[0]))

            if pred[0] == 1:
                adv = True
            else:
                adv = False

            if label == 1:
                if adv:
                    adv_correct += 1

                adv_total += 1
            else:
                if not adv:
                    normal_correct += 1

                normal_total += 1

            progress.update(1)

    print('Acc. on normal data: {}\nAcc. on adv. data: {}'.format(normal_correct / normal_total,
                                                                  adv_correct / adv_total))

    preds = grid.predict(data_test)
    conf_matrix = confusion_matrix(labels_test, preds).tolist()
    class_report = classification_report(labels_test, preds)
    results = "{} \n\n {}".format(conf_matrix, class_report)

    print(results)

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