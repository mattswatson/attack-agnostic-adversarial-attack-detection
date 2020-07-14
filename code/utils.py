import numpy as np
from scipy.sparse import coo_matrix

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import pandas as pd

from visdom import Visdom

from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Visdom Plotting
class VisdomLinePlotter(object):
    def __init__(self, env_name='main'):
        self.vis = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y, xlabel='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.vis.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')

    def plot_matplotlib(self, plot_name, plt):
        self.plots[plot_name] = self.vis.matplot(plt,env=self.env)

    def plot_text(self, text, title='Text'):
        self.vis.text(text, env=self.env, opts=dict(title=title))


""" Custom Dataset """


class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features, reverse):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
            num_features (int): number of total features available
            reverse (bool): If true, reverse the order of sequence (for RETAIN)
        """

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.seqs = []
        # self.labels = []

        for seq, label in zip(seqs, labels):

            if reverse:
                sequence = list(reversed(seq))
            else:
                sequence = seq

            row = []
            col = []
            val = []
            for i, visit in enumerate(sequence):
                for code in visit:
                    if code < num_features:
                        row.append(i)
                        col.append(code)
                        val.append(1.0)

            self.seqs.append(coo_matrix((np.array(val, dtype=np.float32), (np.array(row), np.array(col))), shape=(len(sequence), num_features)))
        self.labels = labels

        # We don't have feature names for this dataset, so just number them
        self.feature_names = [i for i in range(num_features)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]


""" Custom collate_fn for DataLoader"""


def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
    where N is minibatch size, seq is a SparseFloatTensor, and label is a LongTensor

    :returns
        seqs
        labels
        lengths
        indices
    """
    batch_seq, batch_label = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = max(seq_lengths)

    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_seqs = []
    sorted_labels = []

    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded = np.concatenate(
                (batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
        else:
            padded = batch_seq[i].toarray()

        sorted_padded_seqs.append(padded)
        sorted_labels.append(batch_label[i])

    seq_tensor = np.stack(sorted_padded_seqs, axis=0)
    label_tensor = torch.LongTensor(sorted_labels)

    return torch.from_numpy(seq_tensor), label_tensor, list(sorted_lengths), list(sorted_indices)


def batch_patient_tensor_to_list(batch_tensor, lengths, reverse):

    batch_size, max_len, num_features = batch_tensor.size()
    patients_list = []

    for i in range(batch_size):
        patient = []
        for j in range(lengths[i]):
            codes = torch.nonzero(batch_tensor[i][j])
            if codes.is_cuda:
                codes = codes.cpu()
            patient.append(sorted(codes.numpy().flatten().tolist()))

        if reverse:
            patients_list.append(list(reversed(patient)))
        else:
            patients_list.append(patient)

    return patients_list


def dicom_to_rgb(dicom_image):
    img = dicom_image.pixel_array
    image_2d = img.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    normal_img = Image.fromarray(image_2d_scaled).convert('F')

    # Make it an RGB image
    normal_img = normal_img.convert('RGB')

    return normal_img


def split_dataset(dataset, test_split):
    indices = list(range(len(dataset)))
    split = int(test_split * len(dataset))
    train_indices, test_indices = indices[split:], indices[:split]

    train, test = [], []
    for i in train_indices:
        train.append(dataset[i])

    for i in test_indices:
        test.append(dataset[i])

    return pd.DataFrame(train), pd.DataFrame(test)


class LogisticRegression(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogisticRegression, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        return F.sigmoid(x)


class ReconErrorDataset(Dataset):
    def __init__(self, df, labels):
        super(ReconErrorDataset, self).__init__()

        self.df = df
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.df.iloc[idx]
        labels = self.labels.iloc[idx]

        data = torch.from_numpy(data.to_numpy())

        #data = (data - data.min()) / (data - data.max())

        return data, labels.to_numpy()



def train_logistic_regression_epoch(model, loss, opt, dataloader, plotter, device, epoch):
    losses = AverageMeter()
    model = model.train()

    for inputs, labels in dataloader:
        inputs = inputs.requires_grad_()
        inputs = inputs.to(device)
        labels = labels.type(torch.DoubleTensor).flatten()
        labels = labels.to(device)

        opt.zero_grad()
        outputs = model(inputs)

        loss_obj = loss(outputs.flatten(), labels)
        losses.update(loss_obj.data.cpu().numpy(), len(labels))

        loss_obj.backward()
        opt.step()

    print('Train epoch {}: Loss: {}'.format(epoch, losses.avg))

    plotter.plot('loss', 'train', 'Class Loss (LR)', epoch, losses.avg)


def test_logistic_regression_epoch(model, dataloader, loss, plotter, device, epoch):
    losses = AverageMeter()
    model = model.eval()

    with torch.no_grad():
        normal_total = 0
        normal_correct = 0

        adv_total = 0
        adv_correct = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.type(torch.DoubleTensor).flatten()
            labels = labels.to(device)

            outputs = model(inputs)

            loss_obj = loss(outputs.flatten(), labels)
            losses.update(loss_obj.data.cpu().numpy(), len(labels))

            pred = torch.round(outputs.flatten())

            for i in range(len(labels)):
                if labels[i] == 0:
                    normal_total += 1

                    if labels[i] == pred[i]:
                        normal_correct += 1
                else:
                    adv_total += 1

                    if labels[i] == pred[i]:
                        adv_correct += 1

        accuracy = 100 * ((normal_correct + adv_correct) / (normal_total + adv_total))
        normal_accuracy = 100 * (normal_correct / normal_total)
        adv_accuracy = 100 * (adv_correct / adv_total)

        print('Test epoch {}: Loss: {}'.format(epoch, losses.avg))
        print('Overall Acc: {:.2f} | Normal Acc: {:.2f} | Adv. Acc: {:.2f}'.format(accuracy, normal_accuracy,
                                                                                   adv_accuracy))

        plotter.plot('loss', 'val', 'Class Loss (LR)', epoch, losses.avg)
        plotter.plot('acc', 'overall', 'Class Accuracy (LR)', epoch, accuracy)
        plotter.plot('acc', 'normal', 'Class Accuracy (LR)', epoch, normal_accuracy)
        plotter.plot('acc', 'adversarial', 'Class Accuracy (LR)', epoch, adv_accuracy)

    return accuracy