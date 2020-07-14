import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import pickle

import argparse

from torch.utils.data import Dataset


class HenanDataset(Dataset):
    def __init__(self, path, label_type='all', discrete=False):
        """
        :param path: Path to the file containing the Henan dataset
        :param label_type: The type of labelling to use. Can be one of the following:
                        - hypertension - Binary classification
                        - diabetes - Binary classification
                        - fatty_liver - Binary classification
                        - all (default) - Classification into 8 classes
        """
        super(HenanDataset, self).__init__()

        # Load the dataset as described in http://pinfish.cs.usm.edu/dnn/
        all_labels = []
        all_features = []

        with open(path, 'r') as f:
            for line in f:
                exam = line.split(',')

                # Last entry is class label in form hypertension,diabetes,fatty_liver
                # Need to remove linebreak as well
                label = exam[-1][:-1]
                labels = [0, 0, 0]

                for i in range(-1, -len(label) - 1, -1):
                    labels[i] = label[i]

                if label_type == 'all':
                    """Give correct label as follows:
                    - 0 = no diagnosis
                    - 1 = fatty liver only
                    - 2 = diabetes only
                    - 3 = fatty liver and diabetes
                    - 4 = hypertension only
                    - 5 = hypertension and fatty liver
                    - 6 = hypertension and diabetes
                    - 7 = hypertension, diabetes and fatty liver
                    """

                    # The above class labels were chosen so we can just take the labels list as in base 2, and convert
                    # it to base 10
                    labels_str = ""
                    for el in labels:
                        labels_str += str(el)

                    final_label = int(labels_str, 2)
                elif label_type == 'hypertension':
                    final_label = labels[0]
                elif label_type == 'diabetes':
                    final_label = labels[1]
                elif label_type == 'fatty_liver':
                    final_label = labels[2]
                else:
                    raise AttributeError("Incorrect label type.")

                all_labels.append(final_label)

                # The rest of the data is features
                all_features.append(exam[:-1])

        self.labels = pd.DataFrame(all_labels)
        self.df = pd.DataFrame(all_features)

        self.num_columns = len(self.df.columns)

        # RETAIN uses medical codes for diagnosis. To try and make our data look like this, we will discretise each
        # column into several different codes. Start our medical codes at 10
        current_code = 10
        if discrete:
            for column in self.df:
                # All of our columns are originally in the range [0, 1]
                # See how many unique values we have to decide how many bins we'll have
                num_unique = self.df[column].nunique()

                # If we have only 2 different values, we have a binary classification
                if num_unique == 2:
                    values = self.df[column].unique()
                    self.df[column] = self.df[column].map({values[0]: current_code, values[1]: current_code + 1})

                    current_code += 2
                else:
                    # Stick the columns into bins, have a smaller amount of bins than actual values
                    num_bins = 0.125 * num_unique

                    # If we have too few bins, just use the number of unique values
                    if num_bins <= 2:
                        num_bins = num_unique

                    # This is a nasty hack needed as pandas uses range (x, y], so values of 0 would never be added
                    bins = np.linspace(-0.00001, 1, num_bins)
                    labels = [current_code + i for i in range(0, len(bins) - 1)]

                    self.df[column] = pd.cut(self.df[column].astype(float), bins=bins, labels=labels)

                    current_code += len(bins)

        # We need the number of codes when training RETAIN. Add the number used for classification too
        self.num_codes = current_code + len(set(all_labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()

        data = self.df.iloc[ids]
        data = data.to_numpy()

        labels = self.labels.iloc[ids].to_numpy()

        # Convert everything to the correct data type
        data = list(map(float, data))
        labels = list(map(int, labels))

        return data, labels


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seqs', '-s', help='Location to save .seqs file', type=str, default='henan/output.seqs')
    arg_parser.add_argument('--labels', '-l', help='Location to save labels file', type=str,
                            default='henan/output.labels')
    arg_parser.add_argument('--test_ratio', '-t', help='Portion of data to be test set', type=float, default=0.2)
    arg_parser.add_argument('--val_ratio', '-v', help='Portion of data to be validation set', type=float, default=0.1)

    args = arg_parser.parse_args()

    print("=============== Loading Data")
    data = HenanDataset('henan-data.txt', label_type='hypertension', discrete=True)
    print("=============== Data Loaded")

    # We want to output it into the format given here: https://github.com/ast0414/lava
    # List of List of Float = output.seqs
    # List of Int (labels) = output.labels
    seqs = []
    labels = []

    print("=============== Formatting Data")
    with tqdm(total=len(data)) as progress:
        for event, label in iter(data):
            # Every event is a separate patient, each patient has only one event
            patient = [event]
            seqs.append(patient)
            labels.append(label[0])

            progress.update(1)

    with open(args.seqs, 'wb') as f:
        pickle.dump(seqs, f)

    with open(args.labels, 'wb') as f:
        pickle.dump(labels, f)

    print("=============== Successfully saved files for LAVA")

    print("=============== Splitting dataset to train, test and validation sets")
    nTest = int(args.test_ratio * len(data))
    nVal = int(args.val_ratio * len(data))

    test, val, train = seqs[:nTest], seqs[nTest:nTest + nVal], seqs[nTest + nVal:]
    test_labels, val_labels, train_labels = labels[:nTest], labels[nTest:nTest + nVal], labels[nTest + nVal:]

    # Save split data
    with open('henan-codes/split/test.seqs', 'wb') as f:
        pickle.dump(test, f)

    with open('henan-codes/split/val.seqs', 'wb') as f:
        pickle.dump(val, f)

    with open('henan-codes/split/train.seqs', 'wb') as f:
        pickle.dump(train, f)

    with open('henan-codes/split/test.labels', 'wb') as f:
        pickle.dump(test_labels, f)

    with open('henan-codes/split/val.labels', 'wb') as f:
        pickle.dump(val_labels, f)

    with open('henan-codes/split/train.labels', 'wb') as f:
        pickle.dump(train_labels, f)

    print("=============== Split data saved")