import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle

import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from XrayDataset import XrayDataset
from captum.attr import GradientShap

import sys

class ShapDatasetFly(Dataset):
    def __init__(self, normal_path, adv_path, collate_path, model_path, normal_only=False, large_normal=False):
        self.normal_only = normal_only
        self.model = torch.load(model_path)
        self.normal_dataset = XrayDataset(normal_path, collate_path, 'Cardiomegaly', 1, normalise=False, only=None)
        self.adv_dataset = XrayDataset(adv_path, collate_path, 'Cardiomegaly', 1, normalise=False, adv=adv_path)

        self.num_normal = len(self.adv_dataset)

        if large_normal:
            # For training anomaly detection AEs
            self.num_normal = self.num_normal * 5

        self.labels = [0 for i in range(self.num_normal)]

        if not self.normal_only:
            self.labels += [1 for j in range(len(self.adv_dataset))]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

        self.shap = GradientShap(self.model)

        self.baseline = torch.randn(1, 3, 224, 224, requires_grad=True, device=device)

    def calculate_shap_values_sample(self, inputs, targets, shap, plot=False, baseline=None):
        inputs = inputs[None, :, :, :]
        inputs = inputs.to(self.device)

        targets = torch.from_numpy(np.array([targets]))
        targets = targets.to(self.device)

        if baseline is None:
            shap_vals = shap.attribute(inputs, target=targets)
        else:
            shap_vals = shap.attribute(inputs, baselines=baseline, target=targets)
        shap_vals = shap_vals.cpu().detach().numpy()

        return shap_vals

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        if idx < self.num_normal:
            normal_sample, normal_label = self.normal_dataset[idx]
            shap = self.calculate_shap_values_sample(normal_sample, normal_label, self.shap, baseline=self.baseline)
        else:
            adv_sample, adv_label = self.adv_dataset[idx % self.num_normal]
            shap = self.calculate_shap_values_sample(adv_sample, adv_label, self.shap, baseline=self.baseline)

        return shap.squeeze(), label

class ShapDatasetDict(Dataset):
    def __init__(self, normal_file, adv_file, max_num_features):
        with open(normal_file, 'rb') as f:
            self.normal = pickle.load(f)

        with open(adv_file, 'rb') as f:
            self.adversarial = pickle.load(f)

        # ID at which adversarial examples begin
        cutoff = len(self.normal)

        # Create labels array based on above index
        self.labels = [0 for i in range(cutoff)] + [1 for i in range(cutoff,
                                                                     (len(self.normal) + len(self.adversarial)))]

        self.df = self.normal + self.adversarial

        self.max_num_features = max_num_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        values = self.df[item]

        to_return = [0 for i in range(self.max_num_features)]
        for i in range(self.max_num_features):
            if i in values:
                to_return[i] = values[i]

        return to_return, self.labels[item]

class ShapDatasetLoader(Dataset):
    def __init__(self, csv_file):
        # Get just the labels from the CSV file
        self.labels = pd.read_csv(csv_file, usecols=['label'], dtype={'label': np.bool})
        print('loaded labels')

        self.path = csv_file

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        item += 1

        # Get just that row from the csv
        with open(self.path, 'r') as f:
            reader = csv.reader(f)

            # Start at 1 so we don't look at the header
            for i in range(1, item):
                next(reader)

            row = next(reader)

        return row, self.labels.iloc[item].to_numpy()

class ShapDatasetChunked(Dataset):
    def __init__(self, single_file):
        super(ShapDatasetChunked, self).__init__()

        # Read the first line of the CSV file to get the columns
        self.columns = pd.read_csv(single_file, nrows=1).columns.tolist()

        self.chunks = pd.read_csv(single_file, chunksize=1000)

        processed_chunks = []
        i = 1
        for chunk in self.chunks:
            processed_chunk = self.__process_chunk(chunk)

            processed_chunks.append(processed_chunk)

            print('=================== Processed chunk', i)
            i+= 1

        self.df = pd.concat(processed_chunks)

        self.labels = self.df['label']
        del self.df['label']

        self.num_columns = len(self.df.columns)

    def __process_chunk(self, chunk):
        # Change all column types to use float16 instead of float64 (except last column which is label)
        chunk[self.columns[:-1]] = chunk[self.columns[:-1]].astype('float16')
        chunk['label'] = chunk['label'].astype('bool')

        return chunk

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        data = self.df.iloc[item]

        data = data.to_numpy()

        return torch.from_numpy(data), self.labels.iloc[item].to_numpy()


class ShapDatasetTopAdv(Dataset):
    def __init__(self, adversarial_explanations, label=1, collated_label=False, transform=False):
        super(ShapDatasetTopAdv, self).__init__()

        self.transform = transform

        with open(adversarial_explanations, 'rb') as f:
            self.adversarial = pickle.load(f)
            self.adversarial = list(np.asarray(self.adversarial))

        if collated_label:
            l1 = self.adversarial[0]
            l2 = self.adversarial[l1 + 1]

            self.adversarial = np.delete(self.adversarial, 0)
            self.adversarial = np.delete(self.adversarial, l1)
            self.adversarial = list(np.asarray(self.adversarial))

            labels = [[1] for i in range(l1)]
            labels += [[0] for i in range(l2)]
            self.labels = pd.DataFrame.from_records(labels)

        # We have an issue where, for the MIMIC dataset, some patients had multiple visits
        # This means some records will contain more than the specified number of features
        # Here, we just see what the largest number of features is
        num_features = max([len(l) for l in self.adversarial])

        columns = [str(i) for i in range(num_features)]
        self.df = pd.DataFrame(self.adversarial, columns=columns)

        # We just want one big array with corresponding labels
        # 0 = normal explanations
        # 1 = adversarial explanations
        if not collated_label:
            self.labels = pd.DataFrame.from_records([[label] for i in range(len(self.adversarial))])

        # Fill in NaNs with 0 (in this context, stands to reason if a feature isn't present it had 0 impact on outcome)
        self.df = self.df.fillna(0)

        self.num_columns = len(self.df.columns)

        """types = set()
        for index, row in self.df.iterrows():
            types.add(type(row[0]))
            if isinstance(row[0], np.ndarray):
                print('test')
                print(len(row[0].flatten()))
                self.df.iloc[index] = row[0].flatten()

        print(types)"""

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        data = self.df.iloc[item]

        data = data.to_numpy()

        return torch.from_numpy(data), self.labels.iloc[item].to_numpy()

class ShapDatasetTop(Dataset):
    def __init__(self, normal_explanations, adversarial_explanations, normal_only=False, normalise=False):
        super(ShapDatasetTop, self).__init__()
        self.normalise = normalise
       
        # Load the pickled lists
        with open(normal_explanations, 'rb') as f:
            self.normal = pickle.load(f)
            self.normal = list(np.asarray(self.normal))

        with open(adversarial_explanations, 'rb') as f:
            self.adversarial = pickle.load(f)
            self.adversarial = list(np.asarray(self.adversarial))

        # We have an issue where, for the MIMIC dataset, some patients had multiple visits
        # This means some records will contain more than the specified number of features
        # Here, we just see what the largest number of features is
        if normal_only:
            num_features = max([len(l) for l in self.normal])
        else:
            num_features = max([len(l) for l in self.normal] + [len(l) for l in self.adversarial])

        columns = [str(i) for i in range(num_features)]

        if normal_only:
            self.df = pd.DataFrame(self.normal, columns=columns)
        else:
            self.df = pd.DataFrame(self.normal + self.adversarial, columns=columns)

        # We just want one big array with corresponding labels
        # 0 = normal explanations
        # 1 = adversarial explanations
        if normal_only:
            self.labels = pd.DataFrame.from_records([[0] for i in range(len(self.normal))])
        else:
            self.labels = pd.DataFrame.from_records([[0] for i in range(len(self.normal))] +
                                                    [[1] for i in range(len(self.adversarial))])

        # Fill in NaNs with 0 (in this context, stands to reason if a feature isn't present it had 0 impact on outcome)
        self.df = self.df.fillna(0)

        self.num_columns = len(self.df.columns)

        """types = set()
        for index, row in self.df.iterrows():
            types.add(type(row[0]))
            if isinstance(row[0], np.ndarray):
                print('test')
                print(len(row[0].flatten()))
                self.df.iloc[index] = row[0].flatten()

        print(types)"""

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        data = self.df.iloc[item]

        data = data.to_numpy()

        data = torch.from_numpy(data)

        if self.normalise:
            if data.sum().item() != 0:
                data = (data - data.mean()) / data.std()

        return data, self.labels.iloc[item].to_numpy()


class ShapDataset(Dataset):
    def __init__(self, normal_explanations, adversarial_explanations, csv=False, single_file=None):
        super(ShapDataset, self).__init__()

        if csv:
            if single_file is not None:
                self.df = pd.read_csv(single_file, low_memory=True)
            else:
                # We'll read the CSV file in chunks so we don't run out of memory
                # Use small chunksize as we're liited by number of columns, not rows
                #df_normal_chunks = pd.read_csv(normal_explanations, chunksize=100)
                #df_adv_chunks = pd.read_csv(adversarial_explanations, chunksize=100)

                #self.df = pd.concat([chunk for chunk in df_normal_chunks] + [chunk for chunk in df_adv_chunks])
                self.df = pd.concat([pd.read_csv(normal_explanations, low_memory=True),
                                     pd.read_csv(adversarial_explanations, low_memory=True)])
        else:
            # Load the pickled lists
            with open(normal_explanations, 'rb') as f:
                self.normal = pickle.load(f)

            with open(adversarial_explanations, 'rb') as f:
                self.adversarial = pickle.load(f)

            # Initialise the dataframe - we need to tell it what our columns are
            # To find the number of columns, find the longest sublist in normal/adversarial values
            max_num_features = 0
            for orig_sample in self.normal:
                count = 0
                for l in orig_sample:
                    for k in l:
                        for v in k:
                            count += 1
                if count > max_num_features:
                    max_num_features = count

            for orig_sample in self.adversarial:
                count = 0
                for l in orig_sample:
                    for k in l:
                        for v in k:
                            count += 1
                if count > max_num_features:
                    max_num_features = count

            columns = [str(i) for i in range(max_num_features)]
            self.df = pd.DataFrame(columns=columns)

            # We want every sample to be a single list of values
            for orig_sample in self.normal:
                sample = []
                for l in orig_sample:
                    for k in l:
                        for v in k:
                            sample.append(v)

                # Pad the sample as needed
                while len(sample) < max_num_features:
                    sample.append(np.nan)

                self.df.append([pd.DataFrame([sample], columns=columns)], ignore_index=True)

            for orig_sample in self.adversarial:
                sample = []
                for l in orig_sample:
                    for k in l:
                        for v in k:
                            sample.append(v)

                while len(sample) < max_num_features:
                    sample.append(np.nan)

                self.df.append([pd.DataFrame([sample], columns=columns)], ignore_index=True)

        # We just want one big array with corresponding labels
        # 0 = normal explanations
        # 1 = adversarial explanations
        print("Labels")
        if single_file:
            self.labels = self.df['label']

            del self.df['label']
        else:
            self.labels = pd.DataFrame.from_records([[0] for i in range(len(self.normal))] +
                                                    [[1] for i in range(len(self.adversarial))])

        print(self.df)

        # Fill in NaNs with 0 (in this context, stands to reason if a feature isn't present it had 0 impact on outcome)
        self.df = self.df.fillna(0)

        self.num_columns = len(self.df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        data = self.df.iloc[item]

        data = data.to_numpy()

        return torch.from_numpy(data), self.labels.iloc[item].to_numpy()

if __name__ == '__main__':
    # Test the ShapDataset class actually works
    """print('================ Loading Henan Dataset')
    dataset = ShapDatasetTop('./henan-shap/normal_shap_important.pkl', './henan-shap/adv_shap_important.pkl')

    print('================ Loading MIMIC Dataset')
    dataset = ShapDatasetTop('./normal_shap_important.pkl', './adv_shap_important.pkl')

    print('================ Loading Combined Dataset')
    dataset = ShapDatasetTop('./all-top/all_normal_shap.pkl', './all-top/all_adv_shap.pkl')

    print('================ Loading Combined Flat Dataset')
    dataset = ShapDatasetTop('./all-top-cxr-flatten/all_normal_shap.pkl', './all-top-cxr-flatten/all_adv_shap.pkl')"""

    """# Take a look at our large adv. dataset
    print('================ Loading large MIMIC dataset')
    dataset = ShapDatasetTop('mimiciii-shap/all_normal_shap.pkl', 'mimiciii-shap/all_adv_shap.pkl', normal_only=False)

    adv_samples = pd.DataFrame(dataset.adversarial)
    print(adv_samples)"""

    dataset = ShapDatasetFly('/media/hdd/google/', '/media/hdd/mimic-cxr-adv/', '../mimic-cxr/collated_labels.csv',
                             '../mimic-cxr/best_model.pth')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(next(iter(dataloader)))