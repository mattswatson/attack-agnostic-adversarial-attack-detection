import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import pydicom
from PIL import Image

import os

import matplotlib.pyplot as plt

from utils import dicom_to_rgb

# Dataset to collect both normal and adversarial examples
class XrayAdvDataset(Dataset):
    def __init__(self, normal_path, adv_path, record_list, transform=False, normalise=False, return_normal=True,
                 return_dicom_id=False, collated_clusters_path=None, shap_cluster=None, feature_cluster=None,
                 return_label=None):
        super(XrayAdvDataset, self).__init__()

        self.normal_path = normal_path
        self.adv_path = adv_path
        self.record_list = pd.read_csv(record_list)
        self.transform = transform
        self.normalise = normalise
        self.return_normal = return_normal
        self.return_dicom_id = return_dicom_id
        self.return_label = return_label

        if self.return_label is not None:
            self.collated_labels = pd.read_csv(self.return_label)

        self.shap_cluster = shap_cluster
        self.feature_cluster = feature_cluster

        # Get a list of all adv. images
        self.all_adv = os.listdir(self.adv_path)
        self.paths = []

        # Load cluster labels
        if collated_clusters_path is not None:
            collated_clusters = pd.read_csv(collated_clusters_path)

        # To make things quicker, get all normal paths now
        for file in self.all_adv:
            dicom_id = file[:-4]
            study = self.record_list[self.record_list['dicom_id'] == dicom_id]

            # If we're looking at a certain cluster, we only need those images
            if self.feature_cluster is not None:
                # Get the cluster the image belongs to
                cluster = collated_clusters[collated_clusters['dicom_id'] == dicom_id]['feature_cluster'].iloc[0]

                if cluster != self.feature_cluster:
                    continue

            if self.shap_cluster is not None:
                # Get the cluster the image belongs to
                cluster = collated_clusters[collated_clusters['dicom_id'] == dicom_id]['shap_cluster'].iloc[0]

                if cluster != self.shap_cluster:
                    continue

            # Don't need the files/ dir
            normal_img_path = os.path.join(self.normal_path, study['path'].values[0][6:])

            adv_image_path = os.path.join(self.adv_path, file)

            self.paths.append({'normal': normal_img_path, 'adv': adv_image_path, 'id': dicom_id})

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        paths = self.paths[idx]

        # Load normal image
        dcm = pydicom.dcmread(paths['normal'])

        # For now, just keep the pixel data
        normal_img = dicom_to_rgb(dcm)

        # Load adv. image
        adv_img = Image.open(paths['adv'])

        # Make sure this is RGB too
        adv_img = adv_img.convert('RGB')

        # Transform the images as required for Densenet
        if self.transform and self.normalise:
            composed = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.transform:
            composed = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        elif self.normalise:
            composed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if self.transform or self.normalise:
            normal_img = composed(normal_img)
            adv_img = composed(adv_img)

        # Get the label if needed
        if self.return_label is not None:
            # Only want relative path hence [18:]
            row = self.collated_labels[self.collated_labels['dcm_path'] == paths['normal'][18:]]
            label = row['Cardiomegaly'].iloc[0]

            # Our adv. label is the opposite of this
            label = int((not label) if not (pd.isna(label) or -1.0) else 1)

        if self.return_normal:
            if self.return_dicom_id:
                return normal_img, adv_img, paths['id']
            else:
                return normal_img, adv_img
        else:
            if self.return_dicom_id:
                return adv_img, paths['id']
            else:
                if self.return_label is not None:
                    return adv_img, label
                else:
                    return adv_img

# This is a simple dataloader, it only allows for binary classification
class XrayDataset(Dataset):
    def __init__(self, data_path, collated_labels_path, label, mention, normalise=True, include_filenames=False,
                 adv=None, jpg=False, only=None, use_not_mentioned=True, return_idx=False):
        super(XrayDataset, self).__init__()

        self.normalise = normalise
        self.include_filenames = include_filenames
        self.adv = adv
        self.jpg = jpg
        self.return_idx = return_idx

        self.collated_labels = pd.read_csv(collated_labels_path)
        self.data_path = data_path

        self.img_paths = self.__get_images_with_label(label, mention)

        # We need to get the others too, and then label them correctly
        # Assume if not mentioned, it isn't present
        # Label positive images first
        rows_paths = []
        rows_labels = []
        adv_images = [x[:-4] for x in os.listdir(self.adv)]

        for path in self.img_paths:
            filename = os.path.basename(path)[:-4]

            # Check to see we have an adversary of this image
            if only is None or only == mention:
                if self.adv is None or filename in adv_images:
                    rows_paths.append([path])
                    rows_labels.append([1])

        # Now get the rest of the paths
        if use_not_mentioned:
            other_rows = self.collated_labels[self.collated_labels[label] != mention]
        else:
            other_mention = 1.0 - mention
            other_rows = self.collated_labels[self.collated_labels[label] == other_mention]

        other_paths = other_rows['dcm_path']
        for path in other_paths:
            filename = os.path.basename(path)[:-4]

            # Check to see we have an adversary of this image
            if only is None or only != mention:
                if self.adv is None or filename in adv_images:
                    rows_paths.append([path])
                    rows_labels.append([0])
        
        # Create the dataframes
        self.df = pd.DataFrame(rows_paths, columns=['path'])
        self.labels = pd.DataFrame(rows_labels, columns=['label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # We need to return the actual images, not the paths
        if type(idx) is list:
            images = []

            for i in idx:
                row = self.df.iloc[i]
                path = os.path.join(self.data_path, row['path'])
                dcm = pydicom.dcmread(path)
                
                img = dcm.pixel_array.astype(np.float16)
                img = torch.Tensor(img)
                images.append(img)

            return images, self.labels[idx]

        row = self.df.iloc[idx]
        path = os.path.join(self.data_path, row['path'])

        if self.adv is not None:
            # We don't have the patient/study folder structure anymore
            # Our adversaries are also jpgs
            filename = os.path.basename(path)[:-4] + '.jpg'
            path = os.path.join(self.adv, filename)

            # As we have jpgs we don't need to do any processing, just the normal transforms
            img = Image.open(path)

            # Densenet needs an RGB image
            img = img.convert('RGB')
        elif self.jpg:
            # If we're using the JPEG version of the images, we don't need to do any preprocessing
            # Replace .dcm extension with .jpg
            path = path[:-4] + '.jpg'

            img = Image.open(path)

            # Densenet needs an RGB image
            img = img.convert('RGB')
        else:
            filename = os.path.basename(path)[:-4]
            dcm = pydicom.dcmread(path)
        
            # For now, just keep the pixel data
            img = dicom_to_rgb(dcm)

            pixels = list(img.getdata())
            width, height = img.size

            # This is just for displaying the image if we ever want to do that
            pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

        # The x-rays are all different dimensions
        # So make them the same size here
        # Normalise to range expected by densenet
        if self.normalise:
            composed = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Foolbox wants the unnormalised images
            composed = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])

        img = composed(img)

        if self.include_filenames:
            return img, self.labels.iloc[idx].item(), filename
        else:
            if self.return_idx:
                return img, self.labels.iloc[idx].item(), idx
            else:
                return img, self.labels.iloc[idx].item()

    # Get all images with desired label
    def __get_images_with_label(self, label, mention):
        # label = type of diagnosis
        # mention = 1, 0, -1, None (=NaN)

        labels = list(self.collated_labels.columns)[2:]

        if label not in labels:
            raise Exception('label should be a proper label generated by CheXpert')

        if mention not in [1, 0, -1, None]:
            raise Exception('mention should be 1, 0, -1 or None')

        rows = self.collated_labels[self.collated_labels[label] == mention]
        imgs = rows['dcm_path']

        # Adversarial images are in one large folder
        if self.adv:
            imgs = [os.path.basename(path) for path in imgs]

        imgs_list = []
        for img in imgs:
            imgs_list.append(img)

        return imgs_list

# This class doesn't work at the minute! Need to consider how best to handle classes with multiple positive labels
# Do we have 13! possible classes? This is obviously far too many
"""class XrayDatasetMulticlass(Dataset):
    def __init__(self, data_path, collated_labels_path, mention=1):
        super(XrayDatasetMulticlass, self).__init__()

        self.collated_labels = pd.read_csv(collated_labels_path)
        self.data_path = data_path

        # We need to get the others too, and then label them correctly
        # Assume if not mentioned, it isn't present
        # Label positive images first
        rows_paths = []
        rows_labels = []
        for path in self.collated_labels['dcm_path']:
            rows_paths.append([path])

            # Get the correct label
            # Actually, what should we do here - some have multiple positive labels!


        # Now get the rest of the paths
        other_rows = self.collated_labels[self.collated_labels[label] != mention]

        other_paths = other_rows['dcm_path']
        for path in other_paths:
            rows_paths.append([path])
            rows_labels.append([0])

        # Create the dataframes
        self.df = pd.DataFrame(rows_paths, columns=['path'])
        self.labels = pd.DataFrame(rows_labels, columns=['label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # We need to return the actual images, not the paths
        if type(idx) is list:
            images = []

            for i in idx:
                row = self.df.iloc[i]
                path = os.path.join(self.data_path, row['path'])
                dcm = pydicom.dcmread(path)

                img = dcm.pixel_array.astype(np.float16)
                img = torch.Tensor(img)
                images.append(img)

            return images, self.labels[idx]

        row = self.df.iloc[idx]
        path = os.path.join(self.data_path, row['path'])

        dcm = pydicom.dcmread(path)

        # For now, just keep the pixel data
        img = dcm.pixel_array.astype(np.float16)
        img = torch.Tensor(img)
        print(img)

        return img, self.labels.iloc[idx]
"""

if __name__ == '__main__':
    """print("=============== Testing XrayDataset class")
    dataset = XrayDataset('/media/hdd/mimic-cxr-jpg/files/', '../mimic-cxr/collated_labels.csv', 'Cardiomegaly', 1,
                          jpg=True)
    
    print('Length of dataset:', len(dataset))

    print('=============== First item in dataset')
    print(dataset[0])"""

    dataset = XrayAdvDataset('/media/hdd/google/', '/media/hdd/mimic-cxr-adv/', '/media/hdd/google/cxr-record-list.csv',
                             transform=True, collated_clusters_path='./collated_clusters.csv', feature_cluster=6,
                             return_label='../mimic-cxr/collated_labels.csv')

    normal, adv = dataset[0]

    plt.subplot(1, 2, 1)
    plt.imshow(normal)

    # Show the adversarial image
    plt.subplot(1, 2, 2)
    plt.imshow(adv)

    plt.show()