from sklearn.svm import OneClassSVM
from sklearn import preprocessing

import pandas as pd
import numpy as np

from utils import split_dataset

import argparse
import pickle

arg_parser = argparse.ArgumentParser(description='Train one-class SVM to classify SHAP vals')

arg_parser.add_argument('--save', '-s', help='Patht to save SVM to', type=str, default='./one_class_shap.pkl')
arg_parser.add_argument('--test_split', help='Percentage of dataset to be used for testing', type=float, default=0.2)

arg_parser.add_argument('normal_path', help='Path to normal SHAP vals to classify', type=str)
arg_parser.add_argument('adv_path', help='Path to adversarial SHAP vals to classify', type=str)

args = arg_parser.parse_args()

print('======================= Loading datasets')
# Load dataset
with open(args.normal_path, 'rb') as f:
    normal_shap = pickle.load(f)

with open(args.adv_path, 'rb') as f:
    adv_shap = pickle.load(f)

# Get a train and test set for each
train_normal, test_normal = split_dataset(normal_shap, args.test_split)
train_adv, test_adv = split_dataset(adv_shap, args.test_split)

# Make sure we have the correct datatype
train_normal = train_normal.fillna(0)
train_normal = train_normal.apply(pd.to_numeric)
train_normal = preprocessing.scale(train_normal)

# Stick normal and adv test sets together
test_df = pd.concat([test_normal, test_adv])

# Create labels, normal is 1 and outliers are -1
test_labels = [1 for i in range(len(test_normal))] + [-1 for j in range(len(test_adv))]
test_labels = pd.DataFrame(test_labels)

print('======================= Training model')
model = OneClassSVM()
model.fit(train_normal)

# Make sure test dataset is correct format
test_df = test_df.fillna(0)
test_df = test_df.apply(pd.to_numeric)
test_df = preprocessing.scale(test_df)

preds = model.predict(test_df)

print('======================= Finished training')


total = 0
correct = 0
for i in range(len(preds)):
    total += 1
    correct += 1 if preds[i] == test_labels.iloc[i].values[0] else 0

print("Acc: {}".format(correct/total))

if args.save is not None:
    with open(args.save, 'wb') as f:
        pickle.dump(model, f)
