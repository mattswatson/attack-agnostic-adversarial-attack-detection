from visdom import Visdom

from utils import AverageMeter, VisdomLinePlotter

from ShapDataset import ShapDataset, ShapDatasetChunked, ShapDatasetDict, ShapDatasetTop

import argparse

import pickle
import random

global plotter
plotter = VisdomLinePlotter('attack-svm')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.svm import libsvm

import tqdm

import os

arg_parser = argparse.ArgumentParser(description='Use Captum to explain samples from RETAIN')

arg_parser.add_argument('--all', '-a', help='Train all SVM models', action='store_true')
arg_parser.add_argument('--save', '-s', help='Location to save model to', type=str, default=None)
arg_parser.add_argument('--load', '-l', help='Test a pretrained model', type=str, default=None)
arg_parser.add_argument('--load_all_csv', help='Load a single CSV file containing all SHAP values', type=str,
                        default=None)
arg_parser.add_argument('--degree', '-d', help='Degree of polynomial for polynomial SVM', type=int, default=12)
arg_parser.add_argument('--use_dict', help='Use dictionaries to save space', action='store_true')
arg_parser.add_argument('--shrinking', help='Use shriking factor when training SVMs', action='store_true')

arg_parser.add_argument('--max_features', help='Max. number of features in an example', type=int, default=205548)
arg_parser.add_argument('--use_top', help='Use dataset continaing only top N many SHAP values', action='store_true')

arg_parser.add_argument('--save_results', help='Path to save accuracy results to', type=str, default=None)

arg_parser.add_argument('normal_path', help='Location to saved normal SHAP values', type=str)
arg_parser.add_argument('adversarial_path', help='Location to saved adversarial SHAP values', type=str)

args = arg_parser.parse_args()

print("======================= Loading Dataset")
if args.use_top:
    dataset = ShapDatasetTop(args.normal_path, args.adversarial_path)
elif args.load_all_csv is None and not args.use_dict:
    dataset = ShapDataset(args.normal_path, args.adversarial_path)
elif not args.use_dict:
    dataset = ShapDatasetChunked(args.load_all_csv)
else:
    dataset = ShapDatasetDict(args.normal_path, args.adversarial_path, args.max_features)

# Set verbose logging for SVM as we have slow convergence (need to make sure it hasn't crashed!)
libsvm.set_verbosity_wrap(1)

# Split data set into test and train sets
if not args.use_dict:
    data_train, data_test, labels_train, labels_test = train_test_split(dataset.df, dataset.labels, test_size=0.2)
    data_train = preprocessing.scale(data_train)
    data_test = preprocessing.scale(data_test)
else:
    test_size = int(0.2 * len(dataset))
    print(len(dataset))
    test_indices = random.sample(range(len(dataset)), test_size)
    train_indices = [i for i in range(len(dataset)) if i not in test_indices]

    data_train = []
    labels_train = []
    print("======================= Generating Training Set")
    with tqdm.tqdm(total=len(train_indices)) as progress:
        for idx in train_indices:
            data, label = dataset[idx]
            data_train.append(data)
            labels_train.append(label)

            progress.update(1)

    data_test = []
    labels_test = []
    print("======================= Generating Test Set")
    with tqdm.tqdm(total=len(test_indices)) as progress:
        for idx in test_indices:
            data, label = dataset[idx]
            data_test.append(data)
            labels_test.append(label)

            progress.update(1)


results = ""

if args.all:
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['poly'], 'degree': [2, 4, 6, 8, 10], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print('==============================')
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(data_train, labels_train)

        print("Best parameters set found on train set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on train set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full train set.")
        print("The scores are computed on the full train set.")
        print()
        y_true, y_pred = labels_test, clf.predict(data_test)
        print(classification_report(y_true, y_pred))
        print()

