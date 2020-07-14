from visdom import Visdom

from utils import AverageMeter, VisdomLinePlotter

from ShapDataset import ShapDataset, ShapDatasetChunked, ShapDatasetDict, ShapDatasetTop

import argparse

import pickle
import random

global plotter
plotter = VisdomLinePlotter('attack-svm')

from sklearn.model_selection import train_test_split
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
# We only want to train if we haven't loaded a model
if args.load is None:
    print("======================= Training Linear SVM")
    svm = SVC(kernel='linear', verbose=True, shrinking=args.shrinking)
    svm.fit(data_train, labels_train)

    if args.save is not None:
        path = os.path.join(args.save, 'linear-svm.pkl')
        with open(path, 'wb') as f:
            pickle.dump(svm, f)
else:
    print("======================= Loading Linear SVM")
    with open(args.load, 'rb') as f:
        svm = pickle.load(f)

print("======================= Evaluating Linear SVM")
preds = svm.predict(data_test)

conf_matrix = confusion_matrix(labels_test, preds).tolist()
class_report = classification_report(labels_test, preds)

plotter.plot_text(conf_matrix, 'Linear SVM Confusion Matrix')
plotter.plot_text(class_report, 'Linear SVM Classification Report')

print(conf_matrix)
print('=======================')
print(class_report)

results += "======================= Evaluating Linear SVM\n{}\n\n{}=======================\n\n".format(conf_matrix,
                                                                                                       class_report)

if args.all:
    print("======================= Training Polynomial SVM")
    svm = SVC(kernel='poly', degree=args.degree, verbose=True, shrinking=args.shrinking)
    svm.fit(data_train, labels_train)

    if args.save is not None:
        path = os.path.join(args.save, 'poly-svm.pkl')
        with open(path, 'wb') as f:
            pickle.dump(svm, f)

    print("======================= Evaluating Polynomial SVM")
    preds = svm.predict(data_test)

    conf_matrix = confusion_matrix(labels_test, preds).tolist()
    class_report = classification_report(labels_test, preds)

    plotter.plot_text(conf_matrix, 'Polynomial SVM Confusion Matrix')
    plotter.plot_text(class_report, 'polynomial SVM Classification Report')

    print(conf_matrix)
    print('=======================')
    print(class_report)

    results += "======================= Evaluating Poly SVM\n{}\n\n{}=======================\n\n".format(conf_matrix,
                                                                                                         class_report)

    print("======================= Training Gaussian SVM")
    svm = SVC(kernel='rbf', verbose=True, shrinking=args.shrinking)
    svm.fit(data_train, labels_train)

    if args.save is not None:
        path = os.path.join(args.save, 'gauss-svm.pkl')
        with open(path, 'wb') as f:
            pickle.dump(svm, f)

    print("======================= Evaluating Gaussian SVM")
    preds = svm.predict(data_test)

    conf_matrix = confusion_matrix(labels_test, preds).tolist()
    class_report = classification_report(labels_test, preds)

    plotter.plot_text(conf_matrix, 'Gaussian SVM Confusion Matrix')
    plotter.plot_text(class_report, 'Gaussian SVM Classification Report')

    print(conf_matrix)
    print('=======================')
    print(class_report)

    results += "======================= Evaluating RBF SVM\n{}\n\n{}=======================\n\n".format(conf_matrix,
                                                                                                        class_report)

    print("======================= Training Sigmoid SVM")
    svm = SVC(kernel='sigmoid', verbose=True, shrinking=args.shrinking)
    svm.fit(data_train, labels_train)

    if args.save is not None:
        path = os.path.join(args.save, 'sigmoid-svm.pkl')
        with open(path, 'wb') as f:
            pickle.dump(svm, f)

    print("======================= Evaluating Sigmoid SVM")
    preds = svm.predict(data_test)

    conf_matrix = confusion_matrix(labels_test, preds).tolist()
    class_report = classification_report(labels_test, preds)

    plotter.plot_text(conf_matrix, 'Sigmoid SVM Confusion Matrix')
    plotter.plot_text(class_report, 'Sigmoid SVM Classification Report')

    print(conf_matrix)
    print('=======================')
    print(class_report)

    results += "======================= Evaluating Sigmoid SVM\n{}\n\n{}=======================\n\n".format(conf_matrix,
                                                                                                        class_report)

    if args.save_results is not None:
        with open(args.save_results, 'w') as f:
            f.write(results)
