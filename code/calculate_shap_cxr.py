import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

import pickle

from tqdm import tqdm

from utils import VisitSequenceWithLabelDataset, visit_collate_fn, VisdomLinePlotter
from sklearn.metrics import roc_auc_score, average_precision_score
from retain import RETAIN, retain_epoch

from torch.utils.data import Dataset, DataLoader

from captum.attr import LayerConductance, IntegratedGradients, DeepLiftShap, GradientShap

import argparse

from XrayDataset import XrayDataset

import sys

# Function to visualise distribution of importance across features
def vis_importance(feature_names, importances, plotter_instance, title='Average feature importance', plot=True,
                   axis_title='Features'):

    # If we don't have the same size arrays, something has gone wrong
    if len(feature_names) != len(importances):
        raise Exception("feature_names must be of same length as importances")

    print(title)

    # For each feature, print its numeric importance value
    for i in range(len(feature_names)):
        print(feature_names[i], ':', importances[i])

    # Produce a nice plot
    x_pos = (np.arange(len(feature_names)))

    if plot:
        plt.figure() #figsize=(32,10)
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=False, rotation=90, horizontalalignment='center', fontsize='x-small')
        plt.xlabel(axis_title)
        plt.title(title)
        plotter_instance.plot_matplotlib(title, plt)

def calculate_shap_values_sample(inputs, targets, shap, plot=False, baseline=None):
    if baseline is None:
        shap_vals = shap.attribute(inputs, target=targets)
    else:
        shap_vals = shap.attribute(inputs, baselines=baseline, target=targets)
    shap_vals = shap_vals.cpu().detach().numpy()

    # Only show the most important values. Make dictionary first (keys are original feature names)
    if plot:
        avg_shap_vals = np.mean(shap_vals[0][0], axis=0)
        avg_shap_vals_dict = {i: avg_shap_vals[i] for i in range(len(avg_shap_vals))}

        # Now we can get only the top X many results. ordered stores the keys in order
        ordered_shap = sorted(avg_shap_vals_dict, key=lambda i: abs(avg_shap_vals_dict[i]), reverse=True)
        top_shap = {i: avg_shap_vals_dict[i] for i in ordered_shap[:args.num_shap]}
        vis_importance(list(top_shap), list(top_shap.values()), title='SHAP Values (Top {})'.format(args.num_shap),
                       plotter_instance=plotter)

    return shap_vals


if __name__ == '__main__':
    # For now just try Captum so I can get used to working with their code
    arg_parser = argparse.ArgumentParser(description='Use Captum to explain samples from the MIMIC CXR dataset')

    arg_parser.add_argument('model', help='Location of model to explain', type=str)
    arg_parser.add_argument('label', help='Label to classify', type=str)
    arg_parser.add_argument('collate_path', help='Path to collated labels CSV file', type=str)

    arg_parser.add_argument('--cuda', help='Use CUDA', action='store_true')
    arg_parser.add_argument('--normal', help='Location of normal data samples')
    arg_parser.add_argument('--adv', help='Location of adversarial examples', default=None)
    arg_parser.add_argument('--num_ig', help='Number of features to show when plotting integrated gradients', type=int,
                            default=10)
    arg_parser.add_argument('--num_shap', help='Number of features to show when plotting SHAP values', type=int,
                            default=10)
    arg_parser.add_argument('--batch_size', help='Size of batch to do explanations on', type=int, default=16)
    arg_parser.add_argument('--all', '-a', help='Generate SHAP values for all samples', action='store_true')
    arg_parser.add_argument('--num', '-n', help='Number of samples to collect SHAP values for', type=int, default=None)
    arg_parser.add_argument('--save', '-s', help='Path to save SHAP values to', type=str, default=None)
    arg_parser.add_argument('--test', '-t', help='Run tests to check calculated SHAP values', action='store_true')
    arg_parser.add_argument('--mean', help='Use mean of all samples as the baseline value', action='store_true')
    arg_parser.add_argument('--mention', '-m', help='Positive (1) or negative labels (0)', type=int, default=1)
    arg_parser.add_argument('--only', '-o', help='Only look at +ve/-ve samples', type=int, default=None)

    arg_parser.add_argument('--save_top', help="Save only the top X SHAP values", action='store_true')
    arg_parser.add_argument('--choose_shap', help="List of SHAP features to save", default=None, type=int, nargs='*')

    args = arg_parser.parse_args()

    count = 0

    # See if we are using CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.cuda:
        print("======================= Using device:", device)
        if device == 'cpu':
            raise Exception('Ran with --cuda but no GPU available')

    # Load the dataset we were given
    if args.normal:
        print("======================= Loading Normal Dataset")
        dataset = XrayDataset(args.normal, args.collate_path, args.label, args.mention, normalise=False, only=args.only)
    elif args.adv:
        print("======================= Loading Adversarial Dataset")
        dataset = XrayDataset(args.adv, args.collate_path, args.label, args.mention, normalise=False, adv=args.adv)
    else:
        raise Exception('Must provide normal or adversarial dataset')


    test_dataloader = iter(DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True))

    # Load the model
    # Check to see if it was trained onf a GPU or not, act accordingly
    try:
        model = torch.load(args.model)
    except RuntimeError:
        model = torch.load(args.model, map_location=device)

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion = criterion.cuda()

    print("======================= Running model on data...")
    inputs, labels = next(test_dataloader)
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    correct = torch.sum(preds == labels).item()

    print('======================= Got {} sample correct'.format(correct))

    if args.normal:
        plotter = VisdomLinePlotter('cxr-explain-normal')
        type = 'Normal'
    else:
        plotter = VisdomLinePlotter('cxr-explain-adv')
        type = 'Adversarial'

    print("\n============================================== Explanations")

    # Let's try looking at the final layer of the model
    batch_size = args.batch_size
    #cond = LayerConductance(model, model.classifier)

    # Reset our dataloader
    test_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Get an input batch from our dataset
    inputs, targets = next(iter(test_loader))

    inputs = inputs.to(device)

    print('======================= Running batch on device', device)
    targets = targets.to(device)

    #test = model(inputs)

    """print("======================= Calculating Layer Conductance...")
    cond_vals = cond.attribute(inputs, internal_batch_size=batch_size, target=targets)
    cond_vals = cond_vals.detach().numpy()
    
    vis_importance(range(2), np.mean(cond_vals, axis=0), title="Average Neuron Importance", axis_title="Neurons",
                   plotter_instance=plotter)
    print('======================= Done!')"""

    # Look at integrated gradient. This may not change much depending on the data, as the model doesn't change
    ig = IntegratedGradients(model)

    print("======================= Calculating Integrated Gradients...")
    ig_vals = ig.attribute(inputs, internal_batch_size=batch_size, target=targets)
    ig_vals = ig_vals.cpu().numpy()


    # Only show the most important values. Make dictionary first (keys are original feature names)
    avg_ig_vals = np.mean(ig_vals[0][0], axis=0)

    avg_ig_vals_dict = {i: avg_ig_vals[i] for i in range(len(avg_ig_vals))}

    # Now we can get only the top X many results. ordered stores the keys in order
    ordered = sorted(avg_ig_vals_dict, key=lambda i: abs(avg_ig_vals_dict[i]), reverse=True)
    top = {i: avg_ig_vals_dict[i] for i in ordered[:args.num_ig]}

    vis_importance(list(top), list(top.values()), plotter_instance=plotter,
                   title="Integrated Gradients (Top {})".format(args.num_ig))
    print('======================= Done!')

    # Try looking at SHAP values now
    shap = GradientShap(model)

    print("======================= Calculating SHAP Values (GradientShap Approximation)...")
    # We calculate it using only one sample
    batch_size = 1

    # Reset our dataloader
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Using GradientShap so we need a baseline, for now just use a random image
    baseline = torch.randn(3, 3, 224, 224, requires_grad=True, device=device)

    def pad_list(l, length):
        while len(l) < length:
            l.append(0)

        return l

    # If needed, go through all batches
    shap_values = []
    if args.all:
        # We don't have enough memory to run it all at once, so do it in samples
        print('======================= Running all samples...')
        with tqdm(total=len(test_loader)) as progress:
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                shap_values.append(calculate_shap_values_sample(inputs, targets, shap, baseline=baseline))

                progress.update(1)

        # Average the importance over all samples
        # Remove a dimension from the values
        shap_values_reduced_dim = [list(np.asarray(vals).flatten()) for vals in shap_values]
        max_len = max(len(l) for l in shap_values_reduced_dim)
        shap_values_reduced_dim = [pad_list(l, max_len) for l in shap_values_reduced_dim]

        avg_shap_vals = np.mean(shap_values_reduced_dim, axis=0)
        avg_shap_vals_dict = {i: avg_shap_vals[i] for i in range(len(avg_shap_vals))}

        # Now we can get only the top X many results. ordered stores the keys in order
        ordered_shap = sorted(avg_shap_vals_dict, key=lambda i: abs(avg_shap_vals_dict[i]), reverse=True)

        # ordered_shap contains the feature numbers ordered by average importance
        # Go through and only save these values - MUST SAVE SAME SET OF FEATURES FOR NORMAL AND ADVERSARIAL SETS
        all_important_vals = []
        print('======================= Processing SHAP values')
        if args.choose_shap is None:
            with tqdm(total=len(test_loader)) as progress:
                for vals in shap_values:
                    important_vals = []
                    for k in vals:
                        for l in k:
                            count = 0
                            for i in l:
                                if count in ordered_shap[:args.num_ig]:
                                    important_vals.append(i)

                                count += 1

                    # Make sure we have a 1D array
                    important_vals = np.asarray(important_vals).flatten()

                    all_important_vals.append(important_vals)
                    progress.update(1)
        else:
            with tqdm(total=len(test_loader)) as progress:
                for vals in shap_values_reduced_dim:
                    important_vals = []
                    count = 0
                    for k in vals:
                        if count in args.choose_shap:
                            important_vals.append(k)

                        count += 1

                    # Make sure we have a 1D array
                    print(important_vals)
                    sys.exit()
                    all_important_vals.append(important_vals)
                    progress.update(1)

                progress.update(1)

        if args.save_top:
            print('======================= Saving important SHAP values')
            with open(args.save, 'wb') as f:
                pickle.dump(all_important_vals, f)

            print('======================= Important SHAP values saved to', args.save)

        top_shap = {i: avg_shap_vals_dict[i] for i in ordered_shap[:args.num_shap]}
        vis_importance(list(top_shap), list(top_shap.values()), title='SHAP Values (Top {})'.format(args.num_shap),
                       plotter_instance=plotter)

    elif args.num is not None:
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # We don't have enough memory to run it all at once, so do it in samples
        print('======================= Running {} samples...'.format(args.num))
        n = 0
        with tqdm(total=args.num) as progress:
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                shap_values.append(calculate_shap_values_sample(inputs, targets, shap, baseline=baseline))

                progress.update(1)
                n += 1

                if n > args.num:
                    break

        # Average the importance over all samples
        # Make values 1D
        # Average the importance over all samples
        # Remove a dimension from the values
        shap_values_reduced_dim = [list(np.asarray(vals).flatten()) for vals in shap_values]
        max_len = max(len(l) for l in shap_values_reduced_dim)
        shap_values_reduced_dim = [pad_list(l, max_len) for l in shap_values_reduced_dim]

        avg_shap_vals = np.mean(shap_values_reduced_dim, axis=0)
        avg_shap_vals_dict = {i: avg_shap_vals[i] for i in range(len(avg_shap_vals))}

        # Now we can get only the top X many results. ordered stores the keys in order
        ordered_shap = sorted(avg_shap_vals_dict, key=lambda i: abs(avg_shap_vals_dict[i]), reverse=True)

        # ordered_shap contains the feature numbers ordered by average importance
        # Go through and only save these values - MUST SAVE SAME SET OF FEATURES FOR NORMAL AND ADVERSARIAL SETS
        all_important_vals = []
        print('======================= Processing SHAP values')
        if args.choose_shap is None:
            with tqdm(total=len(shap_values_reduced_dim)) as progress:
                for vals in shap_values:
                    important_vals = []
                    for k in vals:
                        for l in k:
                            count = 0
                            for i in l:
                                if count in ordered_shap[:args.num_ig]:
                                    important_vals.append(i)

                                count += 1

                    # Make sure we have a 1D array
                    important_vals = np.asarray(important_vals).flatten()

                    all_important_vals.append(important_vals)
                    progress.update(1)
        else:
            with tqdm(total=len(shap_values_reduced_dim)) as progress:
                for vals in shap_values_reduced_dim:
                    important_vals = []
                    count = 0
                    for k in vals:
                        if count in args.choose_shap:
                            important_vals.append(k)

                        count += 1

                    all_important_vals.append(important_vals)
                    progress.update(1)

                progress.update(1)

        if args.save_top:
            print('======================= Saving important SHAP values')
            with open(args.save, 'wb') as f:
                pickle.dump(all_important_vals, f)

            print('======================= Important SHAP values saved to', args.save)

        top_shap = {i: avg_shap_vals_dict[i] for i in ordered_shap[:args.num_shap]}
        vis_importance(list(top_shap), list(top_shap.values()), title='SHAP Values (Top {})'.format(args.num_shap),
                       plotter_instance=plotter)
    else:
        inputs, targets = next(iter(test_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)

        calculate_shap_values_sample(inputs, targets, shap, plot=True, baseline=baseline)

    if args.save is not None and not args.save_top:
        # Save the SHAP values as a pickled object
        with open(args.save, 'wb') as f:
            pickle.dump(shap_values, f)

        print('======================= Saved SHAP values to', args.save)