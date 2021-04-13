#!/usr/bin/env python

import sys
import argparse
import os
import warnings
import json
import prism
import transformers
import bert_score
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import csv
import seaborn as sns
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

__all__ = ['prism', 'bert_score', 'roberta_ft']

def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('-d','--datadir', type=str, help='Path to data from baseline_preprocess.py.')
    parser.add_argument('-o','--outputdir', type=str, help='Path to output to write scores.')
    parser.add_argument('-p','--plotdir', type=str, help='Path to output to write plots.')
    parser.add_argument('-m', '--heatmapdir', type=str, help='Path to output to write heatmap of human annotations.')
    parser.add_argument('-r', '--ridgeparamsdir', type=str, help='Path to output to write results of Ridge regression.')
    parser.add_argument('--ref', type=str, help='Reference to use: ref, context_last, null.')
    parser.add_argument('--model', type=str, help='Model to use. Implemented for prism, bert_score, roberta_ft.')
    return parser

def get_scores(modelname, datadir, outputdir=None, ref='ref'):
    """
    Uses specified model to score examples read from datadir. See README.md for proper data format
    
    Keyword arguments:
    model -- model to use
    datadir -- string directory to fetch data (tsv)
    outputdir -- optional string directory path to save output
    ref -- optional string denoting reference string. either 'ref' or 'context_last'
    """
    # check ref argument validity
    if ref not in ['ref', 'context_last', 'empty']:
        raise ValueError("ref must be 'ref' or 'context_last' or 'empty'.")
    if modelname not in ['prism', 'bert_score', 'roberta_ft']:
        raise ValueError("model not listed")
    # get scores
    if modelname == 'prism':
        model = prism.Prism(model_dir=os.environ['MODEL_DIR'], lang='en')
    elif modelname == 'bert_score':
        pass # no model directory
    elif modelname == 'roberta_ft':
        pass # no model directory
    else:
        warnings.warn('Model not listed.')
    scores = []
    with open(datadir, 'r') as f:
        data = csv.reader(f, delimiter="\t", quotechar='"')
        header = next(data)
        header.append('prism_score')
        scores.append(header)
        ind = {'context':(header.index('context')), 'ref':(header.index('ref')), 'cand':(header.index('cand')), 'understandable':(header.index('understandable'))}
        for row in data:
            # determine model inputs
            if ref == 'ref':
                ref = row[ind['ref']]
            elif ref == 'context_last':
                ref = row[ind['context']].split('\n')[-3] # context ends in '\n\n'
            elif ref == 'empty':
                ref = ""
            cand = row[ind['cand']]
            # determine model
            if modelname == 'prism':
                score = model.score(cand=[cand], ref=[ref])
            elif modelname == 'bert_score':
                p, r, f1 = bert_score.score(cands=[cand], refs=[ref], lang='en', verbose=True)
                score = f1.mean().item()
            elif modelname == 'roberta_ft':
                p, r, f1 = bert_score.score(cands=[cand], refs=[ref], lang='en', model_type='../Chatbot_evaluation/models/roberta_ft', num_layers=10)
                score = f1.mean().item()
        scores.append(row + [str(score)])
    if outputdir is not None:
        with open(outputdir, 'w') as f:
            dw = csv.writer(f, delimiter='\t')
            dw.writerows(scores)
    return pd.DataFrame(scores[1:],columns=scores[0])

def median_annotation(x):
    """Returns median value from list. Applied to Series."""
    return statistics.median(json.loads(x))

def plot_correlation(scores, plotdir, heatmapdir=None):
    """
    plots correlation between human annotation and PRISM scores
    
    Keyword arguments:
    scores -- list of examples and scores
    plotdir -- string directory path to save plots
    heatmapdir -- string directory path to save plot, optional
    """
    prism_scores = scores['prism_score'].astype(float)
    human_annotations = ['understandable', 'natural', 'maintains_context', 'engaging','uses_knowledge','overall']
    median_annotations = pd.DataFrame()

    # construct correlation plot
    plt.figure()
    fig, ax = plt.subplots(2,3, figsize=(12,8))
    ax = ax.flatten()
    for i,quality in enumerate(human_annotations):
        # compute correlation
        quality_scores = scores[quality].apply(median_annotation).astype(float)
        median_annotations[quality] = quality_scores
        slope, intercept, r_value, p_value, std_err = stats.linregress(prism_scores, quality_scores)
        # plot
        ax[i].plot(prism_scores, quality_scores, 'o', label='original data')
        ax[i].plot(prism_scores, intercept + slope*prism_scores, 'r', label='fitted line')
        ax[i].set_title(label='{0}: $R^2=${1}\n p={2}'.format(quality, str(round(r_value**2,6)), str(round(p_value,6))))
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.text(0.5, 0.00, 'PRISM Score', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Median Human Annotator Score', va='center', rotation='vertical', fontsize=12)
    plt.suptitle('PRISM vs. Median Human Annotator Score Correlation', y=1.05)
    fig.tight_layout()

    plt.savefig(plotdir)
    plt.close()
    
    # plot heatmap
    if heatmapdir is not None:
        plot_heatmap(median_annotations, heatmapdir)
    return(median_annotations)

def plot_heatmap(median_annotations, heatmapdir):
    """
    plots heatmap - correlation  between human annotations
    heatmapdir -- string directory path to save plots
    """
    plt.figure()
    ax = sns.heatmap(median_annotations.corr(), annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Median Human Quality Score Correlation')
    plt.savefig(heatmapdir)
    plt.close()

def ridge_reg(median_annotations, ridgeparamsdir):
    """
    Trains Ridge model from sklearn.linear_model.
    Formula: overall ~ understandable + natural + maintains_context + interesting + uses_knowledge
    median_annotations -- list of human annotations from USR dataset; aggregated by median
    ridgeparamsdir -- output directory. text file containing parameters and scores of best model from cross-validation
    """
    # shuffle data and split train/test
    X = median_annotations.loc[:, median_annotations.columns != 'overall'].to_numpy()   # y = overall in last column
    y = median_annotations[['overall']].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # ridge regression with cross validation
    clf_cv = RidgeCV().fit(X_train,y_train)
    with open(ridgeparamsdir, 'w') as f:
        f.write("Alpha = " +  str(clf_cv.alpha_) + "\n")   # best alpha from cross-validation
        clf_res = Ridge(alpha = clf_cv.alpha_, normalize = True)
        clf_res.fit(X_train, y_train)
        mse = mean_squared_error(y_test, clf_res.predict(X_test))
        f.write("Mean Squared Error = " +  str(mse) + "\n")
        clf_res.fit(X,y)
        f.write('\t'.join(median_annotations.columns) + '\n')
        f.write('\t'.join([str(i) for i in clf_res.coef_]) + '\n')


def main():
    print("Entered main... ")
    arg_parser = create_arg_parser()
    try:
        options = arg_parser.parse_args()
    except ArgumentError():
        print('baseline_scores.py --model <model> --ref <ref> -d <datadir> -o <outputdir> -p <plotdir> -m <heatmapdir> -r <ridgeparamsdir>')
    print(options)
    print('Getting scores... ')
    scores = get_scores(modelname=options.model, datadir=options.datadir, outputdir=options.outputdir, ref=options.ref)
    print(scores.head())
    if options.plotdir is not None:
        print('Getting correlations...')
        median_annotations = plot_correlation(scores=scores, plotdir=options.plotdir, heatmapdir=options.heatmapdir)
    if options.ridgeparamsdir is not None:
        print('Running Ridge regression...')
        ridge_reg(median_annotations=median_annotations, ridgeparamsdir=options.ridgeparamsdir)

if __name__ == '__main__':
    main()
    sys.exit(0)
