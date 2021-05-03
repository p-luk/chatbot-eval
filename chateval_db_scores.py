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

__all__ = ['prism', 'bert_score', 'roberta_ft']

def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('-d','--datadir', type=str, help='Path to data from baseline_preprocess.py.')
    parser.add_argument('-o','--outputdir', type=str, help='Path to output to write scores.')
    parser.add_argument('-p','--plotdir', type=str, help='Path to output to write plots.')
    parser.add_argument('--ref', type=str, help='Reference to use: ref, empty.')
    parser.add_argument('--model', type=str, help='Model to use. Implemented for prism, bert_score, roberta_ft.')
    return parser

def get_scores(modelname, datadir, outputdir=None, ref='ref'):
    """
    Uses specified model to score examples read from datadir. See README.md for proper data format
    
    Keyword arguments:
    model -- model to use
    datadir -- string directory to fetch data (tsv)
    outputdir -- optional string directory path to save output
    ref -- optional string denoting reference string. either 'ref' or 'empty'
    """
    # check ref argument validity
    if ref not in ['ref', 'context_last', 'empty']:
        raise ValueError("ref must be 'ref' or 'context_last' or 'empty'.")
    if modelname not in __all__:
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
    # read in data
    data = pd.read_csv(datadir, sep='\t')
    # determine model inputs
    if ref == 'ref':
        ref = data.reference_text.to_list()
    elif ref == 'context_last':
        ref = data.prompt_text.to_list()
    elif ref == 'empty':
        ref = [''] * len(data.candidate_text.to_list())
    cand = data.candidate_text.to_list()
    # enforce string format
    ref = [str(x) for x in ref]
    cand = [str(x) for x in cand]
    # determine model
    if modelname == 'prism':
        score = model.score(cand=cand, ref=ref)
    elif modelname == 'bert_score':
        p, r, score = bert_score.score(cands=cand, refs=ref, lang='en', verbose=True)
    elif modelname == 'roberta_ft':
        p, r, score = bert_score.score(cands=cand, refs=ref, lang='en', verbose=True, model_type='../Chatbot_evaluation/models/roberta_ft', num_layers=10)
    data['score'] = score
    # write scores to output
    if outputdir is not None:
        data.to_csv(outputdir, sep='\t')
    return data

def plot_correlation(scores, plotdir):
    """
    plots correlation between human annotation and evaluation scores
    
    Keyword arguments:
    scores -- list of examples and scores
    plotdir -- string directory path to save plots
    """
    evaluation_scores = scores['score'].astype(float)
    win_ratio = scores['win_ratio'].astype(float)
    # construct correlation plot
    plt.figure()
    fig, ax = plt.subplots(1,1, figsize=(12,12))
    # compute correlation
    slope, intercept, r_value, p_value, std_err = stats.linregress(evaluation_scores, win_ratio)
    # plot
    ax.plot(evaluation_scores, win_ratio, 'o', label='original data')
    ax.plot(evaluation_scores, intercept + slope*evaluation_scores, 'r', label='fitted line')
    ax.set_title(label='$R=${0}\n p={1}'.format(str(round(r_value,6)), str(round(p_value,6))))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.text(0.5, 0.00, 'Automatic Evaluation Metric Score', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Win Ratio', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout()

    plt.savefig(plotdir)
    plt.close()

def main():
    print("Entered main...")
    arg_parser = create_arg_parser()
    try:
        options = arg_parser.parse_args()
    except ArgumentError():
        print('baseline_scores.py --model <model> --ref <ref> -d <datadir> -o <outputdir> -p <plotdir>')
    print(options)
    print('Getting scores... ')
    scores = get_scores(modelname=options.model, datadir=options.datadir, outputdir=options.outputdir, ref=options.ref)
    if options.plotdir is not None:
        print('Getting correlations...')
        median_annotations = plot_correlation(scores=scores, plotdir=options.plotdir)

if __name__ == '__main__':
    main()
    sys.exit(0)
