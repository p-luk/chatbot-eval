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
import datasets

__models__ = ['prism', 'bert_score', 'roberta_ft', 'bleu', 'bleurt', 'fed']

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
    if ref not in ['ref', 'context_last', 'empty', 'multi_avg', 'multi_max']:
        raise ValueError("ref must be 'ref' or 'context_last' or 'empty' or 'multi_avg' or 'multi_max.")
    if modelname not in __models__:
        raise ValueError("model not listed")
    # get scores
    if modelname == 'prism':
        model = prism.Prism(model_dir=os.environ['MODEL_DIR'], lang='en')
    elif modelname == 'bert_score':
        pass # no model directory
    elif modelname == 'roberta_ft':
        pass # no model directory
    elif modelname == 'bleu':
        model = datasets.load_metric("sacrebleu")
    elif modelname == 'bleurt':
        model = datasets.load_metric('bleurt', 'bleurt-large-512')
    else:
        warnings.warn('Model not listed.')

    # read in data
    data = pd.read_csv(datadir, sep='\t')
    # determine model inputs
    if ref == 'ref':
        ref_list = data['reference_text'].astype(str).to_list()
    elif ref == 'context_last':
        ref_list = data['prompt_text'].astype(str).to_list()
    elif ref == 'empty':
        ref_list = [''] * len(data['candidate_text'])
    cand_list = data['candidate_text'].astype(str).to_list()

    # determine model and calculate scores
    score = []
    if modelname == 'prism':
        if ref == 'multi_avg' or ref == 'multi_max':
            # ref
            ref_list = data['reference_text'].astype(str).to_list()
            ref_score = [model.score([c], [r]) for c, r in zip(cand_list, ref_list)]
            # context_last
            ref_list = data['prompt_text'].apply(lambda x: str(x).split('\n')[-1]).to_list()
            context_score = [model.score([c], [r]) for c, r in zip(cand_list, ref_list)]
            # empty
            ref_list = [''] * len(data['candidate_text'])
            empty_score = [model.score([c], [r]) for c, r in zip(cand_list, ref_list)]
        else:
            score = [model.score([c], [r]) for c, r in zip(cand_list, ref_list)]
    elif modelname == 'bert_score':
        p, r, score = bert_score.score(cands=cand_list, refs=ref_ist, lang='en', verbose=True)
    elif modelname == 'roberta_ft':
        p, r, score = bert_score.score(cands=cand_list, refs=ref_list, lang='en', verbose=True, model_type='../Chatbot_evaluation/models/roberta_ft', num_layers=10)
    elif modelname == 'bleu':
        if ref == 'multi_avg' or ref == 'multi_max':
            # ref
            ref_list = data['reference_text'].astype(str).to_list()
            bs = [model.compute(predictions=[c], references=[[r]]) for c, r in zip(cand_list, ref_list)]
            ref_score = [x['bp'] for x in bs]
            # context_last
            ref_list = data['prompt_text'].apply(lambda x: str(x).split('\n')[-1]).to_list()

            bs = [model.compute(predictions=[c], references=[[r]]) for c, r in zip(cand_list, ref_list)]
            context_score = [x['bp'] for x in bs]
    elif modelname == 'bleurt':
        preds = model.compute(predictions=cand_list, references=ref_list)
        score = preds['scores']
    
    # add scores to dataframe
    if modelname == 'prism' and (ref == 'multi_avg' or ref == 'multi_max'):
        data['ref_score'] = ref_score
        data['context_score'] = context_score
        data['empty_score'] = empty_score
        if ref == 'multi_avg':
            data['score'] = data[['ref_score', 'context_score', 'empty_score']].mean(axis=1)
        elif ref == 'multi_max':
            data['score'] = data[['ref_score', 'context_score', 'empty_score']].max(axis=1)
    elif modelname == 'bleu' and (ref == 'multi_avg' or ref == 'multi_max'):
        data['ref_score'] = ref_score
        data['context_score'] = context_score
        if ref == 'multi_avg':
            data['score'] = data[['ref_score', 'context_score']].mean(axis=1)
        elif ref == 'multi_max':
            data['score'] = data[['ref_score', 'context_score']].max(axis=1)
    else:
        data['score'] = score

    # write scores to output
    if outputdir is not None:
        data.to_csv(outputdir, sep='\t')
    return data

def plot_correlation(scores, plotdir, ref, modelname):
    """
    plots correlation between human annotation and evaluation scores
    
    Keyword arguments:
    scores -- dataframe of examples and scores
    plotdir -- string directory path to save plots
    """
    scores.dropna(subset=['score', 'win_ratio'], inplace=True)
    evaluation_scores = np.array(scores['score'], dtype=float)
    win_ratio = np.array(scores['win_ratio'], dtype=float)
    # construct correlation plot
    plt.figure()
    fig, ax = plt.subplots(1,1, figsize=(12,12))
    # compute correlation
    slope, intercept, r_value, p_value, std_err = stats.linregress(evaluation_scores, win_ratio)
    # plot
    ax.plot(evaluation_scores, win_ratio, 'o', label='original data')
    ax.plot(evaluation_scores, intercept + slope*evaluation_scores, 'r', label='fitted line')
    ax.set_title(label='$R=${0}\n p={1}'.format(str(round(r_value,6)), str(round(p_value,6))))
    print('$R=${0}\n p={1}'.format(str(round(r_value,6)), str(round(p_value,6))))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.text(0.5, 0.00, 'Automatic Evaluation Metric Score', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Win Ratio', va='center', rotation='vertical', fontsize=12)

    fig.tight_layout()

    plt.savefig(plotdir)
    plt.close()
    
    # for multi ref PRISM, also compare to single ref scores
    if modelname == 'prism' and (ref == 'multi_avg' or ref == 'multi_max'):
        for col in ['ref_score', 'context_score', 'empty_score']:
            # compute correlation
            slope, intercept, r_value, p_value, std_err = stats.linregress(evaluation_scores, scores[col])
            print(col)
            print('$R=${0}\n p={1}'.format(str(round(r_value,6)), str(round(p_value,6))))
    elif modelname == 'bleu' and (ref == 'multi_avg' or ref == 'multi_max'):
        for col in ['ref_score', 'context_score']:
            # compute correlation
            slope, intercept, r_value, p_value, std_err = stats.linregress(evaluation_scores, scores[col])
            print(col)
            print('$R=${0}\n p={1}'.format(str(round(r_value,6)), str(round(p_value,6))))


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
        median_annotations = plot_correlation(scores=scores, plotdir=options.plotdir, ref=options.ref, modelname=options.model)

if __name__ == '__main__':
    main()
    sys.exit(0)
