#!/usr/bin/env python


import argparse
import requests
import json
import sys
import getopt
import os
import csv

__datasets__ = ['usr', 'static_data', 'fed']

def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('-u','--url', type=str, help='URL to fetch json data.')
    parser.add_argument('-d','--dataset', type=str, help='One of \'usr\' or \'static_data\'. We hardcode the schema for each.')
    parser.add_argument('-o','--outputdir', type=str, help='Path to the output to write.')
    return parser

def json_preprocess_usr(url, outputdir):
    """
    Fetch Topical Chat/Persona Chat json data and transform to tab-separated value format

    Keyword arguments:
    url -- url to Topical Chat or Persona Chat location to fetch data
    outputdir -- path to save tsv output
    """
    data = json.loads(requests.get(url).text)
    delimiter='\t'
    
    header = ['context', 'fact', 'cand', 'model', 'Understandable', 'Natural', 'Maintains Context', 'Engaging', 'Uses Knowledge', 'Overall']
    with open(outputdir, 'w') as f:
        dw = csv.DictWriter(f, fieldnames=[str.lower(col).replace(' ', '_') for col in header], delimiter=delimiter)
        dw.writeheader()
        for context_fact in data:
            for response in context_fact['responses']:
                example = {}
                for col in header:
                    if col == 'cand':
                        example[col] = response['response']
                    elif col in ('context', 'fact'):
                        example[col] = context_fact[col].replace('"','').strip()
                    else:
                        example[str.lower(col).replace(' ', '_')] = response[col] 
                dw.writerow(example)

def json_preprocess_static_data(url, outputdir):
    """
    Fetch static_data json data and transform to tab-separated value format

    Keyword arguments:
    url -- url to location to fetch data
    outputdir -- path to save tsv output
    """
    data = json.loads(requests.get(url).text)
    delimiter='\t'
    
    header = ['dialog_history', 'dialog_response', 'system', 'fluent', 'understandable', 'interesting', 'relevant', 'human', 'engaging', 'correct', 'semantically appropriate', 'specific']
    normalized_header = ['context', 'cand', 'model', 'fluent', 'understandable', 'interesting', 'relevant', 'human', 'engaging', 'correct', 'semantically appropriate', 'specific']
    with open(outputdir, 'w') as f:
        dw = csv.DictWriter(f, fieldnames=[str.lower(col).replace(' ', '_') for col in normalized_header], delimiter=delimiter)
        dw.writeheader()
        for line in data:
            example = {}
            for col in header:
                if col == 'dialog_response':
                    example['cand'] = line[col]
                elif col == 'dialog_history':
                    example['context'] = line[col].replace('"','').strip()
                elif col == 'system':
                    example['model'] = line[col].replace('"', '').strip()
                else:
                    example[str.lower(col).replace(' ', '_')] = line[col] 
            dw.writerow(example)

def json_preprocess_fed_referenced_data(url, outputdir):
    """
    Fetch fed json data and transform to tab-separated value format

    Keyword arguments:
    url -- url to location to fetch data
    outputdir -- path to save tsv output
    """
    data = json.loads(requests.get(url).text)
    delimiter='\t'

    header = ['context', 'response', 'system', 'Interesting', 'Engaging', 'Specific', 'Relevant', 'Correct', 'Semantically appropriate', 'Understandable', 'Fluent', 'Overall']
    normalized_header = ['context', 'cand', 'model', 'Interesting', 'Engaging', 'Specific', 'Relevant', 'Correct', 'Semantically appropriate', 'Understandable', 'Fluent', 'Overall']

    with open(outputdir, 'w') as f:
        dw = csv.DictWriter(f, fieldnames=[str.lower(col).replace(' ', '_') for col in normalized_header], delimiter=delimiter)
        dw.writeheader()
        for line in data:
            if 'response' in line.keys(): # filter examples without system response
                example = {}
                for col in header:
                    if col == 'response':
                        example['cand'] = line[col].split(':')[-1]
                    elif col == 'context':
                        example['context'] = line[col].replace('"','').strip()
                    elif col == 'system':
                        example['model'] = line[col].replace('"', '').strip()
                    else:
                        example[str.lower(col).replace(' ', '_')] = line['annotations'][col]
                dw.writerow(example)
            else:
                pass

def json_preprocess_fed_unreferenced_data(url, outputdir):
    """
    Fetch fed json data and transform to tab-separated value format

    Keyword arguments:
    url -- url to location to fetch data
    outputdir -- path to save tsv output
    """
    data = json.loads(requests.get(url).text)
    delimiter='\t'

    header = ['context', 'system', 'Coherent', 'Error recovery', 'Consistent', 'Diverse', 'Depth', 'Likeable', 'Understanding', 'Flexible', 'Informative', 'Inquisitive', 'Overall']
    normalized_header = ['context', 'model', 'Coherent', 'Error recovery', 'Consistent', 'Diverse', 'Depth', 'Likeable', 'Understanding', 'Flexible', 'Informative', 'Inquisitive', 'Overall']

    with open(outputdir, 'w') as f:
        dw = csv.DictWriter(f, fieldnames=[str.lower(col).replace(' ', '_') for col in normalized_header], delimiter=delimiter)
        dw.writeheader()
        for line in data:
            if 'response' not in line.keys():
                example = {}
                for col in header:
                    if col == 'context':
                        example['context'] = line[col].replace('"','').strip()
                    elif col == 'system':
                        example['model'] = line[col].replace('"', '').strip()
                    else:
                        example[str.lower(col).replace(' ', '_')] = line['annotations'][col]
                dw.writerow(example)


def main():
    print("entered main")
    arg_parser = create_arg_parser()
    try:
        options = arg_parser.parse_args()
    except ArgumentError():
        print('baseline_preprocess.py -u <url> -d <dataset> -o <outputdir>')
    print(options)
    if options.dataset == 'usr':
        json_preprocess_usr(url=options.url, outputdir=options.outputdir)
    elif options.dataset == 'static_data': 
        json_preprocess_static_data(url=options.url, outputdir=options.outputdir)
    elif options.dataset == 'fed_referenced':
        json_preprocess_fed_referenced_data(url=options.url, outputdir=options.outputdir)
    elif options.dataset == 'fed_unreferenced':
        json_preprocess_fed_unreferenced_data(url=options.url, outputdir=options.outputdir)

if __name__ == '__main__':
    main()
    sys.exit(0)
