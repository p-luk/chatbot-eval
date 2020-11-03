#!/usr/bin/env python


import argparse
import requests
import json
import sys
import getopt
import os
import csv

def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('-u','--url', type=str, help='URL to fetch json data.')
    parser.add_argument('-o','--outputdir', type=str, help='Path to the output to write.')
    return parser

def json_preprocess(url, outputdir):
    """
    Fetch json data and transform to tab-separated value format

    Keyword arguments:
    url -- location to fetch json data
    outputdir -- path to save tsv output
    """
    data = json.loads(requests.get(url).text)
    delimiter='\t'
    header = ['context', 'fact', 'ref', 'cand', 'model', 'Understandable', 'Natural', 'Maintains Context', 'Engaging', 'Uses Knowledge', 'Overall']
    with open(outputdir, 'w') as f:
        dw = csv.DictWriter(f, fieldnames=[str.lower(col).replace(' ', '_') for col in header], delimiter=delimiter)
        dw.writeheader()
        for context_fact in data:
            for response in context_fact['responses']:
                if response['model'] == 'Original Ground Truth':
                    ref = response['response']
                else:
                    example = {}
                    for col in header:
                        if col == 'ref':
                            example[col] = ref.replace('"','').strip()
                        elif col == 'cand':
                            example[col] = response['response']
                        elif col in ('context', 'fact'):
                            example[col] = context_fact[col].replace('"','').strip()
                        else:
                            example[str.lower(col).replace(' ', '_')] = response[col] 
                    dw.writerow(example)

def main():
    print("entered main")
    arg_parser = create_arg_parser()
    try:
        options = arg_parser.parse_args()
    except ArgumentError():
        print('baseline_preprocess.py -u <url> -o <outputdir>')
    print(options)
    json_preprocess(url=options.url, outputdir=options.outputdir)

if __name__ == '__main__':
    main()
    sys.exit(0)
