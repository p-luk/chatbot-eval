# prism-eval

Natural language processing project applying automatic machine translation methods such as PRISM to evaluate and improve conversational agents.

PRISM is available at: https://github.com/thompsonb/prism


## Table of Contents
======================
- `baseline_preprocess.py` usage
- baseline data format

## `baseline_preprocess.py` usage
======================
Usage: baseline_preprocess.py url outputdir

url is the web url to json-format file. Here we use two datasets for baseline metrics:
  1. http://shikib.com/pc_usr_data.json
  2. http://shikib.com/tc_usr_data.json
outputdir is the output file directory (tab-separated format).

## baseline data format
=======================
Because PRISM accepts untokenized input, we use tab-separated value (TSV) format to store examples.
Attributes:
  * context
  * fact
  * ref: human reference
  * cand: model-generated candidate utterance
  * understandable: list of 3 human-annotated scores (0-1)
  * natural: list of 3 human-annotated scores (1-3)
  * maintains context: list of 3 human-annotated scores (1-3)
  * engaging: list of 3 human-annotated scores (1-3)
  * uses knowledge: list of 3 human-annotated scores (0-1)
  * overall: list of 3 human-annotated scores (1-5)
For more context on scores, refer to https://www.aclweb.org/anthology/2020.acl-main.64.pdf.
Note: PRISM does not utilize context or fact. 

