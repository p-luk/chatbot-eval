#!/bin/bash


# preprocess datasets
python baseline_preprocess.py -u http://shikib.com/tc_usr_data.json -d usr -o ./tc_usr_data.tsv
python baseline_preprocess.py -u http://shikib.com/pc_usr_data.json -d usr -o ./pc_usr_data.tsv
python baseline_preprocess.py -u http://dialog.speech.cs.cmu.edu:9993/static_data.json -d static_data -o ./static_data.tsv

# you may need to modify these to point to the correct directories
export MPLCONFIGDIR=/gridapps/bighomes/js11531/labshare/pl1703/prism-eval
export PYTHONPATH=$PYTHONPATH:/gridapps/bighomes/js11531/labshare/pl1703/prism

# set model directories
export MODEL_DIR=prism/m39v1/ # for PRISM
export MODEL_DIR=../../Chatbot_evaluation/models/roberta_ft # for finetuned roberta

# example usage
# PRISM with reference on Topical Chat dataset
python ./chatbot-eval/baseline_scores.py \
    --model prism \
    --ref ref \
    --datadir ./chatbot-eval/tc_usr_data.tsv \
    --outputdir ./chatbot-eval/tc_usr_prism_scores \
    --plotdir ./chatbot-eval/figures/tc_usr_prism_plots.png \
    --heatmapdir ./chatbot-eval/figures/pc_usr_heatmap.png \
    --ridgeparamsdir ./chatbot-eval/pc_usr_ridgeparams

# PRISM with reference on the static dataset
python ./chatbot-eval/baseline_scores.py \
    --model prism \
    --ref ref \
    --datadir ./chatbot-eval/static_data.tsv \
    --outputdir ./chatbot-eval/static_prism_scores \
    --plotdir ./chatbot-eval/figures/static_prism_plots.png

