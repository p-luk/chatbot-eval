#!/bin/bash


# process datasets
python baseline_preprocess.py -u https://shikib.com/pc_usr_data.json -o ./pc_usr_data.tsv
baseline_preprocess.py -u https://shikib.com/tc_usr_data.json -o ./tc_usr_data.tsv

export MODEL_DIR=../prism/m39v1/
# score and plot
python baseline_scores.py --datadir ./pc_usr_data.tsv --outputdir ./pc_usr_scores --plotdir ./figures/pc_usr_plots.png --heatmapdir ./figures/pc_usr_heatmap.png
python baseline_scores.py --datadir ./tc_usr_data.tsv --outputdir ./tc_usr_scores --plotdir ./figures/tc_usr_plots.png --heatmapdir ./figures/tc_usr_heatmap.png

