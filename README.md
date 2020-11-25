# prism-eval

Natural language processing project applying automatic machine translation methods such as PRISM to evaluate and improve conversational agents.

PRISM is available at: https://github.com/thompsonb/prism


### baseline data format

Because PRISM accepts untokenized input, we use tab-separated value (TSV) format to store examples.
Attributes:
  * `context`
  * `fact`
  * `ref`: human reference
  * `cand`: model-generated candidate utterance
  * `understandable`: list of 3 human-annotated scores (0-1)
  * `natural`: list of 3 human-annotated scores (1-3)
  * `maintains_context`: list of 3 human-annotated scores (1-3)
  * `engaging`: list of 3 human-annotated scores (1-3)
  * `uses_knowledge`: list of 3 human-annotated scores (0-1)
  * `overall`: list of 3 human-annotated scores (1-5)
For more context on scores, refer to https://www.aclweb.org/anthology/2020.acl-main.64.pdf.

Note: PRISM does not utilize context or fact. For example usage or to reproduce results  of the following, see `baseline.sh`.



#### `baseline_preprocess.py` usage

Usage: `baseline_preprocess.py url outputdir`

--url: web url to json-format file
--outputdir is the output file directory (tab-separated format). Here we use two datasets for baseline metrics:
  1. https://shikib.com/pc_usr_data.json
  2. https://shikib.com/tc_usr_data.json


#### `baseline_scores.py`

Usage: `datadir outputdir plotdir heatmapdir ridgeparamsdir contextplotdir`

--datadir: directory to save output from `baseline_preprocess`
--outputdir: writes tsv-format file equivalent to the processed data file, with the exception of an additional column for PRISM scores
--plotdir: saves scatterplot correlations and p-values using Pearson's correlation. 
--heatmapdir: saves heatmap of correlations between median annotations
--ridgeparamsdir: text file of results from fitting Ridge regression on annotation data. Includes optimal alpha value from cross-validation, MSE, and coefficients of features
--contextplotdir: plots identical to --plotdir, but using the last line of context rather than ref. 
