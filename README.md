# prism-eval

Natural language processing project applying automatic machine translation methods such as PRISM to evaluate and improve conversational agents.

PRISM is available at: https://github.com/thompsonb/prism
BERTScore is available at: https://github.com/Tiiiger/bert_score


### baseline data format

Because PRISM and BERTScore are embedding-based models, they accept untokenized input. We use tab-separated value (TSV) format to store examples.
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

Usage: `model ref datadir outputdir plotdir heatmapdir ridgeparamsdir`

--model: one of 'prism' or 'bert_score'
--ref: one of 'ref', 'context_last', or 'empty'. 'ref' uses the intended reference. context_last uses the last sentence of the context, which should intuitively correlate well with the 'uses_context' annotation score. 'empty' is used to evaluate the unconditional probability computed by PRISM.
--datadir: directory to save output from `baseline_preprocess`
--outputdir: writes tsv-format file equivalent to the processed data file, with the exception of an additional column for PRISM scores
--plotdir: saves scatterplot correlations and p-values using Pearson's correlation. 
--heatmapdir: saves heatmap of correlations between median annotations
--ridgeparamsdir: text file of results from fitting Ridge regression on annotation data. Includes optimal alpha value from cross-validation, MSE, and coefficients of features
