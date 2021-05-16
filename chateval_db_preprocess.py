#!/usr/bin/env python

import pymysql
import pandas as pd
import argparse
import sys

def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('-d','--dataset', type=str, help='Dataset to load.')
    parser.add_argument('-o','--outputdir', type=str, help='Output directory to save data.')
    return parser

def get_evaluation_df(model_id, ref_id, contexts, references, model_responses, evaluations):
    win_ratio = evaluations.query(f"(model_1=={model_id} and value==0) or (model_2=={model_id} and value==1)").groupby("prompt_id").count().value \
  / \
    evaluations.query(f"model_1=={model_id} or model_2=={model_id}").groupby("prompt_id").count().value
  
    
    m = model_responses[model_responses.model_id==model_id].set_index("prompt_id")
    r1 = references[references.ref_id==ref_id].set_index("prompt_id")
    c = contexts.set_index("prompt_id")
    df = c.join(r1).join(m)
    df["win_ratio"] = win_ratio
    df["counts"] = evaluations.query(f"model_1=={model_id} or model_2=={model_id}").groupby("prompt_id").count().value
    return df

def get_chateval_data(dataset, outputdir):
    """Query ChatEval database and write data in TSV format."""
    
    # database queries
    db = pymysql.connect(host="jsedocc7.scrc.nyu.edu", user="researchers", passwd="x4BLwNY2Ioc=",db="demo")
    c = db.cursor()
    evalset = dataset
    #evalset = 'ESL 3-turn'
    # evalset = 'NCM'
    #evalset = 'DBDC'
    #evalset = 'twitter'
    #evalset ='Cornell Movie Dialogue Corpus'
    
    c.execute("SELECT * from demo.HumanEvaluationsABComparison")
    raw_eval_data = c.fetchall()
    c.execute("SELECT * from demo.HumanEvaluations")
    mturk_runs = c.fetchall()
    c.execute('SELECT model_id, name from demo.Model')
    model_names = c.fetchall()
    c.execute(f"""SELECT prompt_id from demo.EvaluationDatasetText, demo.EvaluationDataset 
               where demo.EvaluationDatasetText.evaluationdataset_id = demo.EvaluationDataset.evalset_id
                and name=\"{evalset}\" """)
    prompt_ids = c.fetchall()
    prompt_ids = [x[0] for x in prompt_ids]
    contexts = pd.read_sql(f"""
    SELECT prompt_id, prompt_text 
    from 
      EvaluationDatasetText, EvaluationDataset 
    where 
      EvaluationDatasetText.evaluationdataset_id=EvaluationDataset.evalset_id and
      demo.EvaluationDataset.name=\"{evalset}\" """, con=db)
    references = pd.read_sql(f"""
    SELECT 
      prompt_id, 
      Model.model_id as ref_id, 
      Model.name as reference_name, 
      response_text as reference_text
    from 
      ModelResponse, 
      Model, 
      EvaluationDataset_baselines, 
      EvaluationDataset 
    where 
      Model.model_id=ModelResponse.model_id and 
      EvaluationDataset_baselines.model_id = Model.model_id and 
      EvaluationDataset.evalset_id = EvaluationDataset_baselines.evaluationdataset_id and
      EvaluationDataset.name=\"{evalset}\" and
      is_baseline<3""",
      con=db)
    model_responses = pd.read_sql(f"""
    SELECT DISTINCT 
      prompt_id, 
      Model.model_id, 
      Model.name as model_name, 
      response_text as candidate_text 
    from 
      ModelResponse, 
      Model, 
      EvaluationDataset_baselines, 
      EvaluationDataset 
    where 
      Model.model_id=ModelResponse.model_id and 
      EvaluationDataset.evalset_id = ModelResponse.evaluationdataset_id and 
      EvaluationDataset.name=\"{evalset}\" and
      (is_baseline>2 or is_baseline=0)""",      
    con=db)
    #HACK: this is to fix the issue of duplicate insertions into the DB
    contexts = contexts[contexts.prompt_id.isin(model_responses.prompt_id.unique())]
    evaluations = pd.read_sql(f"""
    SELECT 
      value, model_1, model_2, prompt_id, mturk_run_id 
    FROM 
      HumanEvaluationsABComparison, 
      HumanEvaluations
    where 
      HumanEvaluationsABComparison.mturk_run_id_id=HumanEvaluations.mturk_run_id and
      HumanEvaluationsABComparison.prompt_id in 
        (select prompt_id 
          from EvaluationDatasetText, EvaluationDataset 
          where EvaluationDatasetText.evaluationdataset_id = EvaluationDataset.evalset_id and
            name=\"{evalset}\")""",
      con=db)
    # HACK using the first reference i.e. references.ref_id[0]
    ref_id = references.ref_id[0]
    eval_metrics_df = None
    for model in model_responses.model_id.unique():
        df = get_evaluation_df(model, ref_id, contexts, references, model_responses, evaluations)
        if type(eval_metrics_df) != type(None):
            eval_metrics_df = pd.concat([eval_metrics_df, df])
        else:
            eval_metrics_df = df
    if dataset == 'ESL 3-turn':
        eval_metrics_df['prompt_text'] = eval_metrics_df['prompt_text'].apply(lambda x: x.split('\n')[-1][2:])
    # write to TSV
    if outputdir is not None:
        eval_metrics_df.to_csv(outputdir, sep='\t')

def main():
    print("entered main")
    arg_parser = create_arg_parser()
    try:
        options = arg_parser.parse_args()
    except ArgumentError():
        print('chateval_db_preprocess.py -d <dataset> -o <outputdir>')
    print(options)
    print("getting data...")
    get_chateval_data(dataset=options.dataset, outputdir=options.outputdir)
        
    print("done")
if __name__ == '__main__':
    main()
    sys.exit(0)


