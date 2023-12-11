########
#### File to loop through trained models and assess the terms
#### outputs a csv file with AUC and qvalues
########

# import modules
import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import pickle
import umap
import matplotlib.pyplot as plt 
import seaborn as sns
import colorcet as cc
import neptune
from neptune.types import File
import sklearn.model_selection as ms
from sklearn.naive_bayes import GaussianNB

OntoVAEPath = os.path.join("a","workspace","onto-vae")[1:]
sys.path.append(OntoVAEPath)
from onto_vae.ontobj import *
from onto_vae.vae_model import *


## set path for data shifted from the other cluster
ImportedDataPath = os.path.join("a","workspace","input_data")[1:]

## options
gene_expr_subset = True
DEA_groups_lists = [['Liver', 'Spleen'],['Brain', 'Pancreas'],['Adipose Tissue', 'Breast'],['Heart', 'Muscle']]
wts_to_use_list = ['None', 'Derived', 'Random']

## Get the run info to help pick the run of interest
project = neptune.init_project(
    project="",
    api_token="",
    mode="read-only",
)

runs_table_df = project.fetch_runs_table(columns=["sys/id", 
                                                  "parameters/DEA_groups",
                                                  "parameters/gene_expr_subset", 
                                                  "parameters/rec_loss_wts_input"]).to_pandas()

project.stop()

## read in sample annotation for all samples
sample_annot_all = pd.read_csv(os.path.join(ImportedDataPath, 'sample_annot_all.csv'), sep=",").to_numpy()

############################
## loop through the options to assess latent node values form a trained model
############################

tissues_used = []
weights_used = []
term_name = []
term_auc = []
term_qval = []

# loop through the tissue type pairs
for g in range(len(DEA_groups_lists)):

    DEA_groups_list = DEA_groups_lists[g]
    DEA_groups = DEA_groups_list[0].replace(" ","") + '_' + DEA_groups_list[1].replace(" ","")
    DEA_group1 = DEA_groups_list[0]
    DEA_group2 = DEA_groups_list[1]

    ## loop through weights used
    for w in range(len(wts_to_use_list)):

        wts_to_use = wts_to_use_list[w]

        if wts_to_use == 'None':
          rec_loss_wts_input =  'None'
        elif wts_to_use == 'Derived':
          rec_loss_wts_input = os.path.join(ImportedDataPath,'gene_weights_' + DEA_groups + '.csv')
        elif wts_to_use == 'Random':
          rec_loss_wts_input = os.path.join(ImportedDataPath,'gene_weights_rndm_' + DEA_groups + '.csv')
        else:
          print('put corrrect option for weights')

        # get the neptune run id and path of model storage folder
        run_id_to_use = str(runs_table_df.loc[
           (runs_table_df["parameters/DEA_groups"]==DEA_groups) & 
           (runs_table_df["parameters/gene_expr_subset"]==gene_expr_subset) & 
           (runs_table_df["parameters/rec_loss_wts_input"]==rec_loss_wts_input), "sys/id"
           ].iloc[0])
        
        model_folder = os.path.join(os.getcwd(),'models',run_id_to_use)

        # import the saved ontology object
        with open(os.path.join(ImportedDataPath, 'GO_ensembl_ontobj.pickle'), 'rb') as f: 
           go_o = pickle.load(f)

        # subset the data and create new dataset and sample annotation object
        sample_select_grps = np.isin(sample_annot_all[:,-1],np.asarray(DEA_groups_list)).copy()
        sample_annot_grps = sample_annot_all[sample_select_grps,:].copy()
        gene_expr_grps = go_o.data['1000_30']['recount3_GTEx'][sample_select_grps,:].copy()
        # add the data to the ontology object
        go_o.add_dataset(dataset=gene_expr_grps, name = DEA_groups)

        # initialize OntoVAE 
        go_o_model = OntoVAE(ontobj=go_o,
                            dataset=DEA_groups,
                            top_thresh=1000,
                            bottom_thresh=30)     
        go_o_model.to(go_o_model.device) 

        # load the saved model
        checkpoint = torch.load(os.path.join(model_folder, 'best_model.pt'),
                                map_location = torch.device(go_o_model.device))
        go_o_model.load_state_dict(checkpoint['model_state_dict'])

        # retrieve latent node annotations and activity
        onto_annot = go_o.extract_annot(top_thresh=1000, bottom_thresh=30)
        go_o_act = go_o_model.get_pathway_activities(ontobj=go_o, dataset=DEA_groups)

        # loop to evaluate each node
        term_pval_tmp = []
        for t in range(onto_annot.shape[0]):
            # record the name of the terms and model params
            tissues_used.append(DEA_groups)
            weights_used.append(wts_to_use)
            term_name.append(str(onto_annot.iloc[t,1]))
            # median auc of the classifier from cross validation
            clf = GaussianNB()
            term_auc.append(np.nanmedian(ms.cross_val_score(clf,
                                    X = go_o_act[:,t].reshape(-1, 1),
                                    y = sample_annot_grps[:,-1],
                                    cv = ms.KFold(n_splits = 10, shuffle = True),
                                    scoring = 'roc_auc')))
            # pvalues from ranksum test
            term_pval_tmp.append(stats.ranksums(go_o_act[sample_annot_grps[:,-1]==DEA_group1, t],
                                                go_o_act[sample_annot_grps[:,-1]==DEA_group2, t])[1])   
        term_qval.append(fdrcorrection(np.asarray(term_pval_tmp))[1])

#############
#### combine into summary table and export
#############

terms_summary_df = pd.DataFrame({
    'tissues_used': tissues_used,
    'weights_used': weights_used,
    'term_name': term_name,
    'term_auc': term_auc,
    'term_qval': np.reshape(np.asarray(term_qval),(len(tissues_used),1))[:,0]
})
terms_summary_df.to_csv('results/terms_summary_df.csv', index=False)

# neptune upload
run = neptune.init_run(
    project="",
    api_token="",
)
run["data/terms-summary-df-html"].upload(File.as_html(terms_summary_df))
run.stop()

###### the end
print("finished")



