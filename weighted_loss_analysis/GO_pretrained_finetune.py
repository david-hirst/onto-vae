######
### Weighted reconstruction loss workflow
######

#######
## import modules, set paths, specify parameters
#######

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

OntoVAEPath = os.path.join("a","workspace","onto-vae")[1:]
sys.path.append(OntoVAEPath)
from onto_vae.ontobj import *
from onto_vae.vae_model import *

## set path for data shifted from the other cluster
ImportedDataPath = os.path.join("a","workspace","input_data")[1:]

## Set whether to use subset of exisitng expr data or new dataset produced when calculating weights
## True or False, if True then data in trained existing ontology object will be subsetted, if False then data will be imported
gene_expr_subset = True

## set the data type and group details - enter in list consistant with naming order of input files
DEA_groups_list = ['Brain', 'Pancreas']
DEA_groups = DEA_groups_list[0].replace(" ","") + '_' + DEA_groups_list[1].replace(" ","")
DEA_group1 = DEA_groups_list[0]
DEA_group2 = DEA_groups_list[1]

# select terms to plot
## liver spleen terms
# term1 = 'lipid transport'
# term2 = 'gluconeogenesis'
## heart nerve terms
# term1 = 'aortic valve morphogenesis'
# term2 = 'myelination'
## brain pancreas terms
term1 = 'digestion'
term2 = 'glutamate receptor signaling pathway'
# muscle thyroid terms
# term1 = 'muscle system process'
# term2 = 'gland development'
## adipose tissue vs breast
# term1 = 'response to lipid'
# term2 = 'cellular response to peptide'
## heart vs muscle
# term1 = 'aortic valve morphogenesis'
# term2 = 'regulation of heart contraction'

## gene weights paths - random weights file will be created
rec_loss_wts_path = os.path.join(ImportedDataPath,'gene_weights_' + DEA_groups + '.csv')
rec_loss_wts_rndm_path = os.path.join(ImportedDataPath,'gene_weights_rndm_' + DEA_groups + '.csv')

## Set model parameters to be saved by neptune

# rec_loss_wts='None'
# rec_loss_wts=rec_loss_wts_path
# rec_loss_wts=rec_loss_wts_rndm_path

params = {
    "DEA_groups": DEA_groups,
    "lr": 1e-4, 
    "kl_coeff": 1e-4,
    "batch_size": 128,
    "epochs": 100,
    "rec_loss_wts_input": 'None',
    "gene_expr_subset": gene_expr_subset
}

#####
## load saved ontology object
#####

# import the saved ontology
with open(os.path.join(ImportedDataPath, 'GO_ensembl_ontobj.pickle'), 'rb') as f:
    go_o = pickle.load(f)

# add data to the otology
if gene_expr_subset:   
    #### subset saved data and annotations and add new dataset to the ontology
    ## read in sample annotation for all samples
    sample_annot_all = pd.read_csv(os.path.join(ImportedDataPath, 'sample_annot_all.csv'), sep=",").to_numpy()
    ## subset stored data and annotation dataset
    sample_select_grps = np.isin(sample_annot_all[:,-1],np.asarray(DEA_groups_list))
    gene_expr_grps = go_o.data['1000_30']['recount3_GTEx'][sample_select_grps,:].copy()
    sample_annot_grps = sample_annot_all[sample_select_grps,:].copy()
    sample_annot_grps_df = pd.DataFrame({
    'sample': sample_annot_grps[:,0],
    'study': sample_annot_grps[:,1],
    'group': sample_annot_grps[:,2]
    })
    ## add in the subsetted dataset
    go_o.add_dataset(dataset=gene_expr_grps, name = DEA_groups)
else:
    # match the desired dataset to the ontology
    go_o.match_dataset(expr_data = os.path.join(ImportedDataPath, 'gene_expr_' + DEA_groups + '.csv'),
                       name=DEA_groups)



#####################
### OntoVAE initialize and load pretrained model
#####################

# initialize OntoVAE 
go_o_model = OntoVAE(ontobj=go_o,              # the Ontobj we will use
                    dataset=DEA_groups,     # which dataset from the Ontobj to use for model training
                    top_thresh=1000,         # which trimmed version to use
                    bottom_thresh=30)        # which trimmed version to use     
go_o_model.to(go_o_model.device)


# load the pretrained model
checkpoint = torch.load(os.path.join("a","workspace","GTEx_GO_best_model.pt")[1:], map_location = torch.device(go_o_model.device))
go_o_model.load_state_dict(checkpoint['model_state_dict'])

#####################
##### Fine tune with data of interest
#####################

## iniate neptune run
run = neptune.init_run(
    project="",
    api_token="",
)

## set output location for fine tuned model
model_folder = os.path.join(os.getcwd(),'models',str(vars(run)['_sys_id']))  
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)


## create and export random weights
genes_list = pd.DataFrame(go_o_model.genes)
rec_loss_wts_rndm = pd.DataFrame({
   'gene_symbol': genes_list.iloc[:,0],
   'gene_weight': np.random.uniform(low = 0, high = 1, size = genes_list.shape[0])
})
rec_loss_wts_rndm.to_csv(rec_loss_wts_rndm_path, index=False)

# train the model

run["parameters"] = params

if params["rec_loss_wts_input"]=='None':
    rec_loss_wts_input = None
else:
    rec_loss_wts_input = params["rec_loss_wts_input"]

go_o_model.train_model(os.path.join(model_folder, 'best_model.pt'),   # where to store the best model
                      lr = params["lr"],                                 # the learning rate
                      kl_coeff = params["kl_coeff"],                           # the weighting coefficient for the Kullback Leibler loss
                      batch_size = params["batch_size"],                          # the size of the minibatches
                      epochs = params["epochs"],                                # over how many epochs to train
                      run = run,                                                              # whether run should be logged to Neptune
                      rec_loss_wts = rec_loss_wts_input)                      # location of weights file to import or None if no weights 


######
### Analysis with fine tuned OntoVAE model
######

######### load the best model
checkpoint = torch.load(os.path.join(model_folder, 'best_model.pt'),
                        map_location = torch.device(go_o_model.device))
go_o_model.load_state_dict(checkpoint['model_state_dict'])

########### retrieve pathway activities
go_o_act = go_o_model.get_pathway_activities(ontobj=go_o, dataset=DEA_groups)

################### scatter plot for two terms of interest

# terms_check = go_o.extract_annot()

## sample annotations and colour dictionary
if gene_expr_subset:
    sample_annotations_df = sample_annot_grps_df.copy()
else:
    sample_annotations_df = pd.read_csv(os.path.join(ImportedDataPath, 'sample_annot_' + DEA_groups + '.csv'), sep=",", index_col=0)

color_by = 'group'
categs = sample_annotations_df.loc[:,color_by].unique().tolist()
palette = sns.color_palette(cc.glasbey, n_colors=len(categs))
color_dict = dict(zip(categs, palette))

# extract ontology annot and get term indices
onto_annot = go_o.extract_annot(top_thresh=1000, bottom_thresh=30)
ind1 = onto_annot[onto_annot.Name == term1].index.to_numpy()
ind2 = onto_annot[onto_annot.Name == term2].index.to_numpy()

# make scatterplot
fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(x=go_o_act[:,ind1].flatten(),
                y=go_o_act[:,ind2].flatten(),
                hue=sample_annotations_df.loc[:,color_by],
                palette=color_dict,
                legend='full',
                s=15,
                rasterized=True)
plt.xlabel(term1)
plt.ylabel(term2)
plt.tight_layout()
run["images/term_activities"].upload(fig)
plt.close()


##### Statistical significance of terms

# extract ontology annot
onto_annot = go_o.extract_annot()
# get sample annotations
if gene_expr_subset:
    sample_annotations = sample_annot_grps.copy()
else:
    sample_annotations = pd.read_csv(os.path.join(ImportedDataPath, 'sample_annot_' + DEA_groups + '.csv'), sep=",", index_col=0).to_numpy()

# test terms

wilcox = [stats.ranksums(go_o_act[sample_annotations[:,-1]==DEA_group1,i], 
                         go_o_act[sample_annotations[:,-1]==DEA_group2,i]) for i in range(go_o_act.shape[1])]
stat = np.array([i[0] for i in wilcox])
pvals = np.array([i[1] for i in wilcox])
qvals = fdrcorrection(np.array(pvals))

res = pd.DataFrame({'id': onto_annot.ID.tolist(),
                    'term': onto_annot.Name.tolist(),
                    'depth': onto_annot.depth.tolist(),
                    'stat': stat,
                    'pval' : pvals,
                    'qval': qvals[1]})

## Number of significant terms
run["sig_terms"] = np.sum(res['qval']<0.01)


########### UMAP plot

## get the embedding
reducer = umap.UMAP()
embedding = reducer.fit_transform(go_o_act)

## do the plot
if gene_expr_subset:
    sample_annotations_df = sample_annot_grps_df.copy()
else:
    sample_annotations_df = pd.read_csv(os.path.join(ImportedDataPath, 'sample_annot_' + DEA_groups + '.csv'), sep=",", index_col=0)

color_by = 'group'
fig, ax = plt.subplots(1,1, figsize=(10,10))
categs = sample_annotations_df.loc[:,color_by].unique().tolist()
palette = sns.color_palette(cc.glasbey, n_colors=len(categs))
color_dict = dict(zip(categs, palette))

sns.scatterplot(x=embedding[:,0],
                y=embedding[:,1], 
                hue=sample_annotations_df.loc[:,color_by],
                palette=color_dict,
                legend='full',
                s=15,
                rasterized=True)
plt.tight_layout()
run["images/UMAP_all"].upload(fig)
plt.close()


##############

print('finished')
run.stop()
