# Onto-VAE with reconstruction loss weights

The object of this project was to allow the optional inclusion of input node weightings for the reconstruction loss of OntoVAE. We have implemented this in the [vae_model.py](https://github.com/david-hirst/onto-vae/blob/main/onto_vae/vae_model.py) module in this forked repository. We have included an argument `rec_loss_wts` in the function which is called to train Onto-VAE 
```
def train_model(self, modelpath, lr=1e-4, kl_coeff=1e-4, batch_size=128, epochs=300, run=None, rec_loss_wts=None):
```
If the user wants to weight the relative contributions of input nodes to the reconstruction loss, they enter a path to a csv file containing the weights, for example
```
ontovae_model.train_model(
modelpath = os.path.join(os.getcwd(),'best_model.pt'),
rec_loss_wts = os.path.join(os.getcwd(),'gene_weights.csv')
)
```
The first column of the csv should contain gene symbols consistant with those used in the loaded ontology object. The second column should contain a weight for each gene. The function expects the csv file to contain column headings, although the headings themselves are ignored. If the user ignores this argument then the training is performed without any node weighting.

## Background

OntoVAE is a variational autoencdoer (VAE) that allows for the incorporation of prior knowledge in the form of a biological ontology. A VAE consists of two neural networks, an encoder and a decoder. The input for the encoder is $\bpldsymbol{x}^{(i)}$, which is a $D$-dimensional vector whose elements are the input nodes.

## Evaluation

The empirical cumulative distribution of AUC scores

<img src="images/AUC-EDCF-plots.png">

Boxplots of the AUC values

<img src="images/AUC_boxplots.png">
