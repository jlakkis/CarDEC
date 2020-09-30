from .CarDEC_utils import normalize_scanpy
from .CarDEC_MainModel import CarDEC_Model
from .CarDEC_count_decoder import count_model

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from pandas import DataFrame

import os

class CarDEC_API:
    def __init__(self, adata, preprocess=True, weights_dir = "CarDEC Weights", batch_key = None, n_high_var = 2000, LVG = True,
                     normalize_samples = True, log_normalize = True, normalize_features = True):
        """ Main CarDEC API the user can use to conduct batch correction and denoising experiments.


        Arguments:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.
        - preprocess: `bool`, If True, then preprocess the data.
        - weights_dir: `str`, the path in which to save the weights of the CarDEC model.
        - batch_key: `str`, string specifying the name of the column in the observation dataframe which identifies the batch of each cell. If this is left as None, then all cells are assumed to be from one batch.
        - n_high_var: `int`, integer specifying the number of genes to be idntified as highly variable. E.g. if n_high_var = 2000, then the 2000 genes with the highest variance are designated as highly variable.
        - LVG: `bool`, If True, also model LVGs. Otherwise, only model HVGs.
        - normalize_samples: `bool`, If True, normalize expression of each gene in each cell by the sum of expression counts in that cell.
        - log_normalize: `bool`, If True, log transform expression. I.e., compute log(expression + 1) for each gene, cell expression count.
        - normalize_features: `bool`, If True, z-score normalize each gene's expression.
        """
    
        if n_high_var is None:
            n_high_var = None
            LVG = False

        self.weights_dir = weights_dir
        self.LVG = LVG

        self.norm_args = (batch_key, n_high_var, LVG, normalize_samples, log_normalize, normalize_features)

        if preprocess:
            self.dataset = normalize_scanpy(adata, *self.norm_args)
        else:
            assert 'Variance Type' in adata.var.keys()
            assert 'normalized input' in adata.layers
            self.dataset = adata

        self.loaded = False
        self.count_loaded = False

    def build_model(self, load_fullmodel = True, dims = [128, 32], LVG_dims = [128, 32], tol = 0.005, n_clusters = None, 
                    random_seed = 201809, louvain_seed = 0, n_neighbors = 15, pretrain_epochs = 2000, batch_size_pretrain = 64,
                    act = 'relu', actincenter = "tanh", ae_lr = 1e-04, ae_decay_factor = 1/3, ae_patience_LR = 3, 
                    ae_patience_ES = 9, clust_weight = 1., load_encoder_weights = True):
        """ Initializes the main CarDEC model.


        Arguments:
        ------------------------------------------------------------------
        - load_fullmodel: `bool`, If True, the API will try to load the weights for the full model from the weight directory.
        - dims: `list`, the number of output features for each layer of the HVG encoder. The length of the list determines the number of layers.
        - LVG_dims: `list`, the number of output features for each layer of the LVG encoder. The length of the list determines the number of layers.
        - tol: `float`, stop criterion, clustering procedure will be stopped when the difference ratio between the current iteration and last iteration larger than tol.
        - n_clusters: `int`, The number of clusters into which cells will be grouped.
        - random_seed: `int`, The seed used for random weight intialization.
        - louvain_seed: `int`, The seed used for louvain clustering intialization.
        - n_neighbors: `int`, The number of neighbors used for building the graph needed for louvain clustering.
        - pretrain_epochs: `int`, The maximum number of epochs for pretraining the HVG autoencoder. In practice, early stopping criteria should stop training much earlier.
        - batch_size_pretrain: `int`, The batch size used for pretraining the HVG autoencoder.
        - act: `str`, The activation function used for the intermediate layers of CarDEC, other than the bottleneck layer.
        - actincenter: `str`, The activation function used for the bottleneck layer of CarDEC.
        - ae_lr: `float`, The learning rate for pretraining the HVG autoencoder.
        - ae_decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - ae_patience_LR: `int`, the number of epochs which the validation loss is allowed to increase before learning rate is decayed when pretraining the autoencoder.
        - ae_patience_ES: `int`, the number of epochs which the validation loss is allowed to increase before training is halted when pretraining the autoencoder.
        - clust_weight: `float`, a number between 0 and 2 qhich balances the clustering and reconstruction losses.
        - load_encoder_weights: `bool`, If True, the API will try to load the weights for the HVG encoder from the weight directory.
        """
        
        assert n_clusters is not None
        
        if 'normalized input' not in list(self.dataset.layers):
            self.dataset = normalize_scanpy(self.dataset, *self.norm_args)
        
        p = sum(self.dataset.var["Variance Type"] == 'HVG')
        self.dims = [p] + dims
        
        if self.LVG:
            LVG_p = sum(self.dataset.var["Variance Type"] == 'LVG')
            self.LVG_dims = [LVG_p] + LVG_dims
        else:
            self.LVG_dims = None
        
        self.load_fullmodel = load_fullmodel
        self.weights_exist = os.path.isfile("./" + self.weights_dir + "/tuned_CarDECweights.index")
        
        set_centroids = not (self.load_fullmodel and self.weights_exist)
        
        self.model = CarDEC_Model(self.dataset, self.dims, self.LVG_dims, tol, n_clusters, random_seed, louvain_seed, 
                                  n_neighbors, pretrain_epochs, batch_size_pretrain, ae_decay_factor, 
                                  ae_patience_LR, ae_patience_ES, act, actincenter, ae_lr, 
                                  clust_weight, load_encoder_weights, set_centroids, self.weights_dir)
        
    def make_inference(self, batch_size = 64, val_split = 0.1, lr = 1e-04, decay_factor = 1/3,
                       iteration_patience_LR = 3, iteration_patience_ES = 6, maxiter = 1e3, epochs_fit = 1, 
                       optimizer = Adam(), printperiter = None, denoise_all = True, denoise_list = None):
        """ This class method makes inference on the data (batch correction + denoising) with the main CarDEC model


        Arguments:
        ------------------------------------------------------------------
        - batch_size: `int`, The batch size used for training the full model.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - lr: `float`, The learning rate for training the full model.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - iteration_patience_LR: `int`, The number of iterations tolerated before decaying the learning rate during which the number of cells that change assignment is less than tol.
        - iteration_patience_ES: `int`, The number of iterations tolerated before stopping training during which the number of cells that change assignment is less than tol.
        - maxiter: `int`, The maximum number of iterations allowed to train the full model. In practice, the model will halt training long before hitting this limit.
        - epochs_fit: `int`, The number of epochs during which to fine-tune weights, before updating the target distribution.
        - optimizer: `tensorflow.python.keras.optimizer_v2`, An instance of a TensorFlow optimizer.
        - printperiter: `int`, Optional integer argument. If specified, denoised values will be returned every printperiter epochs, so that the user can evaluate the progress of denoising as training continues.
        - denoise_all: `bool`, If True, then denoised expression values are provided for all cells.
        - denoise_list: `list`, An optional list of cell names (as strings). If provided, denoised values will be computed only for cells in this list.

        Returns:
        ------------------------------------------------------------------
        - denoised: `pd.DataFrame`, (Optional) If denoise_list was specified, then this will be an array of denoised expression provided only for listed cells. If denoise_all was instead specified as True, then denoised expression for all cells will be added as a layer to adata.
        """

        if denoise_list is not None:
            denoise_all = False
            
        if not self.loaded:
            if self.load_fullmodel and self.weights_exist:
                self.dataset = self.model.reload_model(self.dataset, batch_size, denoise_all)

            elif not self.weights_exist:
                print("CarDEC Model Weights not detected. Training full model.\n")
                self.dataset = self.model.train(self.dataset, batch_size, val_split, lr, decay_factor,
                               iteration_patience_LR, iteration_patience_ES, maxiter,
                               epochs_fit, optimizer, printperiter, denoise_all)

            else:
                print("Training full model.\n")
                self.dataset = self.model.train(self.dataset, batch_size, val_split, lr, decay_factor, 
                                                iteration_patience_LR, iteration_patience_ES, 
                                                maxiter, epochs_fit, optimizer, printperiter, denoise_all)
            
            
            self.loaded = True
            
        elif denoise_all:
            self.dataset = self.model.make_outputs(self.dataset, batch_size, True)
            
        if denoise_list is not None:
            denoise_list = list(denoise_list)
            indices = [x in denoise_list for x in self.dataset.obs.index]
            denoised = DataFrame(np.zeros((len(denoise_list), self.dataset.shape[1]), dtype = 'float32'))
            denoised.index = self.dataset.obs.index[indices]
            denoised.columns = self.dataset.var.index
            
            
            if self.LVG:
                hvg_ds = tf.data.Dataset.from_tensor_slices(self.dataset.obsm["embedding"][indices])
                lvg_ds = tf.data.Dataset.from_tensor_slices(self.dataset.obsm["LVG embedding"][indices])
            
                input_ds = tf.data.Dataset.zip((hvg_ds, lvg_ds))
                input_ds = input_ds.batch(batch_size)

                start = 0     
                for x in input_ds:
                    denoised_batch = {'HVG_denoised': self.model.decoder(x[0]), 'LVG_denoised': self.model.decoderLVG(x[1])}
                    q_batch = self.model.clustering_layer(x[0])
                    end = start + q_batch.shape[0]

                    denoised.iloc[start:end, np.where(self.dataset.var['Variance Type'] == 'HVG')[0]] = denoised_batch['HVG_denoised'].numpy()
                    denoised.iloc[start:end, np.where(self.dataset.var['Variance Type'] == 'LVG')[0]] = denoised_batch['LVG_denoised'].numpy()

                    start = end

            else:
                input_ds = tf.data.Dataset.from_tensor_slices(self.dataset.obsm["embedding"])

                input_ds = input_ds.batch(batch_size)

                start = 0

                for x in input_ds:
                    denoised_batch = {'HVG_denoised': self.model.decoder(x)}
                    q_batch = self.model.clustering_layer(x)
                    end = start + q_batch.shape[0]

                    denoised.iloc[start:end] = denoised_batch['HVG_denoised'].numpy()

                    start = end
            
            return denoised
            
        print(" ")
            
    def model_counts(self, load_weights = True, act = 'relu', random_seed = 201809,
                     optimizer = Adam(), keep_dispersion = False, num_epochs = 2000, batch_size_count = 64,
                     val_split = 0.1, lr = 1e-03, decay_factor = 1/3, patience_LR = 3, patience_ES = 9, 
                     denoise_all = True, denoise_list = None):
        """ This class method makes inference on the data on the count scale.


        Arguments:
        ------------------------------------------------------------------
        - load_weights: `bool`, If true, the API will attempt to load the weights for the count model.
        - act: `str`, A string specifying the activation function for intermediate layers of the count models.
        - random_seed: `int`, A seed used for weight initialization.
        - optimizer: `tensorflow.python.keras.optimizer_v2`, An instance of a TensorFlow optimizer.
	- keep_dispersion: `bool`, If True, the gene, cell dispersions will be returned as well.
        - num_epochs: `int`, The maximum number of epochs allowed to train each count model. In practice, the model will halt
        training long before hitting this limit.
        - batch_size_count: `int`, The batch size used for training the count models.
        - val_split: `float`, The fraction of cells to be reserved for validation during this step.
        - lr: `float`, The learning rate for training the count models.
        - decay_factor: `float`, The multiplicative factor by which to decay the learning rate when validation loss is not decreasing.
        - patience_LR: `int`, The number of epochs tolerated before decaying the learning rate during which the validation loss does not decrease.
        - patience_ES: `int`, The number of iterations tolerated before stopping training during which the validation loss does not decrease.
        - denoise_all: `bool`, If True, then denoised expression values are provided for all cells.
        - denoise_list: `list`, An optional list of cell names (as strings). If provided, denoised values will be computed only for cells in this list.

        Returns:
        ------------------------------------------------------------------
        - denoised: `pd.DataFrame`, (Optional) If denoise_list was specified, then this will be an array of denoised expression on the count scale provided only for listed cells. If denoise_all was instead specified as True, then denoised expression for all cells will be added as a layer to adata.
        - denoised_dispersion: `pd.DataFrame`, (Optional) If denoise_list was specified and "keep_dispersion" was set to True, then this will be an array of dispersions from the fitted negative binomial model provided only for listed cells. If denoise_all was instead specified as False, but "keep_dispersion" was still True then dispersions for all cells will be added as a layer to adata.
        """
    
        if denoise_list is not None:
            denoise_all = False
        
        if not self.count_loaded:
            weights_dir = os.path.join(self.weights_dir, 'count weights')
            weight_files_exist = os.path.isfile(weights_dir + "/countmodel_weights_HVG Count.index")
            if self.LVG:
                weight_files_exist = weight_files_exist and os.path.isfile(weights_dir + "/countmodel_weights_LVG Count.index")

            init_args = (act, random_seed, self.model.splitseed, optimizer, weights_dir)
            train_args = (num_epochs, batch_size_count, val_split, lr, decay_factor, patience_LR, patience_ES)

            self.nbmodel = count_model(self.dims, *init_args, n_features = self.dims[-1], mode = 'HVG')

            if load_weights and weight_files_exist:
                print("Weight files for count models detected, loading weights.")
                self.nbmodel.load_model()

            elif load_weights:
                print("Weight files for count models not detected. Training HVG count model.\n")
                self.nbmodel.train(self.dataset, *train_args)

            else:
                print("Training HVG count model.\n")
                self.nbmodel.train(self.dataset, *train_args)

            if self.LVG:
                self.nbmodel_lvg = count_model(self.LVG_dims, *init_args, 
                    n_features = self.dims[-1] + self.LVG_dims[-1], mode = 'LVG')

                if load_weights and weight_files_exist:
                    self.nbmodel_lvg.load_model()
                    print("Count model weights loaded successfully.")

                elif load_weights:
                    print("\n \n \n")
                    print("Training LVG count model.\n")
                    self.nbmodel_lvg.train(self.dataset, *train_args)

                else:
                    print("\n \n \n")
                    print("Training LVG count model.\n")
                    self.nbmodel_lvg.train(self.dataset, *train_args)
            
            self.count_loaded = True
            
        if denoise_all:
            self.nbmodel.denoise(self.dataset, keep_dispersion, batch_size_count)
            if self.LVG:
                self.nbmodel_lvg.denoise(self.dataset, keep_dispersion, batch_size_count)
                
        elif denoise_list is not None:
            denoise_list = list(denoise_list)
            indices = [x in denoise_list for x in self.dataset.obs.index]
            denoised = DataFrame(np.zeros((len(denoise_list), self.dataset.shape[1]), dtype = 'float32'))
            denoised.index = self.dataset.obs.index[indices]
            denoised.columns = self.dataset.var.index
            if keep_dispersion:
                denoised_dispersion = DataFrame(np.zeros((len(denoise_list), self.dataset.shape[1]), dtype = 'float32'))
                denoised_dispersion.index = self.dataset.obs.index[indices]
                denoised_dispersion.columns = self.dataset.var.index
            
            input_ds_embed = tf.data.Dataset.from_tensor_slices(self.dataset.obsm['embedding'][indices])
            input_ds_sf = tf.data.Dataset.from_tensor_slices(self.dataset.obs['size factors'][indices])
            input_ds = tf.data.Dataset.zip((input_ds_embed, input_ds_sf))
            input_ds = input_ds.batch(batch_size_count)

            type_indices = np.where(self.dataset.var['Variance Type'] == 'HVG')[0]

            if not keep_dispersion:
                start = 0
                for x in input_ds:
                    end = start + x[0].shape[0]
                    denoised.iloc[start:end, type_indices] = self.nbmodel(*x)[0].numpy()
                    start = end

            else:
                start = 0
                for x in input_ds:
                    end = start + x[0].shape[0]
                    batch_output = self.nbmodel(*x)
                    denoised.iloc[start:end, type_indices] = batch_output[0].numpy()
                    denoised_dispersion.iloc[start:end, type_indices] = batch_output[1].numpy()
                    start = end
            
            if self.LVG:
                input_ds_embed = tf.data.Dataset.from_tensor_slices(self.dataset.obsm['LVG embedding'][indices])
                input_ds_sf = tf.data.Dataset.from_tensor_slices(self.dataset.obs['size factors'][indices])
                input_ds = tf.data.Dataset.zip((input_ds_embed, input_ds_sf))
                input_ds = input_ds.batch(batch_size_count)

                type_indices = np.where(self.dataset.var['Variance Type'] == 'LVG')[0]

                if not keep_dispersion:
                    start = 0
                    for x in input_ds:
                        end = start + x[0].shape[0]
                        denoised.iloc[start:end, type_indices] = self.nbmodel_lvg(*x)[0].numpy()
                        start = end

                else:
                    start = 0
                    for x in input_ds:
                        end = start + x[0].shape[0]
                        batch_output = self.nbmodel_lvg(*x)
                        denoised.iloc[start:end, type_indices] = batch_output[0].numpy()
                        denoised_dispersion.iloc[start:end, type_indices] = batch_output[1].numpy()
                        start = end
                        
            if not keep_dispersion:
                return denoised
            else:
                return denoised, denoised_dispersion

