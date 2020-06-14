#!/usr/bin/env python
# coding: utf-8

# In[5]:


from .CarDEC_optimization import grad_reconstruction as grad, NBloss
from .CarDEC_utils import build_dir

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import exp as tf_exp, set_floatx
from time import time

import random
import numpy as np
from scipy.stats import zscore
import os


# In[6]:


set_floatx('float32')


# In[17]:


class count_model(Model):
    """ 
    Stacked autoencoders. It can be trained in layer-wise manner followed by end-to-end fine-tuning.
    For a 5-layer (including input layer) example:
        Autoendoers model: Input -> encoder_0->act -> encoder_1 -> decoder_1->act -> decoder_0;
        stack_0 model: Input->dropout -> encoder_0->act->dropout -> decoder_0;
        stack_1 model: encoder_0->act->dropout -> encoder_1->dropout -> decoder_1->act;

    Usage:
        from SAE import SAE
        sae = SAE(dims=[784, 500, 10])  # define a SAE with 5 layers
        sae.fit(x, epochs=100)
        features = sae.extract_feature(x)

    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
              The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation (default='relu'), not applied to Input, Hidden and Output layers.
        batch_size: batch size
        loss: The loss function to use. One of "NB-FixedDisp", "NB", "ZINB-FixedDisp", "ZINB", or "MSE"
    """
    def __init__(self, dims, act = 'relu', random_seed = 201809, splitseed = 215, optimizer = Adam(),
             weights_dir = 'CarDEC Count Weights', n_features = 32, mode = 'HVG'):
        super(count_model, self).__init__()
        
        tf.keras.backend.clear_session()
        
        self.mode = mode
        self.name_ = mode + " Count"
        
        if mode == 'HVG':
            self.embed_name = 'embedding'
        else:
            self.embed_name = 'LVG embedding'
        
        self.weights_dir = weights_dir
        
        self.dims = dims
        n_stacks = len(dims) - 1
        
        self.optimizer = optimizer
        self.random_seed = random_seed
        self.splitseed = splitseed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        self.activation = act
        self.MeanAct = lambda x: tf.clip_by_value(tf_exp(x), 1e-5, 1e6)
        self.DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
        
        model_layers = []
        for i in range(n_stacks - 1, 0, -1):
            model_layers.append(Dense(dims[i], kernel_initializer = "glorot_uniform", activation = self.activation
                                        , name='base%d' % (i-1)))
        self.base = Sequential(model_layers, name = 'base')

        self.mean_layer = Dense(dims[0], activation = self.MeanAct, name='mean')
        self.disp_layer = Dense(dims[0], activation = self.DispAct, name='dispersion')

        self.rescale = Lambda(lambda l: tf.matmul(tf.linalg.diag(l[0]), l[1]), name = 'sf scaling')
        
        build_dir(self.weights_dir)
        
        self.construct(n_features, self.name_)
        
    def call(self, x):
        c = x[0]
        s = x[1]
        c = self.base(c)
        
        disp = self.disp_layer(c)
        mean = self.mean_layer(c)
        mean = self.rescale([s, mean])
                        
        return mean, disp
        
    def load_model(self, ):
        
        tf.keras.backend.clear_session()
        
        self.load_weights(os.path.join(self.weights_dir, "countmodel_weights_" + self.name_)).expect_partial()
        
    def construct(self, n_features, name, summarize = False):

        x = [tf.zeros(shape = (1, n_features), dtype='float32'), tf.ones(shape = (1,), dtype='float32')]
        out = self(x)
        
        if summarize:
            print("----------Count Model " + name + " Architecture----------")
            self.summary()

            print("\n----------Base Sub-Architecture----------")
            self.base.summary()
        
    def denoise(self, adata, keep_dispersion = False, batch_size = 64):
        input_ds_embed = tf.data.Dataset.from_tensor_slices(adata.obsm[self.embed_name])
        input_ds_sf = tf.data.Dataset.from_tensor_slices(adata.obs['size factors'])
        input_ds = tf.data.Dataset.zip((input_ds_embed, input_ds_sf))
        input_ds = input_ds.batch(batch_size)
        
        if "denoised counts" not in list(adata.layers):
            adata.layers["denoised counts"] = np.zeros(adata.shape, dtype = 'float32')
        
        type_indices = adata.var['Variance Type'] == self.mode
        
        if not keep_dispersion:
            start = 0
            for x in input_ds:
                end = start + x[0].shape[0]
                adata.layers["denoised counts"][start:end, type_indices] = self(x)[0].numpy()
                start = end
                
        else:
            if "dispersion" not in list(adata.layers):
                adata.layers["dispersion"] = np.zeros(adata.shape, dtype = 'float32')
                
            start = 0
            for x in input_ds:
                end = start + x[0].shape[0]
                batch_output = self(x)
                adata.layers["denoised counts"][start:end, type_indices] = batch_output[0].numpy()
                adata.layers["dispersion"][start:end, type_indices] = batch_output[1].numpy()
                start = end
            
    def makegenerators(self, adata, val_split, batch_size, splitseed):
        n, num_features = adata.X.shape
        
        Xobs_target_ds = tf.data.Dataset.from_tensor_slices(adata.X[:, adata.var['Variance Type'] == self.mode])
        Xobs_embed = tf.data.Dataset.from_tensor_slices(adata.obsm[self.embed_name])
        size_factors_ds = tf.data.Dataset.from_tensor_slices(adata.obs['size factors'])
        
        train_dataset = tf.data.Dataset.zip((Xobs_embed, size_factors_ds))
        full_train_dataset = tf.data.Dataset.zip((train_dataset, Xobs_target_ds))
        
        tf.random.set_seed(splitseed) #Set the seed so we get same validation split always
        full_train_dataset = full_train_dataset.shuffle(n, reshuffle_each_iteration = False)
        
        #tf.random.set_seed(self.random_seed)
        
        train_dataset = full_train_dataset.skip(round(val_split * n))
        val_dataset = full_train_dataset.take(round(val_split * n)) 
        
        train_dataset = train_dataset.shuffle(n - round(val_split * n))
        train_dataset = train_dataset.batch(batch_size)
        
        val_dataset = val_dataset.shuffle(round(val_split * n))
        val_dataset = val_dataset.batch(batch_size)
        
        return train_dataset, val_dataset
    
    def train(self, adata, num_epochs = 2000, batch_size = 64, val_split = 0.1, lr = 1e-03, decay_factor = 1/3,
              patience_LR = 3, patience_ES = 9):
        
        tf.keras.backend.clear_session()
                
        loss = NBloss
        
        train_dataset, val_dataset = self.makegenerators(adata, val_split = 0.1, 
                                                         batch_size = batch_size, splitseed = self.splitseed)
        
        counter_LR = 0
        counter_ES = 0
        best_loss = np.inf
        
        self.optimizer.lr = lr
        
        total_start = time()
        
        for epoch in range(num_epochs):
            epoch_start = time()
            
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_loss_avg_val = tf.keras.metrics.Mean()
            
            # Training loop - using batches of batch_size
            for x, target in train_dataset:
                loss_value, grads = grad(self, x, target, loss)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                epoch_loss_avg(loss_value)  # Add current batch loss
            
            # Validation Loop
            for x, target in val_dataset:
                output = self(x)
                loss_value = loss(target, output)
                epoch_loss_avg_val(loss_value)
            
            current_loss_val = epoch_loss_avg_val.result()

            epoch_time = round(time() - epoch_start, 1)
            
            print("Epoch {:03d}: Training Loss: {:.3f}, Validation Loss: {:.3f}, Time: {:.1f} s".format(epoch, epoch_loss_avg.result().numpy(), epoch_loss_avg_val.result().numpy(), epoch_time))
            
            if(current_loss_val + 10**(-3) < best_loss):
                counter_LR = 0
                counter_ES = 0
                best_loss = current_loss_val
            else:
                counter_LR = counter_LR + 1
                counter_ES = counter_ES + 1

            if patience_ES <= counter_ES:
                break

            if patience_LR <= counter_LR:
                self.optimizer.lr = self.optimizer.lr * decay_factor
                counter_LR = 0
                print("\nDecaying Learning Rate to: " + str(self.optimizer.lr.numpy()))
                
            # End epoch
        
        total_time = round(time() - total_start, 2)
        
        if not os.path.isdir("./" + self.weights_dir):
            os.mkdir("./" + self.weights_dir)
        
        self.save_weights(os.path.join(self.weights_dir, "countmodel_weights_" + self.name_), save_format='tf')
                
        print('\nTraining Completed')
        print("Total training time: " + str(total_time) + " seconds")

