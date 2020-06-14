#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .CarDEC_optimization import grad_reconstruction as grad, MSEloss

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import set_floatx
from time import time

import random
import numpy as np
from scipy.stats import zscore
import os


# In[ ]:


set_floatx('float32')


# In[ ]:


class SAE(Model):
    """ 
    Stacked autoencoders. It can be trained in layer-wise manner followed by end-to-end fine-tuning.
    For a 5-layer (including input layer) example:
        Autoendoers model: Input -> encoder_0->act -> encoder_1 -> decoder_1->act -> decoder_0;
        stack_0 model: Input -> encoder_0->act -> decoder_0;
        stack_1 model: encoder_0->act -> encoder_1 -> decoder_1->act;
    
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
    """

    def __init__(self, dims, act = 'relu', actincenter = "tanh", 
                 random_seed = 201809, splitseed = 215, init="glorot_uniform", optimizer = Adam(),
                 weights_dir = 'CarDEC Weights'):
        super(SAE, self).__init__()
        
        tf.keras.backend.clear_session()
        
        self.weights_dir = weights_dir
        
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.init = init
        self.optimizer = optimizer
        self.random_seed = random_seed
        self.splitseed = splitseed
        
        self.activation = act
        self.actincenter = actincenter #hidden layer activation function
        
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
            
        encoder_layers = []
        for i in range(self.n_stacks-1):
            encoder_layers.append(Dense(self.dims[i + 1], kernel_initializer = self.init, activation = self.activation, name='encoder_%d' % i))
                
        encoder_layers.append(Dense(self.dims[-1], kernel_initializer=self.init, activation=self.actincenter, name='embedding'))
        self.encoder = Sequential(encoder_layers, name = 'encoder')

        decoder_layers = []
        for i in range(self.n_stacks - 1, 0, -1):
            decoder_layers.append(Dense(self.dims[i], kernel_initializer = self.init, activation = self.activation
                                        , name = 'decoder%d' % (i-1)))
            
        decoder_layers.append(Dense(self.dims[0], activation = 'linear', name='output'))
        
        self.decoder = Sequential(decoder_layers, name = 'decoder')
        
        self.construct()

    def call(self, x):
                
        c = self.encoder(x)

        output = self.decoder(c)
                    
        return output
    
    def load_encoder(self, random_seed = 2312):
        
        tf.keras.backend.clear_session()
        
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
     
        self.encoder.load_weights("./" + self.weights_dir + "/pretrained_encoder_weights").expect_partial()
        
        decoder_layers = []
        for i in range(self.n_stacks - 1, 0, -1):
            decoder_layers.append(Dense(self.dims[i], kernel_initializer = self.init, activation = self.activation
                                        , name='decoder%d' % (i-1)))
        self.decoder_base = Sequential(decoder_layers, name = 'decoderbase')
        
        self.output_layer = Dense(self.dims[0], activation = 'linear', name='output')
            
        self.construct(summarize = False)
        
    def load_autoencoder(self, ):
        
        tf.keras.backend.clear_session()
        
        self.load_weights("./" + self.weights_dir + "/pretrained_autoencoder_weights").expect_partial()
        
    def construct(self, summarize = False):

        x = tf.zeros(shape = (1, self.dims[0]), dtype=float)
        out = self(x)
        
        if summarize:
            print("----------Autoencoder Architecture----------")
            self.summary()

            print("\n----------Encoder Sub-Architecture----------")
            self.encoder.summary()

            print("\n----------Base Decoder Sub-Architecture----------")
            self.decoder.summary()
        
    def denoise(self, adata, batch_size = 64):
        input_ds = tf.data.Dataset.from_tensor_slices(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'])
        input_ds = input_ds.batch(batch_size)
        
        output = np.zeros((adata.shape[0], self.dims[0]), dtype = 'float32')
        start = 0
        
        for x in input_ds:
            end = start + x.shape[0]
            output[start:end] = self(x).numpy()
            start = end
        
        return output
        
    def embed(self, adata, batch_size = 64):
        input_ds = tf.data.Dataset.from_tensor_slices(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'])
        input_ds = input_ds.batch(batch_size)
        
        embedding = np.zeros((adata.shape[0], self.dims[-1]), dtype = 'float32')
        start = 0

        for x in input_ds:
            end = start + x.shape[0]
            embedding[start:end] = self.encoder(x).numpy()
            start = end
            
        return embedding
    
    def makegenerators(self, adata, val_split, batch_size, splitseed):
        n, num_features = adata.X.shape
                
        Xobs_target_ds = tf.data.Dataset.from_tensor_slices(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'])
        train_dataset = tf.data.Dataset.from_tensor_slices(adata.layers["normalized input"][:, adata.var['Variance Type'] == 'HVG'])
        
        full_train_dataset = tf.data.Dataset.zip((train_dataset, Xobs_target_ds))
        
        tf.random.set_seed(splitseed) #Set the seed so we get same validation split always
        full_train_dataset = full_train_dataset.shuffle(n, reshuffle_each_iteration = False)
                
        train_dataset = full_train_dataset.skip(round(val_split * n))
        val_dataset = full_train_dataset.take(round(val_split * n)) 
        
        train_dataset = train_dataset.shuffle(n - round(val_split * n))
        train_dataset = train_dataset.batch(batch_size)
        
        val_dataset = val_dataset.shuffle(round(val_split * n))
        val_dataset = val_dataset.batch(batch_size)
        
        return train_dataset, val_dataset
    
    def train(self, adata, num_epochs = 2000, batch_size = 64, val_split = 0.1, lr = 1e-03, decay_factor = 1/3,
              patience_LR = 3, patience_ES = 9, save_fullmodel = True):
        
        tf.keras.backend.clear_session()
        
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
                loss_value, grads = grad(self, x, target, MSEloss)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                epoch_loss_avg(loss_value)  # Add current batch loss
            
            # Validation Loop
            for x, target in val_dataset:
                output = self(x)
                loss_value = MSEloss(target, output)
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
        
        self.save_weights("./" + self.weights_dir + "/pretrained_autoencoder_weights", save_format='tf')
        self.encoder.save_weights("./" + self.weights_dir + "/pretrained_encoder_weights", save_format='tf')
        
        print('\nTraining Completed')
        print("Total training time: " + str(total_time) + " seconds")

