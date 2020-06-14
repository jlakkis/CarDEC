#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import tensorflow as tf
from tensorflow.keras.losses import KLD, MSE


# In[ ]:


def grad_MainModel(model, input_, target, target_p, total_loss, LVG_target = None, aeloss_fun = None, clust_weight = 1.):
    with tf.GradientTape() as tape:
        denoised_output, cluster_output = model(input_)
        loss_value, aeloss = total_loss(target, denoised_output, target_p, cluster_output, 
                                LVG_target, aeloss_fun, clust_weight)
        
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# In[ ]:


def grad_reconstruction(model, input_, target, loss):
    with tf.GradientTape() as tape:
        output = model(input_)
        loss_value = loss(target, output)
        
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# In[ ]:


def total_loss(target, denoised_output, p, cluster_output_q, LVG_target = None, aeloss_fun = None, clust_weight = 1.):
    if aeloss_fun is not None:
        
        aeloss_HVG = aeloss_fun(target, denoised_output['HVG_denoised'])
        if LVG_target is not None:
            aeloss_LVG = aeloss_fun(LVG_target, denoised_output['LVG_denoised'])
            aeloss = 0.5*(aeloss_LVG + aeloss_HVG)
        else:
            aeloss = 1. * aeloss_HVG
    else:
        aeloss = 0.
    
    net_loss = clust_weight * tf.reduce_mean(KLD(p, cluster_output_q)) + (2. - clust_weight) * aeloss
    
    return net_loss, aeloss


# In[ ]:


def MSEloss(netinput, netoutput):
    return tf.math.reduce_mean(MSE(netinput, netoutput))


# In[ ]:


def normal_loss(scores, output, eps = 1e-10):
    return tf.reduce_sum(tf.math.log(output[1] + eps) + tf.math.square(scores - output[0])/(output[1] + eps))/output[0].shape[0]/output[0].shape[1]


# In[ ]:


def NBloss(count, output, eps = 1e-10, mean = True):
    count = tf.cast(count, tf.float32)
    mu = tf.cast(output[0], tf.float32)

    theta = tf.minimum(output[1], 1e6)

    t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(count + 1.0) - tf.math.lgamma(count + theta + eps)
    t2 = (theta + count) * tf.math.log(1.0 + (mu/(theta+eps))) + (count * (tf.math.log(theta + eps) - tf.math.log(mu + eps)))

    final = _nan2inf(t1 + t2)
    
    if mean:
        final = tf.reduce_sum(final)/final.shape[0]/final.shape[1]

    return final


# In[ ]:


def ZINBloss(count, output, eps = 1e-10):
    mu = output[0]
    theta = output[1]
    pi = output[2]
    
    NB = NBloss(count, output, eps = eps, mean = False) - tf.math.log(1.0 - pi + eps)
    
    count = tf.cast(count, tf.float32)
    mu = tf.cast(mu, tf.float32)
    
    theta = tf.math.minimum(theta, 1e6)
    
    zero_nb = tf.math.pow(theta/(theta + mu + eps), theta)
    zero_case = -tf.math.log(pi + ((1.0- pi) * zero_nb) + eps)
    final = tf.where(tf.less(count, 1e-8), zero_case, NB)
    
    final = tf.reduce_sum(final)/final.shape[0]/final.shape[1]
            
    return final


# In[ ]:


def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)


# In[4]:


p = [[0.1,0.3,0.6], [0.1,0.3,0.6]]
q = [[0.2,0.28, 0.52], [0.1,0.3,0.6]]
tf.reduce_mean(KLD(p,q))


# In[ ]:




