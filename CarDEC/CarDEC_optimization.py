import numpy as np

import tensorflow as tf
from tensorflow.keras.losses import KLD, MSE


def grad_MainModel(model, input_, target, target_p, total_loss, LVG_target = None, aeloss_fun = None, clust_weight = 1.):
    """Function to do a backprop update to the main CarDEC model for a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - model: `tensorflow.keras.Model`, The main CarDEC model.
    - input_: `list`, A list containing the input HVG and (optionally) LVG expression tensors of the minibatch for the CarDEC model.
    - target: `tf.Tensor`, Tensor containing the reconstruction target of the minibatch for the HVGs.
    - target_p: `tf.Tensor`, Tensor containing cluster membership probability targets for the minibatch.
    - total_loss: `function`, Function to compute the loss for the main CarDEC model for a minibatch.
    - LVG_target: `tf.Tensor` (Optional), Tensor containing the reconstruction target of the minibatch for the LVGs.
    - aeloss_fun: `function`, Function to compute reconstruction loss.
    - clust_weight: `float`, A float between 0 and 2 balancing clustering and reconstruction losses.
    
    Returns:
    ------------------------------------------------------------------
    - loss_value: `tf.Tensor`: The loss computed for the minibatch.
    - gradients: `a list of Tensors`: Gradients to update the model weights.
    """
    
    with tf.GradientTape() as tape:
        denoised_output, cluster_output = model(*input_)
        loss_value, aeloss = total_loss(target, denoised_output, target_p, cluster_output, 
                                LVG_target, aeloss_fun, clust_weight)
        
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def grad_reconstruction(model, input_, target, loss):
    """Function to compute gradient update for pretrained autoencoder only.
    
    
    Arguments:
    ------------------------------------------------------------------
    - model: `tensorflow.keras.Model`, The main CarDEC model.
    - input_: `list`, A list containing the input HVG expression tensor of the minibatch for the CarDEC model.
    - target: `tf.Tensor`, Tensor containing the reconstruction target of the minibatch for the HVGs.
    - loss: `function`, Function to compute reconstruction loss.
    
    Returns:
    ------------------------------------------------------------------
    - loss_value: `tf.Tensor`: The loss computed for the minibatch.
    - gradients: `a list of Tensors`: Gradients to update the model weights.
    """
    
    if type(input_) != tuple:
        input_ = (input_, )
        
    with tf.GradientTape() as tape:
        output = model(*input_)
        loss_value = loss(target, output)
        
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def total_loss(target, denoised_output, p, cluster_output_q, LVG_target = None, aeloss_fun = None, clust_weight = 1.):
    """Function to compute the loss for the main CarDEC model for a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - target: `tf.Tensor`, Tensor containing the reconstruction target of the minibatch for the HVGs.
    - denoised_output: `dict`, Dictionary containing the output tensors from the CarDEC main model's forward pass.
    - p: `tf.Tensor`, Tensor of shape (n_obs, n_cluster) containing cluster membership probability targets for the minibatch.
    - cluster_output_q: `tf.Tensor`, Tensor of shape (n_obs, n_cluster) containing predicted cluster membership probabilities
    for each cell.
    - LVG_target: `tf.Tensor` (Optional), Tensor containing the reconstruction target of the minibatch for the LVGs.
    - aeloss_fun: `function`, Function to compute reconstruction loss.
    - clust_weight: `float`, A float between 0 and 2 balancing clustering and reconstruction losses.
    
    Returns:
    ------------------------------------------------------------------
    - net_loss: `tf.Tensor`, The loss computed for the minibatch.
    - aeloss: `tf.Tensor`, The reconstruction loss computed for the minibatch.
    """

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


def MSEloss(netinput, netoutput):
    """Function to compute the MSEloss for the reconstruction loss of a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - netinput: `tf.Tensor`, Tensor containing the network reconstruction target of the minibatch for the cells.
    - netoutput: `tf.Tensor`, Tensor containing the reconstructed target of the minibatch for the cells.
    
    Returns:
    ------------------------------------------------------------------
    - mse_loss: `tf.Tensor`, The loss computed for the minibatch, averaged over genes and cells.
    """
    
    return tf.math.reduce_mean(MSE(netinput, netoutput))


def NBloss(count, output, eps = 1e-10, mean = True):
    """Function to compute the negative binomial reconstruction loss of a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - count: `tf.Tensor`, Tensor containing the network reconstruction target of the minibatch for the cells (the original 
    counts).
    - output: `tf.Tensor`, Tensor containing the reconstructed target of the minibatch for the cells.
    - eps: `float`, A small number introduced for computational stability
    - mean: `bool`, If True, average negative binomial loss over genes and cells
    
    Returns:
    ------------------------------------------------------------------
    - nbloss: `tf.Tensor`, The loss computed for the minibatch. If mean was True, it has shape (n_obs, n_var). Otherwise, it has shape (1,).
    """
    
    count = tf.cast(count, tf.float32)
    mu = tf.cast(output[0], tf.float32)

    theta = tf.minimum(output[1], 1e6)

    t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(count + 1.0) - tf.math.lgamma(count + theta + eps)
    t2 = (theta + count) * tf.math.log(1.0 + (mu/(theta+eps))) + (count * (tf.math.log(theta + eps) - tf.math.log(mu + eps)))

    final = _nan2inf(t1 + t2)
    
    if mean:
        final = tf.reduce_sum(final)/final.shape[0]/final.shape[1]

    return final


def ZINBloss(count, output, eps = 1e-10):
    """Function to compute the negative binomial reconstruction loss of a minibatch.
    
    
    Arguments:
    ------------------------------------------------------------------
    - count: `tf.Tensor`, Tensor containing the network reconstruction target of the minibatch for the cells (the original counts).
    - output: `tf.Tensor`, Tensor containing the reconstructed target of the minibatch for the cells.
    - eps: `float`, A small number introduced for computational stability
    
    Returns:
    ------------------------------------------------------------------
    - zinbloss: `tf.Tensor`, The loss computed for the minibatch. Has shape (1,).
    """
    
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


def _nan2inf(x):
    """Function to replace nan entries in a Tensor with infinities.
    
    
    Arguments:
    ------------------------------------------------------------------
    - x: `tf.Tensor`, Tensor of arbitrary shape.
    
    Returns:
    ------------------------------------------------------------------
    - x': `tf.Tensor`, Tensor x with nan entries replaced by infinity.
    """
    
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)

