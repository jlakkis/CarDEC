#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Layer


# In[ ]:


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        centroids: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, centroids = None, n_clusters = None, n_features = None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.initial_centroids = centroids

        if centroids is not None:
            n_clusters, n_features = centroids.shape

        self.n_features, self.n_clusters = n_features, n_clusters

        assert self.n_clusters is not None
        assert self.n_features is not None

    def build(self, input_shape):
        assert len(input_shape) == 2
        
        self.centroids = self.add_weight(name = 'clusters', shape = (self.n_clusters, self.n_features), initializer = 'glorot_uniform')
        if self.initial_centroids is not None:
            self.set_weights([self.initial_centroids])
            del self.initial_centroids
        
        self.built = True

    def call(self, x, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """

        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(x, axis = 1) - self.centroids), axis = 2) / self.alpha))
        q = q**((self.alpha + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis = 1, keepdims = True)

        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

