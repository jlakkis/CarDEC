import tensorflow as tf
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, centroids = None, n_clusters = None, n_features = None, alpha=1.0, **kwargs):
        """ The clustering layer predicts the a cell's class membership probability for each cell.
        
        
        Arguments:
        ------------------------------------------------------------------
        - centroids: `tf.Tensor`, Initial cluster ceontroids after pretraining the model.
        - n_clusters: `int`, Number of clusters.
        - n_features: `int`, The number of features of the bottleneck embedding space that the centroids live in.
        - alpha: parameter in Student's t-distribution. Default to 1.0.
        """
        
        super(ClusteringLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.initial_centroids = centroids

        if centroids is not None:
            n_clusters, n_features = centroids.shape

        self.n_features, self.n_clusters = n_features, n_clusters

        assert self.n_clusters is not None
        assert self.n_features is not None

    def build(self, input_shape):
        """ This class method builds the layer fully once it receives an input tensor.
        
        
        Arguments:
        ------------------------------------------------------------------
        - input_shape: `list`, A list specifying the shape of the input tensor.
        """
        
        assert len(input_shape) == 2
        
        self.centroids = self.add_weight(name = 'clusters', shape = (self.n_clusters, self.n_features), initializer = 'glorot_uniform')
        if self.initial_centroids is not None:
            self.set_weights([self.initial_centroids])
            del self.initial_centroids
        
        self.built = True

    def call(self, x, **kwargs):
        """ Forward pass of the clustering layer,
        
        
        ***Inputs***:
            - x: `tf.Tensor`, the embedding tensor of shape = (n_obs, n_var)
        
        ***Returns***:
            - q: `tf.Tensor`, student's t-distribution, or soft labels for each sample of shape = (n_obs, n_clusters)
        """

        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(x, axis = 1) - self.centroids), axis = 2) / self.alpha))
        q = q**((self.alpha + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis = 1, keepdims = True)

        return q

    def compute_output_shape(self, input_shape):
        """ This method infers the output shape from the input shape.
        
        
        Arguments:
        ------------------------------------------------------------------
        - input_shape: `list`, A list specifying the shape of the input tensor.
        
        Returns:
        ------------------------------------------------------------------
        - output_shape: `list`, A tuple specifying the shape of the output for the minibatch (n_obs, n_clusters)
        """
        
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters