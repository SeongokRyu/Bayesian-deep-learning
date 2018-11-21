import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras import layers

# Copyright to https://github.com/yaringal/ConcreteDropout
# Yarin Gal, Jiri Hron, Alex Kendall, Concrete Dropout, NIPS 2017
class ConcreteDropout(layers.Wrapper):
    def __init__(self,
                 layer,
                 weight_regularizer=1e-06,
                 dropout_regularizer=1e-05,
                 init_min=0.1,
                 init_max=0.1, 
                 **kwargs
                 ):

        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))

    def build(self, input_shape=None):
        self.input_sepc = layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

        # Initializer_p
        self.p_logit = tf.Variable(expected_shape=None,
                                   initial_value=tf.random_uniform((1,), self.init_min, self.init_max),
                                   dtype=tf.float32,
                                   trainable=True)
        """
        self.p_logit = self.add_variable(name='p_logit', 
                                         shape=None,
                                         initializer=tf.random_uniform((1,), self.init_min, self.init_max),
                                         dtype=tf.float32,
                                         trainable=True)
        """
        self.p = tf.nn.sigmoid(self.p_logit[0])
        tf.add_to_collection("LAYER_P", self.p)

        # Initialize regularizer / KL-prior (for KL-condition)
        input_dim = int(np.prod(input_shape[1:]))

        weight=self.layer.kernel
        kr = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) * (1. - self.p)
        #kr = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - self.p)
        dr = self.p * tf.log(self.p) + (1. - self.p) * tf.log(1. - self.p)
        dr *= self.dropout_regularizer * input_dim

        rr = tf.reduce_sum(kr + dr)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, rr)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        eps = 1e-07
        temp = 0.1

        unif_noise = tf.random_uniform(shape=tf.shape(x))
        drop_prob = ( tf.log(self.p + eps) - tf.log(1. - self.p + eps) + tf.log(unif_noise + eps) - tf.log(1. - unif_noise + eps) )
        drop_prob = tf.nn.sigmoid(drop_prob/temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob

        return x

    def call(self, inputs, training=None):
        if training:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            return self.layer.call(inputs)
