import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from keras.initializers import glorot_normal


def gaussian_loss(y_true, y_pred):
    return tf.reduce_mean(0.5*tf.math.log(sigma) + 0.5*tf.divide(tf.square(y_true - y_pred), sigma)) + 1e-6
def custom_loss(sigma):
    def gaussian_loss(y_true, y_pred):
        return tf.reduce_mean(0.5*tf.math.log(sigma) + 0.5*tf.divide(tf.square(y_true - y_pred), sigma)) + 1e-6
    return gaussian_loss
class GaussianLayer(Layer):    
    def __init__(self, output_dim=30,hardMax=False, **kwargs):
        self.output_dim = output_dim
        self.hardMax=hardMax
        super(GaussianLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel_1 = self.add_weight(name='kernel_1', 
                                      shape=(256, self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2', 
                                      shape=(256, self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        super(GaussianLayer, self).build(input_shape)
    def call(self, x):
        output_mu  = K.dot(x, self.kernel_1) + self.bias_1
        if self.hardMax==True:
            output_mu=K.relu(output_mu)
            output_mu=output_mu/K.sum(output_mu,axis=-1,keepdims=True)#
            #output_mu=K.softmax(output_mu)
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
        return [output_mu, output_sig_pos]
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]