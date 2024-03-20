# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import numpy as np

def squash(s):
    """
    Squash activation function presented in 'Dynamic routing between capsules'.
    """
    squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + tf.keras.backend.epsilon())
    return scale * s


class PrimaryCaps(tf.keras.layers.Layer):
    """
    Create a primary capsule layer with the methodology described in 'Dynamic routing between capsules'.
    ...
    
    Attributes
    ----------
    C: int
        number of primary capsules
    L: int
        primary capsules dimension (number of properties)
    k: int
        kernel dimension
    s: int
        conv stride
 
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, C, L, k, s, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.C = C
        self.L = L
        self.k = k
        self.s = s
        
    def build(self, input_shape):    
        self.kernel = self.add_weight(shape=(self.k, self.k, input_shape[-1], self.C*self.L), initializer='glorot_uniform', name='kernel')
        self.biases = self.add_weight(shape=(self.C,self.L), initializer='zeros', name='biases')
        self.built = True
    
    def call(self, input):
        data_size = int(input.get_shape()[2])
        batch_size = int(input.get_shape()[0])    
        pose = tf.conv2d_backprop_input_v2(output, num_outputs=self.C * 16,
                               kernel_size=[1, 1], stride=1, padding='VALID', activation_fn=None)
        activation = tf.conv2D(output, num_outputs=self.C, kernel_size=[
                                    1, 1], stride=1, padding='VALID', activation_fn=tf.nn.sigmoid)
        pose = tf.reshape(pose, shape=[batch_size, data_size, data_size, self.C, 16])
        activation = tf.reshape(
            activation, shape=[batch_size, data_size, data_size, self.C, 1])
        output = tf.concat([pose, activation], axis=4)
        output = tf.reshape(output, shape=[batch_size, data_size, data_size, -1])
        return output
    
    def compute_output_shape(self, input_shape):
        H,W = input_shape.shape[1:3]
        return (None, (H - self.k)/self.s + 1, (W - self.k)/self.s + 1, self.C, self.L)

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'k': self.k,
            's': self.s
        }
        base_config = super(PrimaryCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DigitCaps(tf.keras.layers.Layer):
    """
    Create a digitcaps layer as described in 'Dynamic routing between capsules'. 
    
    ...
    
    Attributes
    ----------
    C: int
        number of primary capsules
    L: int
        primary capsules dimension (number of properties)
    routing: int
        number of routing iterations
    kernel_initializer:
        matrix W kernel initializer
 
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, C, L, routing=3, kernel_initializer='glorot_uniform', **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.C = C
        self.L = L
        self.iter_routing = routing
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have shape=[None,H,W,input_C,input_L]"
        H = input_shape[-4]
        W = input_shape[-3]
        input_C = input_shape[-2]
        input_L = input_shape[-1]

        self.W = self.add_weight(shape=[H*W*input_C, input_L, self.L*self.C], initializer=self.kernel_initializer, name='W')
        self.biases = self.add_weight(shape=[self.C,self.L], initializer='zeros', name='biases')
        self.routing_layer = VariationalBayesRouting2D(in_caps=input_C, out_caps=self.C, pose_dim=self.L,
                                                       kernel_size=H, stride=W, alpha0=1.0, m0=0.0,
                                                       kappa0=1.0, Psi0=1.0, nu0=self.L+2,
                                                       cov='diag', iter=self.routing) if self.routing else None
        self.built = True
    
    def call(self, inputs):
        miu, activation_out, test = self.em_routing(inputs, [1], self.C, None,)
        print(miu, activation_out, test)
        v = miu
        return v

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'routing': self.routing
        }
        base_config = super(DigitCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    

    def em_routing(self, votes, activation, caps_num_c, regularizer, tag=False):
        test = []

        batch_size = int(votes.get_shape()[0])
        caps_num_i = int(activation.get_shape()[-2])
        n_channels = int(votes.get_shape()[-1])

        sigma_square = []
        miu = []
        activation_out = []
        beta_v = slim.variable('beta_v', shape=[caps_num_c, n_channels], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                            regularizer=regularizer)
        beta_a = slim.variable('beta_a', shape=[caps_num_c], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                            regularizer=regularizer)

        # votes_in = tf.stop_gradient(votes, name='stop_gradient_votes')
        # activation_in = tf.stop_gradient(activation, name='stop_gradient_activation')
        votes_in = votes
        activation_in = activation

        for iters in range(self.iter_routing):
            # if iters == cfg.iter_routing-1:

            # e-step
            if iters == 0:
                r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
            else:
                # Contributor: Yunzhi Shi
                # log and exp here provide higher numerical stability especially for bigger number of iterations
                log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - \
                            (tf.square(votes_in - miu) / (2 * sigma_square))
                log_p_c_h = log_p_c_h - \
                            (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
                p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))

                ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])

                # ap = tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])

                r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + 1e-9)

            # m-step
            r = r * activation_in
            r = r / (tf.reduce_sum(r, axis=2, keep_dims=True)+ 1e-9)

            r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
            r1 = tf.reshape(r / (r_sum + 1e-9),
                            shape=[batch_size, caps_num_i, caps_num_c, 1])

            miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True)
            sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
                                        axis=1, keep_dims=True) + 1e-9

            if iters == self.iter_routing-1:
                r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
                cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                                                            shape=[batch_size, caps_num_c, n_channels])))) * r_sum

                activation_out = tf.nn.softmax(0.01 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
            else:
                activation_out = tf.nn.softmax(r_sum)
            # if iters <= cfg.iter_routing-1:
            #     activation_out = tf.stop_gradient(activation_out, name='stop_gradient_activation')

        return miu, activation_out, test
    
class VariationalBayesRouting2D(tf.keras.layers.Layer):
    '''Variational Bayes Capsule Routing Layer'''
    def __init__(self, in_caps, out_caps, pose_dim, kernel_size, stride,
                 alpha0, m0, kappa0, Psi0, nu0, cov='diag', iter=3, class_caps=False):
        super(VariationalBayesRouting2D, self).__init__()

        self.B = in_caps
        self.C = out_caps
        self.P = pose_dim
        self.D = max(2, self.P * self.P)
        self.K = kernel_size
        self.S = stride

        self.cov = cov
        self.iter = iter
        self.class_caps = class_caps
        self.n_classes = out_caps if class_caps else None

        self.alpha0 = tf.convert_to_tensor(alpha0, dtype=tf.float32)
        self.m0 = tf.convert_to_tensor(m0, dtype=tf.float32)
        self.kappa0 = tf.convert_to_tensor(kappa0, dtype=tf.float32)
        self.Psi0 = tf.convert_to_tensor(Psi0, dtype=tf.float32)
        self.nu0 = tf.convert_to_tensor(nu0, dtype=tf.float32)

        self.filter = tf.eye(self.K * self.K)

        self.BN_v = tf.keras.layers.BatchNormalization()
        self.BN_a = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        a_i, V_ji = inputs

        self.F_i = tf.shape(a_i)[-2:]  # input capsule (B) votes feature map size (K)
        self.F_o = tf.shape(a_i)[-4:-2]  # output capsule (C) feature map size (F)
        self.N = self.B * self.F_i[0] * self.F_i[1]  # total num of lower level capsules

        R_ij = (1. / self.C) * tf.ones((1, self.B, self.C, 1, 1, 1, 1, 1), dtype=tf.float32)

        for i in range(self.iter):  # routing iters
            self.update_qparam(a_i, V_ji, R_ij)

            if i != self.iter - 1:  # skip last iter
                R_ij = self.update_qlatent(a_i, V_ji)

        a_j, m_j = self.compute_output(a_i, V_ji)

        return a_j, m_j

    def update_qparam(self, a_i, V_ji, R_ij):
        R_ij *= a_i

        self.R_j = tf.reduce_sum(R_ij, axis=(1, -2, -1), keepdims=True)

        self.alpha_j = self.alpha0 + self.R_j
        self.kappa_j = self.kappa0 + self.R_j
        self.nu_j = self.nu0 + self.R_j

        mu_j = (1. / self.R_j) * tf.reduce_sum(R_ij * V_ji, axis=(1, -2, -1), keepdims=True)

        self.m_j = (1. / self.kappa_j) * self.R_j * mu_j

        if self.cov == 'diag':
            sigma_j = tf.reduce_sum(R_ij * tf.square(V_ji - mu_j), axis=(1, -2, -1), keepdims=True)
            self.invPsi_j = self.Psi0 + sigma_j + (self.R_j / self.kappa_j) * tf.square(mu_j)
            self.lndet_Psi_j = -tf.reduce_sum(tf.math.log(self.invPsi_j), axis=(1, -2, -1), keepdims=True)
        elif self.cov == 'full':
            Vm_j = V_ji - self.m_j
            sigma_j = tf.reduce_sum(R_ij * tf.matmul(Vm_j, tf.transpose(Vm_j, perm=[0, 1, 2, 5, 6, 3, 4])), axis=(1, -2, -1), keepdims=True)
            self.invPsi_j = self.Psi0 + sigma_j + (self.kappa0 * self.R_j / self.kappa_j) * tf.matmul(mu_j - self.m0, tf.transpose(mu_j - self.m0, perm=[0, 1, 2, 5, 6, 3, 4]))
            self.lndet_Psi_j = -2 * tf.linalg.cholesky(self.invPsi_j).diagonal(dim1=-2, dim2=-1).log().sum(axis=(1, -2, -1), keepdims=True)

    def update_qlatent(self, a_i, V_ji):
        Elnpi_j = tf.math.digamma(self.alpha_j) - tf.math.digamma(tf.reduce_sum(self.alpha_j, axis=2, keepdims=True))
        Elnlambda_j = self.D * 0.5 * tf.math.log(self.nu_j) + self.D * 0.5 - tf.math.lgamma(0.5 * (self.nu_j - tf.range(self.D, dtype=tf.float32))) + self.lndet_Psi_j

        if self.cov == 'diag':
            ElnQ = (self.D / self.kappa_j) + self.nu_j * tf.reduce_sum((1. / self.invPsi_j) * tf.square(V_ji - self.m_j), axis=(1, -2, -1), keepdims=True)
        elif self.cov == 'full':
            Vm_j = V_ji - self.m_j
            ElnQ = (self.D / self.kappa_j) + self.nu_j * tf.reduce_sum(tf.matmul(Vm_j, tf.linalg.inv(self.invPsi_j)) * Vm_j, axis=(1, -2, -1), keepdims=True)

        lnp_j = 0.5 * Elnlambda_j - 0.5 * self.D * tf.math.log(2 * np.pi) - 0.5 * ElnQ

        p_j = tf.exp(Elnpi_j + lnp_j)
        output_shape = tf.concat([[tf.shape(p_j)[0]], [1], self.F_o], axis=0)
        print(p_j.shape, self.filter.shape, output_shape, self.S, self.K, self.D, self.C, self.P, self.F_o)
        sum_p_j = tf.nn.conv2d_transpose(p_j, self.filter, output_shape=output_shape, strides=[1, self.S, self.S, 1], padding='VALID')
        sum_p_j = tf.image.extract_patches(images=sum_p_j, sizes=[1, self.K, self.K, 1], strides=[1, self.K, self.K, 1], rates=1, padding='VALID')

        return 1. / tf.maximum(sum_p_j, 1e-11) * p_j

    def compute_output(self, a_i, V_ji):
        Elnlambda_j = self.D * 0.5 * tf.math.log(self.nu_j) + self.D * 0.5 - tf.math.lgamma(0.5 * (self.nu_j - tf.range(self.D, dtype=tf.float32))) + self.lndet_Psi_j
        Elnpi_j = tf.math.digamma(self.alpha_j) - tf.math.digamma(tf.reduce_sum(self.alpha_j, axis=2, keepdims=True))
        H_q_j = 0.5 * self.D * tf.math.log(2 * np.pi) - 0.5 * Elnlambda_j

        a_j = - (tf.exp(Elnpi_j) * H_q_j)

        a_j = self.BN_a(a_j)
        a_j = tf.sigmoid(a_j)

        m_j = self.BN_v(self.m_j)
        m_j = tf.reshape(m_j, [-1, self.C, self.P, self.P, *self.F_o])

        return a_j, m_j